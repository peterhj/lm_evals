from lm_torch.prelude import gpu, smp, bf16, f16, f32, i64
from lm_torch.mistral import MistralConfig, CachedMistral
from lm_torch.mixtral import MixtralConfig, CachedMixtral
from lm_torch.sentencepiece import SentencePieceTokenizer
from lm_torch.context import Context
from lm_torch.utils import reseed_torch, load_params_to_f32

import torch
from fire import Fire

import ast
from datetime import datetime
from glob import glob
import json
from itertools import repeat
from multiprocessing import Process, Queue
import os
from queue import Empty
import random
import time
from typing import Optional

def now():
    dt = datetime.now()
    d = "{}".format(dt.date())
    t = "{}".format(dt.time())
    return "{}-{}".format(d, t[:8].replace(":", "-"))

def lap():
    torch.cuda.synchronize()
    return time.clock_gettime(time.CLOCK_REALTIME)

def open_gsm8k(split: str):
    data = []
    with open("data/gsm8k/{}.jsonl".format(split)) as lines:
        for line in lines:
            item = json.loads(line.strip())
            q = item["question"].strip()
            long_a = item["answer"].strip()
            short_pos = long_a.find("####")
            assert short_pos >= 0
            short_a = long_a[short_pos+4:].strip()
            try:
                a = int(short_a)
            except ValueError:
                short_a = short_a.replace(",", "")
                a = int(short_a)
            data.append((q, a, long_a))
    return data

class Interp:
    def __init__(self):
        self.code = None
        self.globals = dict()
        self.locals = dict()
        self.failure = False
        self.timeout = False

    @staticmethod
    def from_context(full_ctx_s):
        this = Interp()
        sol_start_pre = full_ctx_s.rfind("<solution>\n")
        assert sol_start_pre >= 0
        sol_start = sol_start_pre + 11
        sol_end = full_ctx_s.rfind("</solution>\n")
        if sol_end < 0:
            sol_end = len(full_ctx_s)
        sol_text = full_ctx_s[sol_start:sol_end].rstrip()
        this.code = "if True:\n" + sol_text
        return this

    def exec(self):
        try:
            exec(self.code, self.globals, self.locals)
        except Exception:
            self.failure = True

def interp_exec_with_timeout(interp):
    def worker(code, queue):
        interp = Interp()
        interp.code = code
        interp.exec()
        queue.put_nowait((interp.code, interp.locals, interp.failure))
    queue = Queue()
    proc = Process(target=worker, args=(interp.code, queue))
    proc.start()
    ex = False
    try:
        proc.join(timeout=2.0)
    except Exception:
        ex = True
    try:
        if proc.exitcode is None:
            ex = True
            proc.terminate()
    except Exception:
        pass
    try:
        code, locals, failure = queue.get_nowait()
        assert interp.code == code
        interp = Interp()
        interp.code = code
        interp.locals = locals
        interp.failure = failure
    except Empty:
        ex = True
        pass
    if ex:
        interp.failure = True
        interp.timeout = True
    return interp

def main(
        MODEL_PATH: str,
        DATA_SPLIT: str,
        model_format: Optional[str] = "safetensors",
        dtype: Optional[str] = "float16",
        device: Optional[str] = "gpu",
        start_idx: Optional[int] = 0,
        end_idx: Optional[int] = None,
        sleep: Optional[float] = None,
        bench: Optional[bool] = False,
):
    main_t0 = lap()
    #NOW = now()

    random.seed(101)
    reseed_torch()

    print("INFO:   model path = {}".format(MODEL_PATH))
    print("INFO:   data split = {}".format(DATA_SPLIT))

    if not (DATA_SPLIT == "train" or DATA_SPLIT == "test"):
        raise ValueError("unsupported value for DATA_SPLIT")

    print("INFO:   start idx = {}".format(start_idx))
    print("INFO:   end idx = {}".format(end_idx))

    sleep_duration = sleep
    print("INFO:   sleep duration = {}".format(sleep_duration))

    batch_size = 1
    max_seq_len = 2048

    model_filename = os.path.basename(MODEL_PATH)
    if model_filename.find("mistral-7b-") == 0:
        cfg = MistralConfig.mistral_7b()
        model_class = CachedMistral
    elif model_filename.find("mixtral-8x7b-") == 0:
        cfg = MixtralConfig.mixtral_8x7b()
        model_class = CachedMixtral
    elif model_filename.find("mixtral-8x22b-") == 0:
        cfg = MixtralConfig.mixtral_8x22b()
        model_class = CachedMixtral
    else:
        raise ValueError("unsupported model")

    print("INFO:   model config = {}".format(cfg))

    tokenizer = SentencePieceTokenizer(os.path.join(MODEL_PATH, "tokenizer.model"))
    assert len(tokenizer) == cfg.tok_dim

    if dtype == "bfloat16" or dtype == "bf16":
        dtype = bf16
    elif dtype == "float16" or dtype == "f16":
        dtype = f16
    elif dtype == "float32" or dtype == "f32":
        dtype = f32
    else:
        raise ValueError("unsupported value for --dtype")
    print("INFO:   dtype = {}".format(dtype))

    if device == "gpu":
        device = gpu
    elif device == "cpu" or device == "smp":
        device = smp
    else:
        raise ValueError("unsupported value for --device")
    print("INFO:   device = {}".format(device))

    if bench:
        pre_init_t0 = lap()

    params = load_params_to_f32(MODEL_PATH, model_format)
    if cfg.linear_scale is not None:
        with torch.device(smp):
            for k in params:
                if k.find("_proj") >= 0 or k.find("_head") >= 0:
                    params[k].data.mul_(cfg.linear_scale)

    if bench:
        pre_init_t1 = lap()

    if bench:
        print("INFO:   bench: model load = {:.06} s".format(pre_init_t1-pre_init_t0))

    with torch.device(device):
        if bench:
            init_t0 = lap()

        gpu_opt_model = torch.nn.utils.skip_init(model_class, cfg, batch_size, max_seq_len, dtype=dtype, device=device)
        gpu_opt_params = dict(gpu_opt_model.named_parameters())
        for k in params:
            if device == smp and dtype == f32:
                gpu_opt_params[k].data = params[k].data
            else:
                gpu_opt_params[k].data.copy_(params[k])
        torch.cuda.empty_cache()

        if bench:
            init_t1 = lap()

    if bench:
        print("INFO:   bench: model init = {:.06} s".format(init_t1-init_t0))

    param_nan_ct = 0
    for k in params:
        param_nan_ct += torch.count_nonzero(gpu_opt_params[k].data.isnan()).to(device=smp)
    if param_nan_ct > 0:
        print("DEBUG:  param nan ct = {}".format(param_nan_ct))
    assert param_nan_ct == 0

    data = open_gsm8k(DATA_SPLIT)
    ndata = len(data)
    print("INFO:   n = {}".format(ndata))
    if end_idx is None:
        end_idx = ndata
    else:
        end_idx = min(end_idx, ndata)

    initial_short_text = (
            "Your task is to translate a given math problem into equivalent and correct Python code. Your Python code should store the final answer in the variable `__answer__`.\n\n"
            "Here are some examples of how you should complete this task:\n\n"
            "<problem> Alice has 3 cookies. Bob has 5 cookies. How many cookies do Alice and Bob have together? </problem>\n"
            "<solution>\n"
            "    cookies_Alice = 3\n"
            "    cookies_Bob = 5\n"
            "    cookies_Alice_and_Bob = cookies_Alice + cookies_Bob\n"
            "    __answer__ = cookies_Alice_and_Bob\n"
            "</solution>\n\n"
            "<problem>"
            )

    initial_text = (
            "Your task is to translate a given math problem into equivalent and correct Python code. Your Python code should store the final answer in the variable `__answer__`.\n\n"
            "Here are some examples of how you should complete this task:\n\n"
            "<problem> Alice has 3 cookies. Bob has 5 cookies. How many cookies do Alice and Bob have together? </problem>\n"
            "<solution>\n"
            "    cookies_Alice = 3\n"
            "    cookies_Bob = 5\n"
            "    cookies_Alice_and_Bob = cookies_Alice + cookies_Bob\n"
            "    __answer__ = cookies_Alice_and_Bob\n"
            "</solution>\n\n"
            "<problem> Today, Carol has eaten only half of the apples she needs. Every day, Carol needs to eat 10 apples to be healthy. How many apples has Carol eaten today? </problem>\n"
            "<solution>\n"
            "    apples_needed = 10\n"
            "    apples_today = apples_needed / 2\n"
            "    __answer__ = apples_today\n"
            "</solution>\n\n"
            "<problem> Daniel has 8 fewer potatoes than Eve. Eve has ten less than twice the potatoes of Daniel and Frank. If Eve has 12 potatoes, how many potatoes does Frank have? </problem>\n"
            "<solution>\n"
            "    potatoes_Eve = 12\n"
            "    potatoes_Daniel = potatoes_Eve - 8\n"
            "    potatoes_Frank = (potatoes_Eve + 10 - potatoes_Daniel * 2) / 2\n"
            "    __answer__ = potatoes_Frank\n"
            "</solution>\n\n"
            "<problem> A class went pressing Ashmead's kernel apples into cider. While making the cider, some of the students drank some of the cider they pressed. Natascha pressed 1.5 liters of cider, and she drank 250 mL. Sasha pressed 2 liters and didn't drink any of the cider. Piotr pressed 3 liters of cider and drank three fourths as much cider as Natascha and Sasha pressed. Anna made twice as much cider as the other three students pressed combined, and she drank none of it. If the teacher asked the students to make 17 liters of cider, how many milliliters of Ashmead's kernel cider are they short? </problem>\n"
            "<solution>\n"
            "    Ashmeads_kernel_cider_Natascha_pressed_mL = 1.5 * 1000\n"
            "    Ashmeads_kernel_cider_Natascha_drank_mL = 250\n"
            "    Ashmeads_kernel_cider_Sasha_pressed_mL = 2 * 1000\n"
            "    Ashmeads_kernel_cider_Piotr_pressed_mL = 3 * 1000\n"
            "    Ashmeads_kernel_cider_Piotr_drank_mL = (Ashmeads_kernel_cider_Natascha_pressed_mL + Ashmeads_kernel_cider_Sasha_pressed_mL) * 3 / 4\n"
            "    Ashmeads_kernel_cider_Anna_pressed_mL = 2 * (Ashmeads_kernel_cider_Natascha_pressed_mL + Ashmeads_kernel_cider_Sasha_pressed_mL + Ashmeads_kernel_cider_Piotr_pressed_mL)\n"
            "    Ashmeads_kernel_cider_teacher_asked_mL = 17 * 1000\n"
            "    Ashmeads_kernel_cider_pressed_mL = Ashmeads_kernel_cider_Natascha_pressed_mL + Ashmeads_kernel_cider_Sasha_pressed_mL + Ashmeads_kernel_cider_Piotr_pressed_mL + Ashmeads_kernel_cider_Anna_pressed_mL\n"
            "    Ashmeads_kernel_cider_drank_mL = Ashmeads_kernel_cider_Natascha_drank_mL + Ashmeads_kernel_cider_Piotr_drank_mL\n"
            "    Ashmeads_kernel_cider_short_mL = Ashmeads_kernel_cider_teacher_asked_mL - (Ashmeads_kernel_cider_pressed_mL - Ashmeads_kernel_cider_drank_mL)\n"
            "    __answer__ = Ashmeads_kernel_cider_short_mL\n"
            "</solution>\n\n"
            "<problem>"
            )

    short_ctx = Context("short_ctx", tokenizer, gpu_opt_model, max_seq_len, device = device)
    long_ctx = Context("long_ctx", tokenizer, gpu_opt_model, max_seq_len, device = device)

    ncorrect_short = 0
    ncorrect = 0
    nvisit_short = 0
    nvisit = 0

    for data_idx in range(start_idx, end_idx):
        print()
        print("INFO:   problem {}/{}".format(data_idx+1, ndata))
        question = data[data_idx][0]
        answer_ref = data[data_idx][1]

        text = (
                " {} </problem>\n"
                "<solution>\n"
               ).format(question)

        short_ctx.rollout([initial_short_text, text], stop = "solution")
        full_str = "".join(short_ctx.buf)
        interp = Interp.from_context(full_str)
        interp = interp_exec_with_timeout(interp)
        print()
        print("DEBUG:  interp: short failure? {}".format(interp.failure))
        print("DEBUG:  interp: short locals = {}".format(interp.locals))
        print()
        correct = False
        if not interp.failure and "__answer__" in interp.locals:
            if answer_ref == interp.locals["__answer__"]:
                correct = True
            # FIXME(HACK)
            elif isinstance(interp.locals["__answer__"], tuple) and len(interp.locals["__answer__"]) > 0 and answer_ref == interp.locals["__answer__"][0]:
                correct = True
        if correct:
            print("INFO:   correct short answer = {} vs {}".format(interp.locals.get("__answer__"), answer_ref))
        else:
            print("INFO:   failure or wrong short answer = {} vs {}".format(interp.locals.get("__answer__"), answer_ref))
        result = dict()
        result["dataset"] = "gsm8k"
        result["data_split"] = DATA_SPLIT
        result["data_idx"] = data_idx
        result["context"] = "short"
        result["correct"] = correct
        if interp.code is not None and len(interp.code) > 0:
            lines = list(interp.code.splitlines())
            if len(lines[-1]) > 0:
                lines.append("")
            result["solution"] = "<solution>\n" + "\n".join(lines[1:]) + "</solution>\n"
        else:
            result["solution"] = None
        if correct:
            ncorrect_short += 1
        nvisit_short += 1
        print("INFO:   running short score = {}/{} ({:.04})".format(ncorrect_short, nvisit_short, ncorrect_short / nvisit_short), flush=True)
        print()
        print("{}".format(json.dumps(result)), flush=True)
        print()

        for _ in range(2):
            retry = False
            long_ctx.rollout([initial_text, text], stop = "solution")
            full_str = "".join(long_ctx.buf)
            interp = Interp.from_context(full_str)
            interp = interp_exec_with_timeout(interp)
            print()
            print("DEBUG:  interp: failure? {}".format(interp.failure))
            print("DEBUG:  interp: locals = {}".format(interp.locals))
            if interp.failure:
                class ParseUses(ast.NodeVisitor):
                    def __init__(self):
                        super().__init__()
                        self.uses = set()
                    def visit_Name(self, node):
                        self.uses.add(node.id)
                sol = interp.code
                lines = sol.splitlines()
                order = list(range(len(lines)))
                done_reorder = False
                for _ in range(20):
                    interp_defs = dict()
                    interp_uses = dict()
                    interp_nondefs = dict()
                    try_reorder = None
                    idx = 1
                    for line_id in order[1:]:
                        line = lines[line_id]
                        src = line.strip()
                        indent = 0
                        for p in range(len(line)):
                            if line[p] == ' ':
                                indent += 1
                            elif line[p] == '\t':
                                indent = (indent // 8 + 1) * 8
                            else:
                                break
                        try:
                            tree = ast.parse(src)
                        except Exception:
                            break
                        lval = tree.body[0].targets[0].id
                        parser = ParseUses()
                        parser.visit(tree.body[0].value)
                        for rval in parser.uses:
                            if rval not in interp_defs:
                                if try_reorder is None:
                                    try_reorder = rval
                                interp_nondefs.setdefault(rval, idx)
                            interp_uses.setdefault(rval, idx)
                        interp_defs[lval] = idx
                        # FIXME: `__answer__ = ...` might get moved.
                        if lval == "__answer__":
                            break
                        idx += 1
                    print("DEBUG:  interp: uses = {}".format(interp_uses))
                    print("DEBUG:  interp: defs = {}".format(interp_defs))
                    for k in interp_defs:
                        if k in interp_nondefs:
                            del interp_nondefs[k]
                    print("DEBUG:  interp: nondefs = {}".format(interp_nondefs))
                    if len(interp_nondefs) > 0:
                        for lval in reversed(interp_nondefs):
                            if lval == "__answer__":
                                continue
                            print()
                            print("INFO:   retry with hint...")
                            text = (
                                    " {} </problem>\n"
                                    "<solution>\n"
                                    "    {} ="
                                   ).format(question, lval)
                            retry = True
                            break
                        break
                    elif try_reorder is None:
                        done_reorder = True
                        break
                    elif try_reorder is not None and try_reorder in interp_defs:
                        rval = try_reorder
                        idx0 = interp_uses[rval]
                        idx1 = interp_defs[rval]
                        for idx in reversed(range(idx0, idx1)):
                            lidx = order[idx]
                            ridx = order[idx+1]
                            order[idx] = ridx
                            order[idx+1] = lidx
                if done_reorder:
                    print()
                    print("INFO:   try rewrite and reexec...", flush=True)
                    sol2_text = "\n".join([lines[line_id] for line_id in order[1:]])
                    if len(sol2_text) > 0 and sol2_text[-1] == '\n':
                        sol2 = "<solution>\n{}</solution>\n".format(sol2_text)
                    else:
                        sol2 = "<solution>\n{}\n</solution>\n".format(sol2_text)
                    print("{}".format(sol2), end="", flush=True)
                    interp = Interp.from_context(sol2)
                    interp = interp_exec_with_timeout(interp)
                    print()
                    print("DEBUG:  interp: failure? {}".format(interp.failure))
                    print("DEBUG:  interp: locals = {}".format(interp.locals))
            if not retry:
                break
        print()
        correct = False
        if not interp.failure and "__answer__" in interp.locals:
            if answer_ref == interp.locals["__answer__"]:
                correct = True
            # FIXME(HACK)
            elif isinstance(interp.locals["__answer__"], tuple) and len(interp.locals["__answer__"]) > 0 and answer_ref == interp.locals["__answer__"][0]:
                correct = True
        if correct:
            ncorrect += 1
            print("INFO:   correct answer = {} vs {}".format(interp.locals.get("__answer__"), answer_ref))
        else:
            print("INFO:   failure or wrong answer = {} vs {}".format(interp.locals.get("__answer__"), answer_ref))
        nvisit += 1
        print("INFO:   running score = {}/{} ({:.04})".format(ncorrect, nvisit, ncorrect / nvisit), flush=True)
        result = dict()
        result["dataset"] = "gsm8k"
        result["data_split"] = DATA_SPLIT
        result["data_idx"] = data_idx
        result["context"] = "long"
        result["correct"] = correct
        if interp.code is not None and len(interp.code) > 0:
            lines = list(interp.code.splitlines())
            if len(lines[-1]) > 0:
                lines.append("")
            result["solution"] = "<solution>\n" + "\n".join(lines[1:]) + "</solution>\n"
        else:
            result["solution"] = None
        print()
        print("{}".format(json.dumps(result)), flush=True)

        if sleep_duration is not None and sleep_duration > 0.0:
            time.sleep(sleep_duration)

    print("INFO:   final score = {}/{} ({:.04}) (short)".format(ncorrect_short, nvisit, ncorrect_short / nvisit))
    print("INFO:   final score = {}/{} ({:.04})".format(ncorrect, nvisit, ncorrect / nvisit))

    main_t1 = lap()
    print("INFO:   final elapsed = {:.06} s".format(main_t1-main_t0))

if __name__ == "__main__":
    Fire(main)
