This repo currently contains an eval script for Mistral/Mixtral on GSM8K.

Our methodology is based on in-context code, aided by the Python interpreter.
We find that a longer context with Python code can improve pass@1 by ~10%
from a baseline of ~40%. Python interpretation then adds a few extra percent
to get our pass@1 to 53.2%, which is higher than the 52.2% maj@8 originally
reported in the Mistral 7B arXiv preprint.

| version                      | pass@1 (or similar)   | maj@8 (or similar)    |
| ---------------------------- | --------------------- | --------------------- |
| Mistral 7B (arXiv preprint)  |                       | 52.2%                 |
| Mistral 7B (8x22B blog post) | 36.5% (maj@1, 5-shot) | 50.0% (maj@8, 8-shot) |
| Mistral 7B (this repo)       | **53.2%**             |                       |
