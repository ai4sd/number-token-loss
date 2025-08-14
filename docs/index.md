---
title: Number Token Loss
---

# Number Token Loss

<p align="center">
  <a href="https://arxiv.org/abs/2411.02083">
    <img src="https://img.shields.io/badge/Paper-ICML%202025-darkgreen.svg" alt="Paper">
  </a>
  <a href="https://tum-ai.github.io/number-token-loss/">
    <img src="https://img.shields.io/badge/Landing-Page-blue.svg" alt="Landing Page">
  </a>
  <a href="https://huggingface.co/spaces/jannisborn/NumberTokenLoss">
    <img src="https://img.shields.io/badge/🤗-Demo-yellow.svg" alt="Demo">
  </a>
  <a href="https://github.com/AI4SD/number-token-loss/actions/workflows/ci.yaml">
    <img src="https://github.com/AI4SD/number-token-loss/actions/workflows/ci.yaml/badge.svg" alt="CI">
  </a>
  <a href="https://ai4sd.github.io/number-token-loss/">
    <img src="https://github.com/AI4SD/number-token-loss/actions/workflows/docs.yaml/badge.svg" alt="CI">
  </a>
  <a href="https://badge.fury.io/py/ntloss">
    <img src="https://badge.fury.io/py/ntloss.svg" alt="PyPI">
  </a>
  <a href="https://pepy.tech/project/ntloss">
    <img src="https://static.pepy.tech/badge/ntloss" alt="Downloads">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
</p>

<div align="center">
<em>A regression-like loss that improves numerical reasoning in language models.</em><br>
<em>Originally presented in <a href="https://arxiv.org/abs/2411.02083">“Regress, Don’t Guess” (ICML 2025)</a>.</em>
</div>

---

## Getting Started

Install from PyPI:

```bash
uv add ntloss
pip install ntloss # if you are oldschool
```

Use like this:
```py
from ntloss import NTLoss
ntl_fn = NTLoss(tokenizer=tokenizer)
ntl = ntl_fn(logits, labels)

# We recommend
loss = cross_entropy(logits, labels) + 0.3 * ntl
```

`ntloss` is currently in alpha phase and pre-release. Feedback & PRs are very welcome.


## 📝 Citation

If you use `ntloss`, please cite our paper:

```bibtex
@inproceedings{zausinger2025regress,
  title   = {Regress, Don't Guess – A Regression-like Loss on Number Tokens for Language Models},
  author  = {Jonas Zausinger and Lars Pennig and Anamarija Kozina and Sean Sdahl
             and Julian Sikora and Adrian Dendorfer and Timofey Kuznetsov
             and Mohamad Hagog and Nina Wiedemann and Kacper Chlodny
             and Vincent Limbach and Anna Ketteler and Thorben Prein
             and Vishwa Mohan Singh and Michael Danziger and Jannis Born},
  booktitle = {Proc. of the 42nd International Conference on Machine Learning (ICML)},
  year    = {2025},
  url     = {https://tum-ai.github.io/number-token-loss/}
}
```
