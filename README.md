<div align="center">


# `NTLoss` - a regression-like loss for LLMs


[![Paper](https://img.shields.io/badge/Paper-ICML-darkgreen.svg)](https://arxiv.org/abs/2411.02083)
[![Landing](https://img.shields.io/badge/Landing-Page-blue.svg)](https://tum-ai.github.io/number-token-loss/)
[![Demo](https://img.shields.io/badge/🤗-Demo-yellow.svg)](https://huggingface.co/spaces/jannisborn/NumberTokenLoss)
[![CI](https://github.com/AI4SD/number-token-loss/actions/workflows/ci.yaml/badge.svg)](https://github.com/AI4SD/number-token-loss/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://badge.fury.io/py/ntloss.svg)](https://badge.fury.io/py/ntloss)
[![Docs](https://github.com/AI4SD/number-token-loss/actions/workflows/docs.yaml/badge.svg)](https://ai4sd.github.io/number-token-loss/)
[![Downloads](https://static.pepy.tech/badge/ntloss)](https://pepy.tech/project/ntloss)

*`ntloss` is a PyPI package of the "Number Token Loss" for language models. A regression-like loss that improves LLM performance on math tasks. Follows [Regress, Don't Guess, ICML 2025](https://arxiv.org/abs/2411.02083)*


</div>

---

## 📖 Overview
This repo maintains the code for the `ntloss` [PyPI package](https://pypi.org/project/ntloss/)

- 📄 **Paper source code**: [Regress, Don't Guess – ICML 2025](https://github.com/tum-ai/number-token-loss)
- 📄 **Paper**: [Regress, Don't Guess – A Regression-like Loss on Number Tokens for Language Models](https://arxiv.org/abs/2411.02083)
- 🌐 **Project Page**: [https://tum-ai.github.io/number-token-loss/](https://tum-ai.github.io/number-token-loss/)
- 🎮 **Interactive Demo**: [https://huggingface.co/spaces/jannisborn/NumberTokenLoss](https://huggingface.co/spaces/jannisborn/NumberTokenLoss)

## 🏃‍♂️ Quick Start


Simply install `ntloss` into your existing project
```sh
uv add ntloss
pip install ntloss # if you are oldschool
```

Use like this:
```py
from ntloss import NTLoss as NTL
ntl = NTL(tokenizer=tokenizer)
loss = ntl(logits, labels)
```

NOTE: `ntloss` is currently in alpha phase and pre-release. Feedback & PRs are very welcome.


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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.