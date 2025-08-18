<div align="center">


# `NTLoss` - a regression-like loss for LLMs


[![Paper](https://img.shields.io/badge/Paper-ICML-darkgreen.svg)](https://ibm.biz/ntl-paper)
[![Landing](https://img.shields.io/badge/Landing-Page-blue.svg)](https://ibm.biz/ntl-main)
[![Demo](https://img.shields.io/badge/🤗-Demo-yellow.svg)](https://ibm.biz/ntl-demo)
[![CI](https://github.com/AI4SD/number-token-loss/actions/workflows/ci.yaml/badge.svg)](https://github.com/AI4SD/number-token-loss/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ntloss?label=pypi&color=brightgreen)](https://pypi.org/project/ntloss/)
[![Docs](https://github.com/AI4SD/number-token-loss/actions/workflows/docs.yaml/badge.svg)](https://ibm.biz/ntl-docs)
[![Downloads](https://static.pepy.tech/badge/ntloss)](https://pepy.tech/project/ntloss)

*`ntloss` is a PyPI package of the "Number Token Loss" for language models. A regression-like loss that improves LLM performance on math tasks. Follows* **Regress, Don't Guess, ICML 2025**


</div>

---

## 📖 Overview
This repo maintains the code for the `ntloss` [PyPI package](https://pypi.org/project/ntloss/)

- 🧑🏽‍💻 **Paper source code**: [Regress, Don't Guess – ICML 2025](https://ibm.biz/ntl-code)
- 📄 **Paper**: [Regress, Don't Guess – A Regression-like Loss on Number Tokens for Language Models](https://ibm.biz/ntl-paper)
- 🌐 **Project Page**: [Landing Page](https://ibm.biz/ntl-main)
- 🎮 **Demo**: [HuggingFace Spaces Demo (Streamlit)](https://ibm.biz/ntl-demo)
- 📖 **Docs**: [Documentation for the PyPI package](https://ibm.biz/ntl-docs)


## 🏃‍♂️ Quick Start


Simply install `ntloss` into your existing project
```sh
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
  url     = {https://ibm.biz/ntl-main}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.