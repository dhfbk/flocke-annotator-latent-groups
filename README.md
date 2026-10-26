# Annotators of a Feather FLOCKE Together: A Framework to Analyze Latent Group Patterns Based on Annotator Behavior

This is the code repository for the paper "Annotators of a Feather FLOCKE Together: A Framework to Analyze Latent Group Patterns Based on Annotator Behavior" by Giovanni Valer, Elisa Leonardelli, Camilla Casula, Sara Tonelli, and Michele Lombardi (EMNLP 2026).

[![PDF](https://img.shields.io/badge/PDF-Read-red?style=flat&logo=googledocs&logoColor=white)](https://aclanthology.org/2026.emnlp-main.2665.pdf)
[![PyPI](https://img.shields.io/badge/PyPI-flocke-blue?style=flat&logo=pypi&logoColor=white)](https://pypi.org/project/flocke/)
[![Citation](https://img.shields.io/badge/Citation-EMNLP_2026-orange?style=flat&logo=google-scholar&logoColor=white)](#citation)

---

FLOCKE is a framework for datasets with disaggregated annotations (it is available as a Python package: [`flocke`](https://pypi.org/project/flocke/)). It learns the task together with a small number of **latent annotator groups**: groups of people who label in a similar way. The groups are found from labeling behavior alone, without relying on socio-demographics.


## Tutorial Notebooks
Two short tutorials in [`notebooks/`](notebooks/):
- [`01_using_flocke.ipynb`](notebooks/01_using_flocke.ipynb): apply FLOCKE to any disaggregated annotations, choose the number of groups, inspect the groups, and evaluate with your metrics.
- [`02_validation_framework.ipynb`](notebooks/02_validation_framework.ipynb): use the synthetic HateCheck benchmark to evaluate any method.


## Reproducing the Paper's Experiments
This repository contains the code around FLOCKE for reproducing the paper's experiments.

### Data
Follow the instructions in the [data/README.md](data/README.md) file to download and prepare the datasets. In short:

```bash
python data/download_data.py
```

This fetches **HateCheck** (from Hugging Face) and **MultiPICo** (from the LeWiDi repository), both pinned to a fixed revision. **MD-Agreement** is not publicly available: request it through the [MD-Agreement repository](https://github.com/dhfbk/annotators-agreement-dataset/) and place the files as follows:

```
data/
  hatecheck/test.csv
  multipico/MP_train.json, MP_annotators_meta.json, ...
  MD/MD_train.json, MD_dev.json, MD_test.json
```

### 🪶 Preparation
The code is based on Python **3.12.11**. Using your preferred package manager, create a virtual environment, e.g., with uv:
```bash
uv sync
source .venv/bin/activate
```

Alternatively, you can install the dependencies using pip:
```bash
pip install -r requirements.txt
```

> [!NOTE]
> `pyproject.toml` installs the CUDA 12.6 build of PyTorch. FLOCKE itself is lightweight and can also train and run on CPU.


### 🐣 Validation
```bash
python validation.py
```
On real data there is no ground truth for annotator groups, so this script builds a synthetic benchmark (based on [HateCheck](https://huggingface.co/datasets/Paul/hatecheck)) where the groups are known, and checks whether FLOCKE can recover them.


### 🦜 Experiments on Real Data
```bash
python experiments.py
```

This script applies FLOCKE to two binary classification datasets with disaggregated annotations, where the groups are unknown:

- **[MD-Agreement](https://github.com/dhfbk/annotators-agreement-dataset/)**: offensiveness detection on 10,753 tweets, with 5 annotations per item from 819 annotators.
- **[MultiPICo](https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/MultiPICo)**: irony detection on post-reply pairs, restricted here to English subset (British, American and Australian annotators). It also includes the socio-demographic traits of the annotators.

## Citation
If you use this code in your research, please cite the following paper:
```bibtex
@inproceedings{valer-etal-2026-annotators,
    title = "Annotators of a Feather {FLOCKE} Together: A Framework to Analyze Latent Group Patterns Based on Annotator Behavior",
    author = "Valer, Giovanni  and Leonardelli, Elisa  and Casula, Camilla  and Tonelli, Sara  and Lombardi, Michele",
    editor = "Lee, Hung-Yi  and Dev, Sunipa  and Li, Jessy  and Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.2665/",
    pages = "60662--60676",
    ISBN = "979-8-89176-459-0",
}
```
