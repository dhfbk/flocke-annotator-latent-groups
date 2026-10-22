# Data

Our experiments use external dataset:
- HateCheck (https://huggingface.co/datasets/Paul/hatecheck)
- MD-Agreement (https://github.com/dhfbk/annotators-agreement-dataset/)
- MultiPICo (https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/MultiPICo)

The [`download_data.py`](https://github.com/jo-valer/flocke-annotator-latent-groups/blob/main/data/download_data.py) script can be used to download the datasets in the `data/` directory.

> [!IMPORTANT]
> The script does not download the MD-Agreement dataset, which is not publicly available. Please follow the instructions in the [MD-Agreement repository](https://github.com/dhfbk/annotators-agreement-dataset/) to obtain the dataset and place it in the `data/` directory.