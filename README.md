# Multimodal analysis of methylomics and fragmentomics in plasma cell-free DNA for multi-cancer early detection and localization

## Overview

This repository contains a pytorch implementation of Graph Convolutional Neural Networks (GCNNs) for the tissue of origin (TOO) prediction task. The details of the method is described in the paper [link](https://elifesciences.org/reviewed-preprints/89083#s2). <br />
The repository includes:
- Source code of GCNNs.
- Training code for GCNNs.
- Datasets employed for the TOO task.
- Checkpoint of GCNNs used in the paper.

## Installation
```
    conda create --name graphCNNs python=3.8.13
    conda activate graphCNNs
    pip install -r requirements.txt
```

## Dataset
The dataset for the TOO task are included in the repository by two files:
- Data used for training: ECD-Data_pan_cancer_paper.csv
- Metadata of data: New Metadata for pan-cancer paper.xlsx

## Training:
To train the GraphCNNs model, simply running <br />
`python train.py` <br />
The output will be the checkpoint saved at `./ckpt/best_ckpt.pt`. If you want to change the saving directory, please change the `self.ckpt_path` in the `config.py` <br />

## Inferencing:
To perform the inference and generate the Sensitivity and Specifity on the Test dataset, simply running <br />
`python infer.py`

## Authors and acknowledgment
Tandoan@genesolutions.vn

