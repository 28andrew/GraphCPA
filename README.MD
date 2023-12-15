# GraphCPA

This repository has our code for our model GraphCPA and its variants, created for CPSC 483.

## Usage
- Download checkpoint of pretrained molecule encoder from [here](https://drive.google.com/file/d/191jIh-tTBsihfRDIxFPM_xsXGn1m8yy0/view?usp=share_link) and place into jeffrey/base_checkpoint.
- Download de_train.parquet from [here](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data) and place into the root of the repository.
- For all training and inference besides from inside the `jeffrey` (molecule transformer), create and activate a Conda environment from `environment.yml`. For the variant of the model with the molecule transformer in `jeffrey`, use `transformer_environment.yml`.
### GraphCPA Base
- Use `finetune.py` and `inference.py`.

### GraphCPA + Molecule Transformer
- Use `jeffrey/finetune.py` and `inference.py` for the trainable molecule encoder.
- Use `jeffrey/finetune_fixed.py` and `inference_fixed.py` for the fixed molecule encoder.

### GraphCPA + Gene Relationship GCN
- Use `andrew/finetune.py` and `andrew/inference.py`.

### GraphCPA + Gene Relationship GCN + Chromatin Attention
- Use `bill/finetune.py` and `bill/inference.py`.

## Evaluation
- Upload the csv files from inference to [this Kaggle competition](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data) for results on the private test set.