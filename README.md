# Table-to-Text generation via Logical Forms

This repository contains the code for the pipeline-based system for logical table-to-text generation.

## Requirements

* Python >=3.9
* `pip install -r requirements.txt`

## Outputs

`.json` files with the outputs of intermediate steps and final results are in `model_outputs` folder.

## Pre-trained models

* [Content selection](https://drive.google.com/file/d/1NMOh_gmc6QEqJUNf6mjBOkMXSGavVb5p/view?usp=sharing)
* [LF-to-text generation](https://drive.google.com/file/d/1Cj2FJrt_-0Zo2aJzPZa9x1Yd8CsKcn3y/view?usp=sharing)

Archives have pytorch models and json files with processing (if relevant) and generation parameters.

## Run inference

Example for the downloaded models:

```
python end_to_end_inference.py \
    --dataset logicnlg \
    --part test \
    --content-selection-model content_selection_model \
    --content-selection-generation content_selection_gen_params.json \
    --lf-to-text-model lf_to_text_model \
    --lf-to-text-generation lf_to_text_gen_params.json \
    --predictions-dir predictions_logicnlg_test \
    --batch-size 64 \
    --selection all
```
