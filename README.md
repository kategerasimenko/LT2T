# Table-to-Text generation via Logical Forms

This repository contains the code for the pipeline-based system for logical table-to-text generation.

## Requirements

* Python >=3.9
* `pip install -r requirements.txt`

Training and inference were conducted on A100 GPU card.

Datasets are mostly taken from [`tabgenie` package](https://github.com/kasnerz/tabgenie), which provides data-to-text datasets in a unified form.

## Outputs

`.json` files with the outputs of intermediate steps and final results are in `model_outputs` folder.

## Pre-trained models

Pre-trained models are available via Huggingface Hub:
* Content selection: `kategaranina/lt2t_content_selection`
* LF-to-text generation: `kategaranina/lt2t_lf_to_text`

Preprocessing and generation configurations are stored in `inference_config` folder in this repository and used during inference.

## Inference and evaluation

For running inference on LogicNLG, download the original TabFact dataset from [the original repository](https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/master/collected_data/r2_training_all.json) and put it in `data/LogicNLG` folder.

Inference and evaluation on test set of LogicNLG, using HF models and predefined processing and generation configurations:

```
python end_to_end_inference.py \
    --dataset logicnlg \
    --part test \
    --predictions-dir predictions_logicnlg_test \
    --batch-size 64 \
    --selection all
```

For further parametrization, refer to `end_to_end_inference.py`.


## Training

For running training, Neptune account and credentials (`NEPTUNE_ORG` and `NEPTUNE_API_TOKEN`) are required.

By default, checkpoints are saved to `TMP_DIR`, if set, or to repository directory. 
The final model is saved to the repository directory, with predictions and metrics values inside the model folder. 

For running training of our system steps, preprocessed tables are required.
Run preprocessing with the following command:

```
python parse_dataset.py --dataset logic2text
```

The `.pkl` file with preprocessed data will be saved to `data` folder.

#### Examples

Baseline:

```
python train_baseline.py \
    --do-train \
    --do-predict \
    --dataset=logicnlg \
    --linearize-style=nl \
    --references=tabfact \
    --epochs=20 \
    --base-model=t5-base \
    --batch-size=16 \
    --num-beams=3
```

Content selection:
```
python train_content_selection.py \
    --do-train \
    --do-predict \
    --sources=main \
    --include-stats \
    --include-num-stats \
    --include-value \
    --epochs=20 \
    --base-model=t5-base \
    --batch-size=8 \
    --num-beams=1 \
    --do-sample \
    --top-k=50 \
    --n-generated=5
```

LF-to-text:
```
python train_lf_to_text.py \
    --do-train \
    --do-predict \
    --epochs=30 \
    --base-model=t5-base \
    --learning-rate=2e-5 \
    --batch-size=8 \
    --num-beams=3 \
    --n-generated=1
```

## Contact

Please open an issue in case of any questions, requests, or comments.