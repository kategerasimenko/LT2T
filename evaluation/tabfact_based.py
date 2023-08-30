from pathlib import Path
from io import StringIO
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BartForSequenceClassification, TapexTokenizer,
    TapasForSequenceClassification, TapasTokenizer,
    DataCollatorWithPadding
)

from training_utils.utils import set_seed
from input_processing.utils import get_table_id


SEED = 42

ROOT_DIR = str(Path(__file__).parent.parent)

EVAL_PARTS = ['dev', 'test']
INP_COLUMNS = ["label", "input_ids", "attention_mask", "token_type_ids"]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 16


MODELS = {
    'tapas': {
        'model': TapasForSequenceClassification.from_pretrained('google/tapas-large-finetuned-tabfact').to(DEVICE),
        'tokenizer': TapasTokenizer.from_pretrained('google/tapas-large-finetuned-tabfact'),
        'max_length': 512
    },
    'tapex': {
        'model': BartForSequenceClassification.from_pretrained('microsoft/tapex-large-finetuned-tabfact').to(DEVICE),
        'tokenizer': TapexTokenizer.from_pretrained('microsoft/tapex-large-finetuned-tabfact', add_prefix_space=True),
        'max_length': 1024
    }
}


def raw_to_csv(table, dataset):
    table_obj = dataset.prepare_table(table)
    df = dataset.table_to_csv(table_obj)
    return df


def process_dataset(model_name, dataset, tokenizer):
    def process_example(example):
        table_df = pd.read_csv(StringIO(example['table_csv'])).astype(str)
        inp = tokenizer(
            table_df,
            example['prediction'].rstrip('. '),  # for compatibility with tabfact
            max_length=MODELS[model_name]['max_length'],
            truncation=True
        )
        return inp

    res_dataset = dataset.map(process_example, batched=False)
    extra_columns = [col for col in res_dataset.features.keys() if col not in INP_COLUMNS]
    res_dataset = res_dataset.remove_columns(extra_columns)

    return res_dataset


def predict(model_name, dataset):
    set_seed(SEED)

    model = MODELS[model_name]['model']
    tokenizer = MODELS[model_name]['tokenizer']
    collator = DataCollatorWithPadding(tokenizer)

    proc_dataset = process_dataset(model_name, dataset, tokenizer)
    dataloader = DataLoader(proc_dataset, batch_size=BATCH_SIZE, collate_fn=collator)

    all_nli_preds = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            nli_preds = model(**batch).logits.cpu()
            nli_preds = np.argmax(nli_preds, axis=1).tolist()
            all_nli_preds.extend(nli_preds)

    # some predictions are empty in e2e inference
    all_nli_preds = [
        pred if item['prediction'] else 0
        for pred, item in zip(all_nli_preds, dataset)
    ]

    return all_nli_preds


def run_evaluation(dataset, model_name):  # dataset is table_df and prediction
    nli_preds = predict(model_name, dataset)
    dataset = dataset.add_column('nli_pred', nli_preds)

    micro_acc = sum(nli_preds) / len(nli_preds)

    accs_by_id = defaultdict(list)

    for item in dataset:
        accs_by_id[item['id']].append(item['nli_pred'])

    avg_by_id = {k: sum(v) / len(v) for k, v in accs_by_id.items()}
    macro_acc = sum(avg_by_id.values()) / len(avg_by_id)

    return {
        f'{model_name}_micro': micro_acc,
        f'{model_name}_macro': macro_acc
    }


def create_data_from_tg_dataset(tg_dataset, part, preds):
    items = []
    for item, pred in zip(tg_dataset.data[part], preds):
        items.append({
            'id': get_table_id(item),
            'table_csv': raw_to_csv(item, tg_dataset),
            'prediction': pred
        })
    parsed_dataset = Dataset.from_list(items)
    return parsed_dataset


def run_eval_on_tg_dataset(tg_dataset, part, preds, model_name):
    parsed_dataset = create_data_from_tg_dataset(tg_dataset, part, preds)
    nli_acc = run_evaluation(dataset=parsed_dataset, model_name=model_name)
    return nli_acc

