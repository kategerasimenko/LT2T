import os
import json
from pathlib import Path
from collections import defaultdict

import nltk
import click
import evaluate
from tqdm import tqdm
from tabgenie import load_dataset


ROOT_DIR = str(Path(__file__).parent.parent)

SACREBLEU = evaluate.load("sacrebleu")
BLEU_WEIGHTS = {
    'bleu_1': (1, 0, 0),
    'bleu_2': (0.5, 0.5, 0),
    'bleu_3': (0.33, 0.33, 0.33)
}


def collect_preds_and_refs(dataset):
    preds = defaultdict(list)
    refs = defaultdict(list)

    for item in dataset:
        refs[item['table_id']].append(item['ref'])  # todo: it's a field from raw dataset
        preds[item['table_id']].append(item['prediction'])

    return preds, refs


def calc_nltk_bleu(predictions, references):
    bleu_scores = defaultdict(list)

    for table_id, refs in tqdm(references.items()):
        refs = [r.lower().split(' ') for r in refs]

        for pred in predictions[table_id]:
            pred = pred.lower().split(' ')

            for n, weights in BLEU_WEIGHTS.items():
                bleu_n = nltk.translate.bleu_score.sentence_bleu(refs, pred, weights=weights)
                bleu_scores[n].append(bleu_n)

    for n, scores in bleu_scores.items():
        bleu_scores[n] = sum(scores) / len(scores)

    return bleu_scores


def calc_sacrebleu(predictions, references):
    pred_list = []
    ref_list = []

    for table_id, refs in references.items():
        for pred in predictions[table_id]:
            ref_list.append(refs)
            pred_list.append(pred)

    # padding references
    max_ref_len = max(len(x) for x in ref_list)
    min_ref_len = min(len(x) for x in ref_list)
    if max_ref_len == 0:
        return None

    print(f'Number of refs for one table: max {max_ref_len}, min {min_ref_len}')

    ref_list = [x + ['' for _ in range(max_ref_len - len(x))] for x in ref_list]

    # todo: no option for whitespace tokenization
    result = SACREBLEU.compute(predictions=pred_list, references=ref_list)

    return result["score"]


def calc_bleu(predictions, references):
    scores = calc_nltk_bleu(predictions, references)
    scores['sacrebleu'] = calc_sacrebleu(predictions, references)
    return scores


def calc_bleu_for_dataset(dataset):
    predictions, references = collect_preds_and_refs(dataset)
    scores = calc_bleu(predictions, references)
    return scores


def run_eval_on_tg_dataset(tg_dataset, part, preds):
    tg_dataset.data[part] = tg_dataset.data[part].add_column('prediction', preds)
    bleu_scores = calc_bleu_for_dataset(tg_dataset.data[part])
    return bleu_scores


@click.command()
@click.option("--dataset", "-d", required=True, type=str, help="Dataset to run evaluation on")
@click.option("--part", "-p", required=True, type=str, help="Part for evaluadation (dev or test).")
@click.option("--predictions-file", "-o", required=True, type=str, help="jsonl file with predictions.")
def main(dataset, part, predictions_file):
    print(f'Evaluating {dataset} {part} with BLEU ({predictions_file})...')

    if not (predictions_file.startswith('/') or predictions_file.startswith('~')):  # absolute paths
        predictions_file = os.path.join(ROOT_DIR, predictions_file)

    with open(predictions_file) as f:
        preds = [json.loads(line)['out'][0] for line in f.readlines()]

    tg_dataset = load_dataset(dataset)
    bleu_scores = run_eval_on_tg_dataset(tg_dataset, part, preds)

    for n, bleu in bleu_scores.items():
        print(f'{dataset} {part} {n}: {bleu}')


if __name__ == '__main__':
    main()
