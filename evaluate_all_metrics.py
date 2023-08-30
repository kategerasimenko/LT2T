import os
import json

import click
from tabgenie import load_dataset

from input_processing.utils import get_table_id, replace_refs_with_tabfact
from evaluation.tabfact_based import run_eval_on_tg_dataset as tabfact_eval
from evaluation.bleu import run_eval_on_tg_dataset as bleu_eval


def filter_unique(preds, dataset):
    idxs_to_keep = set()
    seen = set()

    for i, item in enumerate(preds):
        table_id = get_table_id(dataset[i])
        if (table_id, item) not in seen:
            idxs_to_keep.add(i)
            seen.add((table_id, item))

    return idxs_to_keep


@click.command()
@click.option("--dataset", required=True, type=str, help="Dataset name to run evaluation on")
@click.option("--part", required=True, type=str, help="Dataset part to run evaluation on")
@click.option("--references", default="original", type=str, help="References to train on (can be `tabfact` for logicnlg). Default: original.")
@click.option("--scores", required=True, type=str, help="Scores to calculate. Options: 'tapas', 'tapex', 'bleu'.")
@click.option("--predictions-file", required=True, type=str, help="jsonl file with predictions.")
@click.option("--scores-file", type=str, help="Path to output file with scores.")
@click.option("--do-random-experiment", is_flag=True, help="Whether to run experiment with scoring random predictions.")
def main(dataset, part, references, scores, predictions_file, scores_file, do_random_experiment):
    scores = scores.split(',')
    score_vals = {}

    tg_dataset = load_dataset(dataset)
    if dataset == 'logicnlg' and references == 'tabfact':
        tg_dataset = replace_refs_with_tabfact(tg_dataset)

    if predictions_file.endswith('.jsonl'):
        with open(predictions_file) as f:
            preds = [json.loads(line)['out'][0] for line in f.readlines()]

    elif predictions_file.endswith('.json'):
        with open(predictions_file) as f:
            raw_preds = json.load(f)
            # relying on the fact that they go sequentially
            preds = [vv for v in raw_preds.values() for vv in v]

    else:
        raise ValueError

    if do_random_experiment:
        import random
        random.shuffle(preds)

    for score_name in scores:
        if score_name in ['tapas', 'tapex']:
            nli_scores = tabfact_eval(tg_dataset, part, preds, score_name)
            for name, nli_score in nli_scores.items():
                score_vals[name] = nli_score
                print(name, nli_score)

        elif score_name == 'bleu':
            bleu_scores = bleu_eval(tg_dataset, part, preds)
            for n, bleu_score in bleu_scores.items():
                score_vals[n] = bleu_score
                print(n, bleu_score)

    if do_random_experiment:
        return  # do not write to file

    if scores_file is None:
        full_pred_path = os.path.abspath(predictions_file)
        only_pred_name = os.path.basename(predictions_file).rsplit('.', 1)[0]
        scores_file = os.path.join(os.path.dirname(full_pred_path), f'scores_{only_pred_name}.json')

    with open(scores_file, 'w') as f:
        json.dump(score_vals, f, indent=2)


if __name__ == '__main__':
    main()
