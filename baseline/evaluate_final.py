from collections import defaultdict

from datasets import Dataset

from evaluation.bleu import calc_bleu
from evaluation.tabfact_based import run_evaluation as calc_tabfact_eval


def evaluate_bleu(preds, dataset):
    pred_texts = defaultdict(list)
    refs = defaultdict(list)

    for pred, dataset_item in zip(preds, dataset):
        assert len(pred['out']) == 1  # one generation per dataset instance
        table_id = dataset_item['table_id']
        refs[table_id].append(dataset_item['output'])
        pred_texts[table_id].append(pred['out'][0])

    scores = calc_bleu(pred_texts, refs)
    return scores


def evaluate_with_tapex(preds, dataset):
    eval_dataset_items = []

    for i, pred in enumerate(preds):
        dataset_entry = dataset[i]
        for p in pred['out']:
            eval_dataset_items.append({
                'id': dataset_entry['table_id'],
                'table_csv': dataset_entry['table_csv'],
                'prediction': p
            })

    dataset = Dataset.from_list(eval_dataset_items)
    scores = calc_tabfact_eval(dataset, 'tapex')
    scores.update(calc_tabfact_eval(dataset, 'tapas'))
    return scores


def evaluate_preds(preds, dataset):
    assert len(preds) == len(dataset)
    scores = evaluate_bleu(preds, dataset)
    scores.update(evaluate_with_tapex(preds, dataset))
    return scores
