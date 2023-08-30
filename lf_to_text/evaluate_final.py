from evaluation.bleu import calc_bleu
from evaluation.blec import calc_blec


def evaluate_preds(preds, dataset):
    assert len(preds) == len(dataset)

    pred_texts = {}
    refs = {}
    lfs = {}

    for i, pred_group in enumerate(preds):
        dataset_item = dataset[i]
        item_id = f"{dataset_item['table_id']}_{dataset_item['lf']}"
        pred_texts[item_id] = pred_group['out']  # can be several from sampling
        refs[item_id] = [dataset_item['output']]  # only one ref
        lfs[item_id] = dataset_item['lf']

    # must be one prediction group per dataset entry
    assert len(pred_texts) == len(dataset)

    scores = calc_bleu(pred_texts, refs)
    scores['blec'] = calc_blec(pred_texts, refs, lfs)
    return scores
