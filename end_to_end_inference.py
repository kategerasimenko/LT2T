import os
import json
import random
import pickle
from collections import defaultdict

import click
import numpy as np
from tqdm import tqdm
from datasets import Dataset

import content_selection
import lf_to_text
from config import ROOT_DIR
from input_processing.parse_dataset import parse_tables
from template_filling import fill_template
from training_utils.seq2seq_training import run_inference
from training_utils.utils import set_seed
from evaluation.bleu import calc_bleu
from evaluation.tabfact_based import run_evaluation as calc_tabfact_eval


MAX_OPTIONS_PER_TEMPLATE = 5
EVAL_ITERATIONS = 10
SEED = 42


def run_content_selection(parsed_tables, split, model, generation_params, batch_size, raw_preds_path):
    with open(os.path.join(model, 'processing_params.json')) as f:
        processing_params = json.load(f)

    os.makedirs(raw_preds_path, exist_ok=True)

    content_selection_data = content_selection.get_training_data(
        parsed_tables,
        mode='predict',
        splits=[split],
        processing_params=processing_params
    )
    preds, _ = run_inference(
        ds=content_selection_data[split],
        model_path=os.path.join(model, 'model'),
        data_part=split,
        batch_size=batch_size,
        generation_params=generation_params,
        postprocessing_func=content_selection.fix_tokenization,
        eval_func=content_selection.evaluate_preds,  # will evaluate only executability and variability
        eval_kwargs={'parsed_tables': parsed_tables},
        preds_path=raw_preds_path
    )

    preds_by_table = {
        content_selection_data[split][i]['table_id']: sorted(set(pred['out']))  # no repeats in generation
        for i, pred in enumerate(preds)
    }
    assert len(preds_by_table) == len(preds)  # preds must be for unique tables

    return preds_by_table


def run_template_filling(predicted_templates, parsed_tables):
    filled_templates = {}

    for table_id, pred_template_group in tqdm(predicted_templates.items(), total=len(predicted_templates)):
        table_info = parsed_tables[table_id]
        filled_templates[table_id] = []

        for template in pred_template_group:
            try:  # some formulas are malformed in a complex way and end up with exception
                filled = fill_template(template, table_info['table_obj'], table_info['coltypes'])
                filled = filled.lfs
            except Exception as e:
                print(table_id, template)
                print(e)
                print()
                filled = []

            filled_templates[table_id].append(filled)

    return filled_templates


def run_lf_to_text(filled_templates, parsed_tables, model, generation_params, split, batch_size, raw_preds_path):
    data = lf_to_text.prepare_lf2text_data(
        parsed_tables,
        splits=[split],
        mode='predict',
        logical_forms=filled_templates
    )

    os.makedirs(raw_preds_path, exist_ok=True)

    # no eval func because both BLEU and BLEC require refs which we don't have
    # at least in this setup (and at all for LogicNLG)
    preds, _ = run_inference(
        ds=data[split],
        model_path=os.path.join(model, 'model'),
        data_part=split,
        batch_size=batch_size,
        generation_params=generation_params,
        preds_path=raw_preds_path
    )

    preds_by_id = defaultdict(list)
    item_idx = 0

    for table_id, lf_groups in filled_templates.items():
        for lfs in lf_groups:
            n_lfs = len(lfs)
            lf_preds = [x['out'][0] for x in preds[item_idx:item_idx+n_lfs]]
            preds_by_id[table_id].append(lf_preds)
            item_idx += n_lfs

    return preds_by_id


def select_outputs(predictions, selection_mode, seed=SEED):
    set_seed(seed)
    selected_predictions = defaultdict(list)  # creating new object

    if selection_mode == 'all':
        max_options = None
    elif selection_mode == 'several':
        max_options = MAX_OPTIONS_PER_TEMPLATE
    elif selection_mode in ['single', 'one']:
        max_options = 1
    else:
        raise ValueError(f'Unknown selection mode: {selection_mode}.')

    for table_id, pred_groups in predictions.items():
        selected = []

        for preds in pred_groups:
            if max_options is not None and len(preds) > max_options:
                preds = random.sample(preds, max_options)
            selected.extend(preds)

        if selection_mode == 'one' and selected:
            selected = [random.choice(selected)]

        selected_predictions[table_id] = selected

    return selected_predictions


def evaluate_pred_stats(predictions, nested=False):
    if nested:  # flatten
        predictions = {k: [lf for lfs in v for lf in lfs] for k, v in predictions.items()}

    total_preds = sum(len(v) for v in predictions.values())
    return {
        'total_preds': total_preds,
        'avg_preds_per_table': total_preds / len(predictions),
        'tables_without_preds': len([v for v in predictions.values() if not v])
    }


def evaluate_bleu(parsed_tables, predictions):
    preds = {
        table_id: p if p else ['']  # making BLEU score 0 for empty option sets
        for table_id, p in predictions.items()
    }
    refs = {
        table_id: table_info['references']
        for table_id, table_info in parsed_tables.items()
    }
    scores = calc_bleu(preds, refs)
    return scores


def evaluate_with_tapex(parsed_tables, predictions):
    dataset_items = []
    for table_id, preds in predictions.items():
        if not preds:
            preds = ['']

        for pred in preds:
            dataset_items.append({
                'id': table_id,
                'table_csv': parsed_tables[table_id]['table_csv'],
                'prediction': pred
            })

    dataset = Dataset.from_list(dataset_items)

    print(dataset)
    print(dataset['prediction'][0])

    scores = calc_tabfact_eval(dataset, 'tapex')
    scores.update(calc_tabfact_eval(dataset, 'tapas'))
    return scores


def run_evaluation(parsed_tables, predictions):
    scores = evaluate_pred_stats(predictions)

    tapex_scores = evaluate_with_tapex(parsed_tables, predictions)
    scores.update(tapex_scores)

    bleu_scores = evaluate_bleu(parsed_tables, predictions)
    scores.update(bleu_scores)

    return scores


def run_selection_and_evaluation(parsed_tables, predictions_dir, full_texts, selection_mode):
    n_runs = 1 if selection_mode == 'all' else 10
    runs_scores = defaultdict(list)

    for i in range(n_runs):
        selected_texts = select_outputs(full_texts, selection_mode=selection_mode, seed=SEED+i)
        if not i:  # save first selection
            with open(os.path.join(predictions_dir, f'e2e_selected_{selection_mode}.json'), 'w') as f:
                json.dump(selected_texts, f, indent=2, ensure_ascii=False)

        run_score = run_evaluation(parsed_tables, selected_texts)
        for metric, score in run_score.items():
            runs_scores[metric].append(score)

    runs_scores_avg = {
        metric:  {'avg': np.mean(scores), 'std': np.std(scores)}
        for metric, scores in runs_scores.items()
    }
    return runs_scores_avg


@click.command()
@click.option("--dataset", required=True, type=str, help="Dataset to train on")
@click.option("--part", required=True, type=str, help="Part for evaluadation (dev or test).")
@click.option("--parsed-tables-path", type=str, help="Path to parsed tables")
@click.option("--content-selection-model", required=True, type=str, help="Path to model for content selection.")
@click.option("--content-selection-generation", required=True, type=str, help="Path to generation parameters for content selection.")
@click.option("--lf-to-text-model", required=True, type=str, help="Path to model for lf2text.")
@click.option("--lf-to-text-generation", required=True, type=str, help="Path to generation parameters for lf2text.")
@click.option("--batch-size", default=16, type=int, help="Batch size for model inference.")
@click.option("--predictions-dir", default=None, type=str, help="Directory to write predictions into.")
@click.option("--eval-only", is_flag=True, help="Whether to do only evaluation on existing predictions.")
@click.option("--selection", type=str, default='single', help="Sample selection per table: all, several, single, one.")
def main(
        dataset, part, parsed_tables_path,
        content_selection_model, content_selection_generation,
        lf_to_text_model, lf_to_text_generation, batch_size,
        predictions_dir, eval_only, selection
):
    set_seed(SEED)

    if predictions_dir is None:
        # todo: no generation params here either
        params_str = f'{content_selection_model.rsplit("/", 1)[-1]}_{lf_to_text_model.rsplit("/", 1)[-1]}'
        predictions_dir = os.path.join(ROOT_DIR, 'predictions', f'e2e_{params_str}')
    os.makedirs(predictions_dir, exist_ok=True)
    score_file = os.path.join(predictions_dir, f'scores_{selection}.json')

    if parsed_tables_path is not None:
        with open(parsed_tables_path, 'rb') as f:
            parsed_tables = pickle.load(f)
    else:
        parsed_tables = parse_tables(dataset, splits=[part])

    if not eval_only:
        with open(os.path.join(content_selection_model, content_selection_generation)) as f:
            content_selection_generation_params = json.load(f)

        with open(os.path.join(lf_to_text_model, lf_to_text_generation)) as f:
            lf_to_text_generation_params = json.load(f)

        content_selection_outputs = run_content_selection(
            parsed_tables,
            split=part,
            model=content_selection_model,
            generation_params=content_selection_generation_params,
            batch_size=batch_size,
            raw_preds_path=os.path.join(predictions_dir, 'content_selection_raw')
        )
        with open(os.path.join(predictions_dir, 'content_selection.json'), 'w') as f:
            json.dump(content_selection_outputs, f, indent=2, ensure_ascii=False)

        filled_templates = run_template_filling(content_selection_outputs, parsed_tables)
        with open(os.path.join(predictions_dir, 'filled_templates.json'), 'w') as f:
            json.dump(filled_templates, f, indent=2, ensure_ascii=False)

        full_texts = run_lf_to_text(
            filled_templates,
            parsed_tables,
            split=part,
            model=lf_to_text_model,
            generation_params=lf_to_text_generation_params,
            batch_size=batch_size,
            raw_preds_path=os.path.join(predictions_dir, 'lf_to_text_raw')
        )

        with open(os.path.join(predictions_dir, 'e2e_full.json'), 'w') as f:
            json.dump(full_texts, f, indent=2, ensure_ascii=False)

    else:
        with open(os.path.join(predictions_dir, 'e2e_full.json')) as f:
            full_texts = json.load(f)

    full_stats = evaluate_pred_stats(full_texts, nested=True)

    scores = run_selection_and_evaluation(
        parsed_tables=parsed_tables,
        predictions_dir=predictions_dir,
        full_texts=full_texts,
        selection_mode=selection
    )

    scores.update({
        f'full_{k}': {'avg': v, 'std': 0}
        for k, v in full_stats.items()
    })

    for score_name, score in scores.items():
        print(score_name, round(score['avg'], 4), round(score['std'], 4))

    with open(score_file, 'w') as f:
        json.dump(scores, f, indent=2)


if __name__ == '__main__':
    main()
