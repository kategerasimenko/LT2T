from evaluation.bleu import calc_bleu
from input_processing.str2form import parse_str, linearize_lf
from template_filling import fill_template
from .data_processing.get_lf_templates import replace_with_placeholders


def evaluate_bleu(preds, dataset):
    refs = {item['table_id']: item['outputs'] for item in dataset}
    pred_texts = {
        dataset_item['table_id']: pred['out']  # multiple preds per table
        for pred, dataset_item in zip(preds, dataset)
    }
    scores = calc_bleu(pred_texts, refs)

    return scores


def calculate_variability(lst):
    if len(lst) == 1:
        return 1

    return (len(set(lst)) - 1) / (len(lst) - 1)


def evaluate_table_variability(preds_per_table):
    if not preds_per_table:
        return 0

    table_variability_scores = [
        calculate_variability(pred)
        for pred in preds_per_table
    ]
    table_variability = sum(table_variability_scores) / len(table_variability_scores)
    return table_variability


def evaluate_executability_and_variability(preds, dataset, parsed_tables):
    texts = []
    templates = []
    syntax_ok = 0
    executable = 0
    total = 0

    for i, table_preds in enumerate(preds):
        table_texts = []
        table_templates = []
        table_id = dataset[i]['table_id']
        table_info = parsed_tables[table_id]
        table_obj = table_info['table_obj']
        header = [x.value.strip().lower() for x in table_obj.get_cells()[0]]

        for j, pred in enumerate(table_preds['out']):
            total += 1
            table_texts.append(pred)

            try:
                parsed = parse_str(pred, func_map={})
                filling_options = fill_template(pred, table_obj, table_info['coltypes'])
            except:  # fill_template is tested on all training examples -> if error, example is malformed
                continue

            if filling_options.syntax_ok:
                syntax_ok += 1

            if filling_options.lfs:
                executable += 1

                template = replace_with_placeholders(header, parsed, replace_col_names=True)
                template = linearize_lf(template)
                table_templates.append(template)

        texts.append(table_texts)
        templates.append(table_templates)

    executability = executable / total
    syntax = syntax_ok / total

    table_text_variability = evaluate_table_variability(texts)
    table_template_variability = evaluate_table_variability(templates)

    flat_templates = [p for table_preds in templates for p in table_preds]
    n_corpus_templates = len(set(flat_templates))

    exec_var_mean = (n_corpus_templates + executability * 100) / 2

    return {
        'syntax': round(syntax, 3),
        'executability': round(executability, 3),
        'table_text_variability': round(table_text_variability, 3),
        'table_template_variability': round(table_template_variability, 3),
        'n_corpus_templates': n_corpus_templates,
        'exec_var_mean': round(exec_var_mean, 1)
    }


def evaluate_preds(preds, dataset, parsed_tables):
    assert len(preds) == len(dataset)

    scores = {}
    if 'outputs' in dataset:
        scores.update(evaluate_bleu(preds, dataset))

    variability_scores = evaluate_executability_and_variability(preds, dataset, parsed_tables)
    scores.update(variability_scores)

    return scores
