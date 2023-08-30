
from collections import defaultdict

from tabgenie import load_dataset
from datasets import Dataset, DatasetDict

from input_processing.parse_dataset import get_logicnlg_table_ids
from input_processing.utils import get_table_id, replace_refs_with_tabfact


MARKERS = {
    'title': '<T>',
    'row': '<R>',
    'cell': '<C>',
    'header': '<H>'
}


def linearize_nl_idxs(table, highlighted_only=False):
    lins = [f"Title: {table.props['title']}"]
    header = None

    for i, row in enumerate(table.get_cells()):
        if not i:
            header = [x.value for x in row]
            continue

        row_lins = [f"Row: {i}"]
        for j, cell in enumerate(row):
            if not highlighted_only or cell.is_highlighted:
                row_lins.append(f"{header[j]}: {cell.value}")

        lins.append(', '.join(row_lins))

    return '. '.join(lins)


def linearize_nl(table, highlighted_only=False):
    lins = [f"Title: {table.props['title']}"]
    header = None

    for i, row in enumerate(table.get_cells()):
        if not i:
            header = [x.value for x in row]
            continue

        row_lins = [f"Row"]
        for j, cell in enumerate(row):
            if not highlighted_only or cell.is_highlighted:
                row_lins.append(f"{header[j]}: {cell.value}")

        lins.append(', '.join(row_lins))

    return '. '.join(lins)


def linearize_markers(table, highlighted_only=False):
    lins = [f"{MARKERS['title']} {table.props['title']}"]
    header = None

    for i, row in enumerate(table.get_cells()):
        if not i:
            header = [x.value for x in row]
            continue

        lins.append(f"{MARKERS['row']} {i}")
        for j, cell in enumerate(row):
            if not highlighted_only or cell.is_highlighted:
                lins.append(f"{MARKERS['header']} {header[j]} {MARKERS['cell']} {cell.value}")

    return ' '.join(lins)


def process_tables(tg_dataset, splits, linearize_func, highlighted_only=False):
    processed_data = defaultdict(list)
    logicnlg_table_ids = get_logicnlg_table_ids()

    for part in splits:
        for item in tg_dataset.data[part]:
            table_id = get_table_id(item)
            table_obj = tg_dataset.prepare_table(item)

            # rm train examples of Logic2Text that are in test of LogicNLG
            logicnlg_spl = logicnlg_table_ids.get(table_id, part)
            if logicnlg_spl in ['dev', 'test'] and part == 'train':
                continue

            table_csv = tg_dataset.table_to_csv(table_obj)  # for metric
            table_lin = linearize_func(table_obj, highlighted_only=highlighted_only)

            processed_item = {
                'input': table_lin,
                'output': table_obj.props['reference'],
                'table_id': table_id,
                'table_csv': table_csv
            }
            processed_data[part].append(processed_item)

    dataset = DatasetDict({
        spl: Dataset.from_list(items)
        for spl, items in processed_data.items()
    })

    return dataset


def get_training_data(dataset, splits=None, linearize_style='nl', references='original'):
    tg_dataset = load_dataset(dataset)
    if dataset == 'logicnlg' and references == 'tabfact':
        tg_dataset = replace_refs_with_tabfact(tg_dataset)

    highlighted_only = True if dataset == 'logicnlg' else False

    if linearize_style == 'nl':
        linearize = linearize_nl
    elif linearize_style == 'nl_idxs':
        linearize = linearize_nl_idxs
    elif linearize_style == 'markers':
        linearize = linearize_markers
    else:
        raise ValueError(
            f'Linearize style {linearize_style} is unavailable. '
            f'Available styles: `nl`, `nl_idxs`, `markers`'
        )

    hf_dataset = process_tables(
        tg_dataset,
        splits=splits,
        linearize_func=linearize,
        highlighted_only=highlighted_only
    )
    return hf_dataset
