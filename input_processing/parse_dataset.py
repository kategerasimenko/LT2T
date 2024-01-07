from collections import defaultdict

from tqdm import tqdm
from tabgenie import load_dataset

from input_processing.parse_table import parse_table
from input_processing.utils import replace_refs_with_tabfact, print_coltype_stats


def get_logicnlg_table_ids():
    logicnlg = load_dataset('logicnlg')
    logicnlg_table_ids = {
        item['table_id']: part
        for part, items in logicnlg.data.items()
        for item in items
    }
    return logicnlg_table_ids


def parse_tables(dataset_name, splits=None):
    parsed_tables = {}
    all_coltypes_for_stat = defaultdict(list)
    dataset = load_dataset(dataset_name, splits=splits)
    if dataset_name == 'logicnlg':
        dataset = replace_refs_with_tabfact(dataset)

    # for removing intersections between Logic2Text and LogicNLG
    logicnlg_table_ids = get_logicnlg_table_ids()

    for spl, data in dataset.data.items():
        for i, raw_table in tqdm(enumerate(data), total=len(data)):
            table_obj = dataset.prepare_table(raw_table)
            table_csv = dataset.table_to_csv(table_obj)  # for metric
            ref = table_obj.props['reference']

            if 'table_id' in table_obj.props:  # LogicNLG
                table_id = table_obj.props['table_id']
            else:  # Logic2Text
                table_id = table_obj.props['url'].rsplit('/', 1)[-1]

            # rm train examples of Logic2Text that are in test of LogicNLG
            logicnlg_spl = logicnlg_table_ids.get(table_id, spl)
            if logicnlg_spl in ['dev', 'test'] and spl == 'train':
                continue

            if table_id not in parsed_tables:
                try:
                    table, coltypes = parse_table(table_obj, dataset=dataset_name)
                except Exception as e:
                    print('\n', spl, i, e)
                    continue

                parsed_tables[table_id] = {
                    'table_obj': table,
                    'table_csv': table_csv,
                    'coltypes': coltypes,
                    'references': [],
                    'spl': spl,
                    'idx': i
                }

                for j, coltype in enumerate(coltypes):
                    all_coltypes_for_stat[coltype].append((spl, i, j))

                if dataset_name == 'logic2text':
                    parsed_tables[table_id]['lfs'] = []

            parsed_tables[table_id]['references'].append(ref)
            if dataset_name == 'logic2text':
                lf = table_obj.props['logic_str']
                parsed_tables[table_id]['lfs'].append(lf)

    print('coltypes stat')
    print_coltype_stats(all_coltypes_for_stat)

    return parsed_tables
