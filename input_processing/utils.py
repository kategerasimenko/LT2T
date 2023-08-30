import os
import json
import random

from config import DATA_DIR


def get_table_id(table):  # 'table_id' for logicnlg and 'url' for logic2text
    return table.get('table_id') or table['url'].rsplit('/', 1)[-1]


def print_coltype_stats(coltypes):
    random.seed(42)

    cols_total = sum(len(v) for v in coltypes.values())
    coltypes_stat = sorted(coltypes.items(), key=lambda x: len(x[1]), reverse=True)

    for coltype, items in coltypes_stat:
        print(coltype, round(len(items) / cols_total, 3))
        items = [i for i in items if i[0] != 'test']  # this is for debug, so no test split here
        examples = random.sample(items, k=min(50, len(items)))
        print(examples)
        print()


def replace_refs_with_tabfact(dataset):
    with open(os.path.join(DATA_DIR, 'LogicNLG', 'r2_training_all.json')) as f:
        raw_tabfact = json.load(f)

    for part, items in dataset.data.items():
        if not items:
            continue

        new_refs = []

        for item in items:
            is_entailed = False
            # relying on the fact that entailed sentences are in the same order in LogicNLG and this TabFact file
            # we remove all non-entailed sentences until we find the next entailed sentence
            while not is_entailed:
                is_entailed = raw_tabfact[item['table_id']][1].pop(0)  # int (0 or 1) if the sentence is true
                tabfact_sent = raw_tabfact[item['table_id']][0].pop(0)
            new_refs.append(tabfact_sent)  # inplace, tabfact_sent must be present because all sentences
                                           # in LogicNLG have `entailed` counterpart in TabFact

        dataset.data[part] = dataset.data[part].remove_columns(['ref']).add_column('ref', new_refs)

    return dataset
