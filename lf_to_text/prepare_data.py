from collections import defaultdict

from datasets import Dataset, DatasetDict


def linearize(title, header, lf):
    if lf.endswith(' = true'):
        lf = lf[:-len(' = true')]

    lin = f'Title: {title}. Header: {" | ".join(header)}. Logical form: {lf}'
    return lin


def prepare_lf2text_data(parsed_tables, splits=None, mode='train', logical_forms=None):
    processed_data = defaultdict(list)

    for table_id, table_info in parsed_tables.items():
        spl = table_info['spl']
        if splits is not None and spl not in splits:
            continue

        table_obj = table_info['table_obj']
        header = [x.value.strip().lower() for x in table_obj.get_cells()[0]]

        # training mode
        if mode == 'train':
            lfs = table_info['lfs']
            refs = table_info['references']  # in lf2text, only one reference per input

        # inference mode
        else:
            # logical_forms must be provided
            assert logical_forms is not None
            # logical_forms[table_id] is a list of lists - list of filling options for every generated template
            lfs = [x for lfgrp in logical_forms[table_id] for x in lfgrp]
            refs = None  # no references during inference

        for i, lf in enumerate(lfs):
            lin_input = linearize(
                header=header,
                title=table_obj.props['title'],
                lf=lf
            )

            item = {
                'input': lin_input,
                'lf': lf,
                'table_id': table_id
            }
            if refs is not None:
                item['output'] = refs[i]

            processed_data[spl].append(item)

    data = DatasetDict({
        spl: Dataset.from_list(items)
        for spl, items in processed_data.items()
    })

    return data
