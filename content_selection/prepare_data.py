import os
import re
import json
from collections import defaultdict

from tqdm import tqdm
from datasets import Dataset, DatasetDict

from config import DATA_DIR
from content_selection.data_processing.download_generated import save_generated_dataset
from content_selection.data_processing.get_table_features import table_to_features
from content_selection.data_processing.get_lf_templates import get_lf_template
from content_selection.data_processing.balance_dataset import oversample


def balance(
        data,
        parsed_tables,
        processing_params,
        strategy='oversample',
        balancing_power=0.5,
        oversample_from_generated=False,
):
    if strategy not in ['oversample']:
        raise ValueError(
            'Unknown sampling strategy. Available choices: `oversample`.'
        )

    if strategy == 'oversample':
        if oversample_from_generated:
            oversampling_source = get_training_data(  # mmm recursion
                parsed_tables,
                mode='train',
                splits=['train'],
                add_main=False,
                add_generated=True,
                processing_params=processing_params
            )['train']
        else:
            oversampling_source = data

        data = oversample(
            dataset=data,
            parsed_tables=parsed_tables,
            oversampling_source=oversampling_source,
            power=balancing_power
        )

    return data


def get_joint_training_data(
        parsed_tables,
        mode='train',
        splits=None,
        use_main_lfs=True,
        additional_lfs=None,
        processing_version='v2',
        include_stats=True,
        include_num_stats=False,
        include_value=False
):
    processed = defaultdict(list)
    for table_id, table_info in tqdm(parsed_tables.items()):
        spl = table_info['spl']
        if splits is not None and spl not in splits:
            continue

        table_obj = table_info['table_obj']
        header = [x.value.strip().lower() for x in table_obj.get_cells()[0]]
        table_lin = table_to_features(
            table_obj,
            table_info['coltypes'],
            version=processing_version,
            include_stats=include_stats,
            include_num_stats=include_num_stats,
            include_value=include_value
        )

        if mode == 'train':
            lf_list = []
            if use_main_lfs:
                lf_list.extend([(lf, 'main') for lf in table_info['lfs']])

            if additional_lfs is not None and table_id in additional_lfs:
                lf_list.extend([(lf, 'generated') for lf in additional_lfs[table_id]])

            # main lfs are not included and there're no generated lfs for this table
            if not lf_list:
                continue

            for lf, source in lf_list:
                lf_template = get_lf_template(raw_lf=lf, header=header)
                sample = {
                    'input': table_lin,
                    'output': lf_template,
                    'table_id': table_id,
                    'source': source
                }
                processed[spl].append(sample)

        # eval during training
        elif mode == 'eval':
            lf_templates = [get_lf_template(raw_lf=lf, header=header) for lf in table_info['lfs']]
            sample = {
                'input': table_lin,
                'outputs': lf_templates,
                'header': header,  # todo: this is a workaround to calc variability later, think how to refactor
                'table_id': table_id
            }
            processed[table_info['spl']].append(sample)

        # inference mode
        else:
            sample = {
                'input': table_lin,
                'table_id': table_id
            }
            processed[table_info['spl']].append(sample)

    for spl, data in processed.items():
        print(spl, len(data))

    return processed


def get_training_data(
        parsed_tables,
        mode='train',
        splits=None,
        add_main=True,
        add_generated=False,
        balancing_strategy=None,
        balancing_power=0.5,
        oversample_from_generated=False,
        processing_params=None
):
    if processing_params is None:
        processing_params = {}

    if add_generated:
        generated_filepath = os.path.join(DATA_DIR, 'content_selection_generated.json')  # todo: this is hardcoded
        if not os.path.exists(generated_filepath):
            save_generated_dataset(generated_filepath)

        with open(generated_filepath) as f:
            generated_lfs = json.load(f)

    else:
        generated_lfs = None

    data = get_joint_training_data(
        parsed_tables,
        mode=mode,
        splits=splits,
        use_main_lfs=add_main,
        additional_lfs=generated_lfs,
        **processing_params
    )

    data = DatasetDict({
        spl: Dataset.from_list(items)
        for spl, items in data.items()
    })

    if 'train' in data and balancing_strategy is not None:
        data['train'] = balance(
            data=data['train'],
            parsed_tables=parsed_tables,
            strategy=balancing_strategy,
            balancing_power=balancing_power,
            oversample_from_generated=oversample_from_generated,
            processing_params=processing_params
        )

    return data


def fix_tokenization(text):
    text = text.replace('}', ' }').replace('{', ' {')
    text = re.sub(' +', ' ', text)
    return text
