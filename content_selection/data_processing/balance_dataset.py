import random
from collections import defaultdict, Counter

import numpy as np
from datasets import concatenate_datasets

from input_processing.str2form import parse_str, linearize_lf
from .get_lf_templates import replace_with_placeholders


def extract_structures(dataset, parsed_tables):
    structures = []

    for item in dataset:
        table_id = item['table_id']
        table_info = parsed_tables[table_id]
        table_obj = table_info['table_obj']
        header = [x.value.strip().lower() for x in table_obj.get_cells()[0]]

        template = parse_str(item['output'], func_map={})
        struct = replace_with_placeholders(header, template, replace_col_names=True)
        struct = linearize_lf(struct)
        structures.append(struct)

    return structures


def calculate_balanced_counts(counts, power):
    root_counts = {k: np.power(v, power) for k, v in counts.items()}
    sum_roots = sum(root_counts.values())
    balanced_probs = {k: v / sum_roots for k, v in root_counts.items()}

    # fix size of the balanced dataset based on the most frequent class
    most_freq, most_freq_count = counts.most_common()[0]
    new_total = most_freq_count / balanced_probs[most_freq]

    balanced_counts = {k: int(new_total * v) for k, v in balanced_probs.items()}
    return balanced_counts


def get_oversample_indices(counts, balanced_counts, oversampling_by_idx):
    indices_to_add = []

    for struct, real_count in counts.items():
        n_to_add = balanced_counts[struct] - real_count
        if n_to_add <= 0 or struct not in oversampling_by_idx:
            continue

        oversample_indices = oversampling_by_idx[struct]
        if len(oversample_indices) < n_to_add:  # oversample with replacement
            sample = random.choices(oversample_indices, k=n_to_add)
        else:  # subsample unique
            sample = random.sample(oversample_indices, k=n_to_add)
        indices_to_add.extend(sample)

    return indices_to_add


def oversample(dataset, parsed_tables, oversampling_source=None, power=0.5):
    random.seed(42)

    structures = extract_structures(dataset, parsed_tables)
    counts = Counter(structures)
    balanced_counts = calculate_balanced_counts(counts, power)

    if oversampling_source is not None:
        oversampling_source = concatenate_datasets([dataset, oversampling_source])
        oversampling_structures = extract_structures(oversampling_source, parsed_tables)
    else:
        oversampling_source = dataset
        oversampling_structures = structures

    oversampling_by_idx = defaultdict(list)
    for i, struct in enumerate(oversampling_structures):
        oversampling_by_idx[struct].append(i)

    indices_to_add = get_oversample_indices(counts, balanced_counts, oversampling_by_idx)
    balanced = concatenate_datasets([
        dataset,
        oversampling_source.select(indices_to_add)
    ])

    return balanced
