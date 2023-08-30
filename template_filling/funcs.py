import re
import numbers
from collections import Counter

import pandas as pd
import numpy as np

from .utils import (
    get_processed_to_raw_mapping,
    get_substrings,
    round_number,
    get_range_step,
    get_floor,
    get_ceil
)
from .val_func_checks import run_checks


def filter_all(table, col):
    return [{'filled': None, 'output': table}]


def filter_eq(table, col, upper_funcs, children):
    val_options = []
    proc2raw = get_processed_to_raw_mapping(table, col, preserve_all_raw=True)
    substrings = get_substrings(table[col + '_raw'])
    for substring in substrings:
        proc2raw[substring] = [substring]

    for val, raw_vals in proc2raw.items():
        if len(raw_vals) > 1:
            continue

        raw_val = raw_vals[0]

        if isinstance(val, str):
            val = f'(?:^| ){re.escape(val)}(?: |$)'
            possible_output = table.loc[table[col + '_raw'].str.contains(val)]
        elif pd.isna(val):  # empty value
            possible_output = table.loc[table[col].isna()]
        else:  # processed value
            possible_output = table.loc[table[col] == val]

        children['args'][children['placeholder_idx']] = raw_val
        is_suitable = run_checks(
            func_name='filter_eq',
            upper_funcs=upper_funcs,
            children=children['args'],
            val_idxs=list(possible_output.index),  # assuming that index of the whole table is sequential
            val_count=possible_output.shape[0],
            table=table
        )
        if is_suitable:
            val_options.append({'filled': raw_val, 'output': possible_output.reset_index()})

    return val_options


def filter_not_eq(table, col, upper_funcs, children):
    val_options = []
    proc2raw = get_processed_to_raw_mapping(table, col, preserve_all_raw=True)

    for val, raw_vals in proc2raw.items():
        if len(raw_vals) > 1:
            continue

        raw_val = raw_vals[0]

        if pd.isna(val):  # empty value
            possible_output = table.loc[~(table[col].isna())]
        else:  # processed value
            possible_output = table.loc[table[col] != val]

        children['args'][children['placeholder_idx']] = raw_val
        is_suitable = run_checks(
            func_name='filter_not_eq',
            upper_funcs=upper_funcs,
            children=children['args'],
            val_idxs=list(possible_output.index),  # assuming that index of the whole table is sequential
            val_count=possible_output.shape[0],
            table=table
        )
        if is_suitable:
            val_options.append({'filled': raw_val, 'output': possible_output.reset_index()})

    return val_options


def filter_greater_eq(table, col, upper_funcs, children):
    val_options = []
    max_val = table[col].max()
    proc2raw = get_processed_to_raw_mapping(table, col)

    for val, raw_val in proc2raw.items():
        if pd.isna(val) or val == max_val:  # taking null or max element doesn't make sense
            continue

        possible_output = table.loc[table[col] >= val]

        children['args'][children['placeholder_idx']] = raw_val
        is_suitable = run_checks(
            func_name='filter_greater_eq',
            upper_funcs=upper_funcs,
            children=children['args'],
            val_idxs=list(possible_output.index),  # assuming that index of the whole table is sequential
            val_count=possible_output.shape[0],
            table=table
        )
        if is_suitable:
            val_options.append({'filled': raw_val, 'output': possible_output.reset_index()})

    return val_options


def filter_less_eq(table, col, upper_funcs, children):
    val_options = []
    min_val = table[col].min()
    proc2raw = get_processed_to_raw_mapping(table, col)

    for val, raw_val in proc2raw.items():
        if pd.isna(val) or val == min_val:  # taking null or min element doesn't make sense
            continue

        possible_output = table.loc[table[col] <= val]

        children['args'][children['placeholder_idx']] = raw_val
        is_suitable = run_checks(
            func_name='filter_less_eq',
            upper_funcs=upper_funcs,
            children=children['args'],
            val_idxs=list(possible_output.index),  # assuming that index of the whole table is sequential
            val_count=possible_output.shape[0],
            table=table
        )
        if is_suitable:
            val_options.append({'filled': raw_val, 'output': possible_output.reset_index()})

    return val_options


def filter_greater(table, col, upper_funcs, children):
    val_options = []
    col_values = table[col].dropna()
    step = get_range_step(col_values)

    sorted_unique_vals = sorted(set(col_values))

    for i, val in enumerate(sorted_unique_vals):
        floor = get_floor(val, step)

        if isinstance(val, numbers.Number):
            lower_bound = floor
            lower_bound_to_fill = floor
        else:
            continue

        possible_output = table.loc[table[col] > lower_bound]
        if possible_output.shape[0] == len(col_values):  # whole table is selected
            continue

        children['args'][children['placeholder_idx']] = lower_bound_to_fill
        is_suitable = run_checks(
            func_name='filter_greater',
            upper_funcs=upper_funcs,
            children=children['args'],
            val_idxs=list(possible_output.index),  # assuming that index of the whole table is sequential
            val_count=possible_output.shape[0],
            table=table
        )
        if is_suitable:
            val_options.append({'filled': lower_bound_to_fill, 'output': possible_output.reset_index()})

    return val_options


def filter_less(table, col, upper_funcs, children):
    val_options = []
    col_values = table[col].dropna()
    proc2raw = get_processed_to_raw_mapping(table, col)
    step = get_range_step(col_values)

    sorted_unique_vals = sorted(set(col_values), reverse=True)

    for i, val in enumerate(sorted_unique_vals):
        ceil = get_ceil(val, step)

        if isinstance(val, numbers.Number):
            upper_bound = ceil
            upper_bound_to_fill = ceil
        else:
            continue

        possible_output = table.loc[table[col] < upper_bound]
        if possible_output.shape[0] == len(col_values):
            continue

        children['args'][children['placeholder_idx']] = upper_bound_to_fill
        is_suitable = run_checks(
            func_name='filter_less',
            upper_funcs=upper_funcs,
            children=children['args'],
            val_idxs=list(possible_output.index),  # assuming that index of the whole table is sequential
            val_count=possible_output.shape[0],
            table=table
        )
        if is_suitable:
            val_options.append({'filled': upper_bound_to_fill, 'output': possible_output.reset_index()})

    return val_options


def col_max(table, col):
    max_val = table[col].max()
    if pd.isna(max_val):
        return []

    proc2raw = get_processed_to_raw_mapping(table, col)
    max_val = proc2raw[max_val]

    return [{'filled': None, 'output': max_val}]  # only one option


def col_min(table, col):
    min_val = table[col].min()
    if pd.isna(min_val):
        return []

    proc2raw = get_processed_to_raw_mapping(table, col)
    min_val = proc2raw[min_val]

    return [{'filled': None, 'output': min_val}]


def col_argmax(table, col):
    if pd.isna(table[col].max()):
        return []

    return [{'filled': None, 'output': table.iloc[table[col].argmax()]}]


def col_argmin(table, col):
    if pd.isna(table[col].min()):
        return []

    return [{'filled': None, 'output': table.iloc[table[col].argmin()]}]


def col_avg(table, col):
    val = table[col].mean()
    if pd.isna(val):
        return []

    rounded = round_number(val)
    return [{'filled': None, 'output': rounded}]


def col_sum(table, col):
    val = table[col].sum()
    if pd.isna(val):
        return []

    rounded = round_number(val)
    return [{'filled': None, 'output': rounded}]


# todo: join nth_max and nth_min?
def col_nth_max(table, col):
    col_values = table[col].dropna()
    proc2raw = get_processed_to_raw_mapping(table, col)

    half = len(col_values) // 2  # common sense
    sorted_vals = col_values.sort_values(ascending=False).tolist()

    val_options = []
    for i in range(half):
        val = proc2raw[sorted_vals[i]]
        val_options.append({'filled': i + 1, 'output': val})

    return val_options


def col_nth_min(table, col):
    col_values = table[col].dropna()
    proc2raw = get_processed_to_raw_mapping(table, col)

    half = len(col_values) // 2
    sorted_vals = col_values.sort_values().tolist()

    val_options = []
    for i in range(half):
        val = proc2raw[sorted_vals[i]]
        val_options.append({'filled': i + 1, 'output': val})

    return val_options


def col_nth_argmax(table, col, upper_funcs, children):
    col_values = table[col].dropna()
    half = len(col_values) // 2
    sorted_vals = col_values.sort_values(ascending=False)

    val_options = []
    for i in range(half):
        children['args'][children['placeholder_idx']] = i + 1
        is_suitable = run_checks(
            check_list=['and'],  # in the data, nth_argmax is not used in greater / less / not_eq
            func_name='nth_argmax',
            upper_funcs=upper_funcs,
            children=children['args']
        )

        if is_suitable:
            item_idx = sorted_vals.index[i]
            val_options.append({'filled': i + 1, 'output': table.iloc[item_idx]})

    return val_options


def col_nth_argmin(table, col, upper_funcs, children):
    col_values = table[col].dropna()
    half = len(col_values) // 2
    sorted_vals = col_values.sort_values()

    val_options = []
    for i in range(half):
        children['args'][children['placeholder_idx']] = i + 1
        is_suitable = run_checks(
            check_list=['and'],  # in the data, nth_argmin is not used in greater / less / not_eq
            func_name='nth_argmin',
            upper_funcs=upper_funcs,
            children=children['args']
        )

        if is_suitable:
            item_idx = sorted_vals.index[i]
            val_options.append({'filled': i + 1, 'output': table.iloc[item_idx]})

    return val_options


def hop(row, col, upper_funcs):
    val = row[col]
    raw_val = row[col + '_raw']
    if isinstance(val, pd.Series):
        val = val.item()
        raw_val = raw_val.item()

    parent = upper_funcs[-1]
    if parent['func'] == 'eq':
        val = raw_val

    return [{'filled': None, 'output': val}]


def count(table):
    return [{'filled': None, 'output': table.shape[0]}]


def only(table):
    return [{'filled': None, 'output': table.shape[0] == 1}]


def logic_and(bool1, bool2):
    return [{'filled': None, 'output': bool1 and bool2}]


def eq(val):
    return [{'filled': val, 'output': True}]  # weird but how do I check the execution result?


def not_eq(val1, val2):  # in training set, not_eq is used only with two subformulas, so no placeholder
    return [{'filled': None, 'output': val1 != val2}]


def round_eq(val):
    rounded = round_number(val)
    return [{'filled': rounded, 'output': True}]  # weird but how do I check the execution result?


def diff(val1, val2):
    val = val1 - val2
    if isinstance(val, pd.Timedelta):
        val = val.days
    rounded = round_number(val)
    return [{'filled': None, 'output': rounded}]


def greater(val1, val2):
    return [{'filled': None, 'output': val1 > val2}]


def less(val1, val2):
    return [{'filled': None, 'output': val1 < val2}]


def most_eq(table, col):
    col_values = table[col].replace([np.nan], [None])
    proc2raw = get_processed_to_raw_mapping(table, col, preserve_all_raw=True)
    substrings = get_substrings(table[col + '_raw'])
    for substring in substrings:
        proc2raw[substring] = [substring]

    total = len(col_values)
    half = total / 2

    def options_from_counts(counts):
        options = []
        for val, c in counts.items():
            if c > half and c != total and len(proc2raw[val]) == 1:
                options.append({'filled': proc2raw[val][0], 'output': True})
        return options

    main_counts = Counter(col_values.tolist())
    main_options = options_from_counts(main_counts)
    if main_options:
        return main_options

    substring_counts = Counter(substrings)
    substring_options = options_from_counts(substring_counts)
    return substring_options


def most_not_eq(table, col):
    col_values = table[col].replace([np.nan], [None])
    proc2raw = get_processed_to_raw_mapping(table, col, preserve_all_raw=True)

    num_values = len(col_values)
    half = num_values / 2
    counts = Counter(col_values)

    options = []
    for val, c in counts.items():
        if num_values - c > half and len(proc2raw[val]) == 1:
            options.append({'filled': proc2raw[val][0], 'output': True})
    return options


def most_greater(table, col):
    nums = table[col].dropna()  # logical - we're interested in "most out of present values"
    step = get_range_step(nums)

    half = len(nums) // 2
    first_half = sorted(nums)[:half]
    first_half_unique = sorted(set(first_half))

    options = []
    for i, num in enumerate(first_half_unique):
        floor = get_floor(num, step)

        if isinstance(num, numbers.Number):
            lower_bound = floor
        else:
            continue

        n_greater = sum(nums > lower_bound)
        if n_greater > half and n_greater != len(nums):
            options.append({'filled': lower_bound, 'output': True})

    return options


def most_less(table, col):
    nums = table[col].dropna()  # logical - we're interested in "most out of present values"
    step = get_range_step(nums)

    half = len(nums) // 2
    second_half = sorted(nums, reverse=True)[:half]
    second_half_unique = sorted(set(second_half), reverse=True)

    options = []
    for i, num in enumerate(second_half_unique):
        ceil = get_ceil(num, step)

        if isinstance(num, numbers.Number):
            upper_bound = ceil
        else:
            continue

        n_less = sum(nums < upper_bound)
        if n_less > half and n_less != len(nums):
            options.append({'filled': upper_bound, 'output': True})

    return options


def most_greater_eq(table, col):
    nums = table[col].dropna()  # logical - we're interested in "most out of present values"
    proc2raw = get_processed_to_raw_mapping(table, col)

    half = len(nums) // 2
    first_half_wo_min = sorted(nums)[1:half]

    options = [
        {'filled': proc2raw[num], 'output': True}
        for num in first_half_wo_min
    ]
    return options


def most_less_eq(table, col):
    nums = table[col].dropna()  # logical - we're interested in "most out of present values"
    proc2raw = get_processed_to_raw_mapping(table, col)

    half = len(nums) // 2
    second_half_wo_max = sorted(nums)[-half:][1::-1]

    options = [
        {'filled': proc2raw[num], 'output': True}
        for num in second_half_wo_max
    ]
    return options


def all_eq(table, col):
    options = []
    vals = table[col].replace([np.nan], [None])

    proc2raw = get_processed_to_raw_mapping(table, col, preserve_all_raw=True)
    substrings = get_substrings(table[col + '_raw'])
    for substring in substrings:
        proc2raw[substring] = [substring]

    val_counts = Counter(vals.tolist() + substrings)
    for val, count in val_counts.items():
        if count == len(table[col]) and len(proc2raw[val]) == 1:
            options.append({'filled': proc2raw[val][0], 'output': True})

    return options


def all_not_eq(table, col):
    return []  # not clear what to put, the only example is train 3450


def all_greater_eq(table, col):
    min_val = table[col].min()
    if pd.isna(min_val):
        return []

    proc2raw = get_processed_to_raw_mapping(table, col)
    return [{'filled': proc2raw[min_val], 'output': True}]


def all_less_eq(table, col):
    max_val = table[col].max()
    if pd.isna(max_val):
        return []

    proc2raw = get_processed_to_raw_mapping(table, col)
    return [{'filled': proc2raw[max_val], 'output': True}]


def all_greater(table, col):
    nums = table[col]
    min_val = nums.min()
    if pd.isna(min_val):
        return []

    step = get_range_step(nums)
    floor = get_floor(min_val, step)
    return [{'filled': floor, 'output': True}]


def all_less(table, col):
    nums = table[col]
    max_val = nums.max()
    if pd.isna(max_val):
        return []

    step = get_range_step(nums)
    ceil = get_ceil(max_val, step)
    return [{'filled': ceil, 'output': True}]
