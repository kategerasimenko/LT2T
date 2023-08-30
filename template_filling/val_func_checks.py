import pandas as pd

from input_processing.str2form import linearize_lf
from .utils import find_upper_func


# ------- UNIQUE function -------


def check_suits_unique(upper_funcs, val_count, **kwargs):
    has_unique = False
    has_count = False  # for counting, everything will do

    for func_info in upper_funcs:
        if func_info['func'] in ['only', 'hop']:
            has_unique = True

        if func_info['func'] == 'count':
            has_count = True

        # if some other filtering is on upper levels, it will fulfill the uniqueness req
        if any(func_info['func'].startswith(x) for x in ['filter', 'arg', 'nth_arg']):
            has_unique = False
            has_count = False

    non_empty = bool(val_count)
    ok_for_uniqueness = (has_unique and val_count == 1) or (not has_unique and val_count > 1)
    return non_empty and ok_for_uniqueness


# ------- GREATER function -------


def check_suits_greater(upper_funcs, val_idxs, table, **kwargs):
    greater_present = False
    first_greater_arg = None
    greater_col_vals = None

    for func_info in upper_funcs:
        if func_info['func'] == 'greater':
            greater_present = True
            if func_info['filled_args']:
                first_greater_arg = func_info['filled_args'][0]

        # if some other filtering is on upper levels, it will fulfill the greater req
        if any(func_info['func'].startswith(x) for x in ['filter', 'arg', 'nth_arg']):
            greater_present = False

        # hop *must* be after filtering and before greater
        if greater_present and func_info['func'] == 'hop':
            greater_col_vals = table[func_info['raw_args'][1]]

    if not greater_present:
        return True

    if len(val_idxs) > 1:  # value must be unique
        return False

    val_idx = val_idxs[0]
    if pd.isna(greater_col_vals[val_idx]):
        return False

    # not yet filled -> this argument is the first one
    if first_greater_arg is None:
        # we don't accept minimum value as the first arg (need the second arg to be smaller)
        if greater_col_vals[val_idx] == greater_col_vals.min():
            return False

    # first argument already selected, so we need to select lower value
    elif greater_col_vals[val_idx] >= first_greater_arg:
        return False

    return True


# ------- LESS function -------


def check_suits_less(upper_funcs, val_idxs, table, **kwargs):
    less_present = False
    first_less_arg = None
    less_col_vals = None

    for func_info in upper_funcs:
        if func_info['func'] == 'less':
            less_present = True
            if func_info['filled_args']:
                first_less_arg = func_info['filled_args'][0]

        # if some other filtering is on upper levels, it will fulfill the less req
        if any(func_info['func'].startswith(x) for x in ['filter', 'arg', 'nth_arg']):
            less_present = False

        # hop *must* be after filtering and before less
        if less_present and func_info['func'] == 'hop':
            less_col_vals = table[func_info['raw_args'][1]]

    if not less_present:
        return True

    if len(val_idxs) > 1:  # value must be unique
        return False

    val_idx = val_idxs[0]
    if pd.isna(less_col_vals[val_idx]):
        return False

    # not yet filled -> this argument is the first one
    if first_less_arg is None:
        # we don't accept maximum value as the first arg (need the second arg to be larger)
        if less_col_vals[val_idx] == less_col_vals.max():
            return False

    # first argument already selected, so we need to select higher value
    elif less_col_vals[val_idx] <= first_less_arg:
        return False

    return True


# ------- EQ function -------


def check_suits_eq(func_name, upper_funcs, children, val_idxs, table, **kwargs):
    # In general, eq can be applied to anything.
    # Here we are interested in eq { hop {...} ; hop {...} } constructions
    # and check the second argument
    current_lin_func = linearize_lf({'func': func_name, 'args': children})
    first_eq_arg = None
    eq_col_vals = None

    for func_info in upper_funcs:
        if func_info['func'] == 'eq':
            if func_info['filled_args']:
                first_eq_arg = {
                    'lf': func_info['arg_lfs'][0],
                    'filled': func_info['filled_args'][0]
                }

        # if some other filtering is on upper levels, it will fulfill the not_eq req
        if any(func_info['func'].startswith(x) for x in ['filter', 'arg', 'nth_arg']):
            first_eq_arg = None

        if first_eq_arg is not None and func_info['func'] == 'hop':
            eq_col_vals = table[func_info['raw_args'][1]]

    # there are no restrictions on the first argument and it can be anything
    if first_eq_arg is None:
        return True

    if len(val_idxs) > 1:  # value must be unique
        return False

    # first argument already selected, so we need to select an equal value...
    is_eq_result = eq_col_vals[val_idxs[0]] == first_eq_arg['filled']
    # ...but not the same selection (without this check, value will match with itself) - terrible workaround
    is_not_eq_selection = current_lin_func not in linearize_lf(first_eq_arg['lf'])

    if not is_eq_result or not is_not_eq_selection:
        return False

    return True


# ------- NOT_EQ / DIFF function -------


def check_suits_not_eq(upper_funcs, val_idxs, table, **kwargs):
    not_eq_func = None
    first_not_eq_arg = None
    not_eq_col_vals = None

    for func_info in upper_funcs:
        if func_info['func'] in ['not_eq', 'diff']:
            not_eq_func = func_info['func']
            if func_info['filled_args']:
                first_not_eq_arg = func_info['filled_args'][0]

        # if some other filtering is on upper levels, it will fulfill the not_eq req
        if any(func_info['func'].startswith(x) for x in ['filter', 'arg', 'nth_arg']):
            not_eq_func = None

        # hop *must* be after filtering and before not_eq
        if not_eq_func is not None and func_info['func'] == 'hop':
            not_eq_col_vals = table[func_info['raw_args'][1]]

    if not_eq_func is None:
        return True

    if len(val_idxs) > 1:  # value must be unique
        return False

    val_idx = val_idxs[0]
    if not_eq_func == 'diff' and pd.isna(not_eq_col_vals[val_idx]):
        return False

    # first argument already selected, so we need to select non-equal value
    if first_not_eq_arg is not None and not_eq_col_vals[val_idx] == first_not_eq_arg:
        return False

    return True


# ------- AND function -------


def collect_prev_filters_for_and_check(func_name, upper_funcs):
    prev_and_filters = []
    and_present = False

    if func_name.startswith('filter_'):
        filtering_func_to_find = 'filter_'
    elif func_name.startswith('nth_arg'):
        filtering_func_to_find = 'nth_'
    else:
        # we come here only for filter_ and nth_ funcs
        raise ValueError(f'check is not designed for {func_name} function.')

    for func_info in upper_funcs:
        if func_info['func'] == 'and':
            and_present = True
            if not func_info['arg_lfs']:
                continue

            # only two args and if we're in the second arg, we need to take filters from first arg
            first_arg_filters = find_upper_func(func_info['arg_lfs'][0], func=filtering_func_to_find)
            if first_arg_filters:
                prev_and_filters.append(first_arg_filters)

        # if some other filtering is on upper levels, it will check the match for "and" args
        if any(func_info['func'].startswith(x) for x in ['filter', 'nth_arg']):
            and_present = False

    return and_present, prev_and_filters


def get_prev_filter_for_and_check(upper_funcs, upper_and_filters):
    # filter_, nth_arg
    if len(upper_and_filters[0]) == 1:
        filter_func_to_compare = upper_and_filters[-1][0]

    # diff, greater, less, eq, not_eq
    else:
        # if "and" is with diff etc., there must be the second "and" to introduce new values in lf
        assert len(upper_funcs) > 1 and upper_funcs[1]['func'] == 'and'

        # we're deeper in the formula than first two "and"s and the first arg is there
        if any(f['func'] == 'and' for f in upper_funcs[2:]) and upper_and_filters[-1]:
            filter_func_to_compare = upper_and_filters[-1][0]
        else:
            # in which arg of second "and" we are now
            n_arg = len(upper_funcs[1]['filled_args'])
            # take corresponding filter from the diff etc
            filter_func_to_compare = upper_and_filters[0][n_arg]

    return filter_func_to_compare


def check_suits_and(func_name, upper_funcs, children, **kwargs):
    and_present, upper_and_filters = collect_prev_filters_for_and_check(func_name, upper_funcs)

    # no "and" functions
    if not and_present:
        return True

    # this is the first filter to be processed from all func filters
    if not upper_and_filters:
        return True

    filter_func_to_compare = get_prev_filter_for_and_check(upper_funcs, upper_and_filters)
    previous_filter = linearize_lf(filter_func_to_compare)
    current_filter = linearize_lf({'func': func_name, 'args': children})

    # nth_argmax -> nth_max to match these two parts
    previous_filter = previous_filter.replace('_arg', '_')
    current_filter = current_filter.replace('_arg', '_')

    if previous_filter != current_filter:
        return False

    return True


# ------- main function -------


def run_checks(
        check_list='all',
        func_name=None,  # for and, eq
        upper_funcs=None,  # for all
        children=None,  # for and, eq
        val_idxs=None,  # for eq, not_eq, greater, less
        val_count=None,  # for unique
        table=None  # for eq, not_eq, greater, less
):
    check_funcs = {
        'unique': check_suits_unique,
        'greater': check_suits_greater,
        'less': check_suits_less,
        'eq': check_suits_eq,
        'not_eq': check_suits_not_eq,
        'and': check_suits_and
    }
    if check_list != 'all':
        check_funcs = {k: v for k, v in check_funcs.items() if k in check_list}

    for check_name, func in check_funcs.items():
        check_result = func(
            func_name=func_name,
            upper_funcs=upper_funcs,
            children=children,
            val_idxs=val_idxs,
            val_count=val_count,
            table=table
        )
        if not check_result:
            return False

    return True
