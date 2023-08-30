from .funcs import (
    filter_eq, filter_not_eq, filter_greater, filter_greater_eq, filter_less, filter_less_eq, filter_all,
    col_max, col_min, col_argmax, col_argmin, col_nth_max, col_nth_min, col_nth_argmax, col_nth_argmin,
    col_avg, col_sum,
    count, diff, hop,
    only, logic_and, eq, not_eq, round_eq, greater, less,
    most_eq, most_not_eq, most_greater, most_greater_eq, most_less, most_less_eq,
    all_eq, all_not_eq, all_greater, all_greater_eq, all_less, all_less_eq
)

# todo: normal floor / ceil for year, formatting for time
FUNCS = {
    'filter_eq': {'func': filter_eq, 'type': 'all'},
    'filter_not_eq': {'func': filter_not_eq, 'type': 'all'},
    'filter_greater': {'func': filter_greater, 'type': ['num', 'rank']},  # cannot define proper floor for date, year, and time
    'filter_greater_eq': {'func': filter_greater_eq, 'type': ['num', 'date', 'rank', 'year', 'time']},
    'filter_less': {'func': filter_less, 'type': ['num', 'rank']},  # cannot define proper floor for date, year, and time
    'filter_less_eq': {'func': filter_less_eq, 'type': ['num', 'date', 'rank', 'year', 'time']},
    'filter_all': {'func': filter_all, 'type': 'all'},
    'max': {'func': col_max, 'type': ['num', 'date', 'rank', 'year', 'time']},
    'min': {'func': col_min, 'type': ['num', 'date', 'rank', 'year', 'time']},
    'argmax': {'func': col_argmax, 'type': ['num', 'date', 'rank', 'year', 'time']},
    'argmin': {'func': col_argmin, 'type': ['num', 'date', 'rank', 'year', 'time']},
    'nth_max': {'func': col_nth_max, 'type': ['num', 'date', 'rank', 'year', 'time']},
    'nth_min': {'func': col_nth_min, 'type': ['num', 'date', 'rank', 'year', 'time']},
    'nth_argmax': {'func': col_nth_argmax, 'type': ['num', 'date', 'rank', 'year', 'time']},
    'nth_argmin': {'func': col_nth_argmin, 'type': ['num', 'date', 'rank', 'year', 'time']},
    'avg': {'func': col_avg, 'type': ['num', 'time']},
    'sum': {'func': col_sum, 'type': ['num', 'time']},
    'count': {'func': count, 'type': 'all'},
    'diff': {'func': diff, 'type': ['num', 'date', 'rank', 'year', 'time']},  # doesn't control coltypes
    'hop': {'func': hop, 'type': 'all'},
    'only': {'func': only, 'type': 'all'},
    'and': {'func': logic_and, 'type': 'all'},
    'eq': {'func': eq, 'type': 'all'},
    'not_eq': {'func': not_eq, 'type': 'all'},
    'round_eq': {'func': round_eq, 'type': ['num', 'time']},
    'greater': {'func': greater, 'type': ['num', 'date', 'year', 'rank', 'time']},
    'less': {'func': less, 'type': ['num', 'date', 'year', 'rank', 'time']},
    'most_eq': {'func': most_eq, 'type': 'all'},
    'most_not_eq': {'func': most_not_eq, 'type': 'all'},
    'most_greater': {'func': most_greater, 'type': ['num']},  # cannot define proper floor for date, year, and time
    'most_greater_eq': {'func': most_greater_eq, 'type': ['num', 'date', 'year', 'time']},
    'most_less': {'func': most_less, 'type': ['num']},  # cannot define proper ceil for date, year, and time
    'most_less_eq': {'func': most_less_eq, 'type': ['num', 'date', 'year', 'time']},
    'all_eq': {'func': all_eq, 'type': 'all'},
    'all_not_eq': {'func': all_not_eq, 'type': 'all'},
    'all_greater': {'func': all_greater, 'type': ['num']},  # cannot define proper floor for date, year, and time
    'all_greater_eq': {'func': all_greater_eq, 'type': ['num', 'date', 'year', 'time']},
    'all_less': {'func': all_less, 'type': ['num']},  # cannot define proper ceil for date, year, and time
    'all_less_eq': {'func': all_less_eq, 'type': ['num', 'date', 'year', 'time']}
}
for func_name, func_info in FUNCS.items():
    func = func_info['func']
    func_info['args'] = func.__code__.co_varnames[:func.__code__.co_argcount]

# words that are expected apart from header
ALL_ROWS = 'all_rows'
PLACEHOLDER = 'X'
ADDITIONAL_ARGS = ['upper_funcs', 'children']

COLTYPES_TO_REVISE = {
    'same_n_nums': 'num',
    'year': 'year'
}

