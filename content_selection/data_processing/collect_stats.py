from collections import Counter, defaultdict


def get_quantity_group(num, total):
    if num == total:
        return 'all'
    if num == 1:
        return 'one'
    if num / total <= 0.5:
        return 'several'
    else:
        return 'most'


def collect_num_stats(col):
    return {
        'max': max(col),
        'min': min(col),
        'sum': round(sum(col), 3),
        'avg': round(sum(col) / len(col), 3),
        'max diff': round(max(col) - min(col), 3)
    }


def collect_restricted_num_stats(col):
    return {
        'max': max(col),
        'min': min(col),
        'max diff': round(max(col) - min(col), 3)
    }


def collect_multiple_stats(col, stat_func):
    num_groups = list(zip(*col))
    grp_stats = [stat_func(grp) for grp in num_groups]
    stats = {
        k: ', '.join(str(grp_stat[k]) for grp_stat in grp_stats)
        for k in grp_stats[0].keys()
    }
    return stats


def collect_nums_stats(col):
    return collect_multiple_stats(col, collect_num_stats)


def collect_restricted_nums_stats(col):
    return collect_multiple_stats(col, collect_restricted_num_stats)


def collect_date_stats(col):
    max_date = max(col)
    min_date = min(col)
    diff_days = (max_date - min_date).days
    diff_months = (max_date.year * 12 + max_date.month) - (min_date.year * 12 + min_date.month)
    diff_years = max_date.year - min_date.year
    return {
        'first': min_date.date(),
        'last': max_date.date(),
        'max diff days': diff_days,
        'max diff months': diff_months,
        'max diff years': diff_years
    }


def collect_common_stats_v1(col):
    n_unique = len(set(col))
    is_all_unique = n_unique == len(col)

    counts = Counter(col)
    count_of_counts = Counter(counts.values())
    counts_stat = []
    for n_groups, count in count_of_counts.most_common():
        value_str = f'{count} value{"" if count == 1 else "s"}'
        verb = 'has' if count == 1 else 'have'
        occ_str = f'{n_groups} occurrence{"" if n_groups == 1 else "s"}'
        counts_stat.append(f'{value_str} {verb} {occ_str}')

    return {
        'unique': n_unique,
        'only unique': is_all_unique,
        'counts': ', '.join(counts_stat)
    }


def collect_common_stats_v2(col):
    total = len(col)
    total_unique = len(set(col))
    agg_n_unique = get_quantity_group(total_unique, total)

    count_of_counts = Counter(Counter(col).values())
    agg_counts = defaultdict(set)

    for n_occurrences, n_unique in count_of_counts.items():
        agg_unique = get_quantity_group(n_unique, total_unique)
        agg_occurrences = get_quantity_group(n_occurrences, total)
        agg_counts[agg_unique].add(agg_occurrences)

    unique_stat = f'Column has {agg_n_unique} unique value' + ('s' if agg_n_unique != 'one' else '')
    counts_stat = [unique_stat]

    for agg_unique, agg_occurrences in agg_counts.items():
        for agg_occurrence in sorted(agg_occurrences):
            value_str = f'{agg_unique} value' + ('s' if agg_unique != 'one' else '')
            verb = 'has' if agg_unique == 'one' else 'have'
            occ_str = f'{agg_occurrence} occurrence' + ('s' if agg_occurrence != 'one' else '')
            counts_stat.append(f'{value_str} {verb} {occ_str}')

    return counts_stat


TYPE_FUNCS = {
    'num': {'v1': collect_num_stats, 'v2': collect_num_stats},
    'same_n_nums': {'v1': collect_nums_stats, 'v2': collect_nums_stats},
    'date': {'v1': collect_date_stats, 'v2': collect_date_stats},
    'all': {'v1': collect_common_stats_v1, 'v2': collect_common_stats_v2},
    'rank': {'v1': collect_restricted_num_stats, 'v2': collect_restricted_num_stats},
    'year': {'v1': collect_restricted_nums_stats, 'v2': collect_restricted_nums_stats},
    'time': {'v1': collect_restricted_num_stats, 'v2': collect_restricted_num_stats},
}


def apply_type_func(key, version, col):
    func = TYPE_FUNCS.get(key, {}).get(version)
    if func is None:
        return []

    func_output = func(col)
    if isinstance(func_output, dict):
        func_output = [f'{k}: {v}' for k, v in func_output.items()]
    return func_output


def get_col_stats(col, coltype, version='v2', include_num_stats=False):
    raw_col = [c.value['raw'] for c in col]

    stats = apply_type_func(key='all', version=version, col=raw_col)
    if include_num_stats and coltype in TYPE_FUNCS:
        processed_col = [  # implies that for stats collection we have preprocessed raw values
            c.value['processed']
            for c in col if c.value['processed'] is not None
        ]
        coltype_stats = apply_type_func(key=coltype, version=version, col=processed_col)
        stats.extend(coltype_stats)

    return ', '.join(stats)
