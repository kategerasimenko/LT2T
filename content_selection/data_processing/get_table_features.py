from .collect_stats import get_col_stats


def table_to_features(
        table,
        coltypes,
        version='v2',
        include_stats=True,
        include_num_stats=False,
        include_value=False
):
    cols = list(zip(*table.cells))
    col_features = []

    for col, coltype in zip(cols, coltypes):
        col_header = col[0].value.lower()
        col_feature_str = f'Column {col_header}. Type: {coltype}.'

        if include_stats:
            col_stats = get_col_stats(col[1:], coltype, version=version, include_num_stats=include_num_stats)
            col_feature_str += f' Stats: {col_stats}.'

        if include_value:
            value_example = col[1].value['raw']
            col_feature_str += f' Value: {value_example}.'

        col_features.append(col_feature_str)

    col_str = ' '.join(col_features)
    full_feature_str = f"Title: {table.props['title']}. {col_str}"

    return full_feature_str


if __name__ == '__main__':
    import os
    import pickle

    with open(os.path.join('..', '..', 'data', 'logicnlg_parsed_0816_test.pkl'), 'rb') as f:
        tables = pickle.load(f)

    table = tables['2-1554464-3.html.csv']
    a = table_to_features(
        table['table_obj'],
        table['coltypes'],
        include_stats=True,
        include_value=True,
        include_num_stats=True
    )
    print(a)
