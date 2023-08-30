from tabgenie import load_dataset

from input_processing.str2form import parse_str, linearize_lf


def replace_with_placeholders(header, lf, replace_col_names=False):
    for i, arg in enumerate(lf['args']):
        if isinstance(arg, dict):
            lf['args'][i] = replace_with_placeholders(header, arg, replace_col_names)
        elif replace_col_names and arg in header:
            lf['args'][i] = 'Y'
        elif arg not in header and arg != 'all_rows':  # this is some value then
            lf['args'][i] = 'X'
    return lf


def get_lf_template(raw_lf, header):
    lf = parse_str(raw_lf, func_map={})
    lf = replace_with_placeholders(header, lf)
    lf_str = linearize_lf(lf)
    return lf_str


if __name__ == '__main__':
    dataset = load_dataset('logic2text')
    dataset.load('train')
    t = dataset.prepare_table(dataset.data['train'][1739])
    lf = parse_str(t.props['logic_str'], func_map={})
    header = [c.value.lower() for c in t.get_cells()[0]]
    print(t.props['logic_str'])
    lf = replace_with_placeholders(header, lf)
    lf_str = linearize_lf(lf)
    print(lf)
    print(lf_str)
