from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

from input_processing.str2form import parse_str, linearize_lf
from .calc_config import (
    FUNCS, ALL_ROWS, PLACEHOLDER, ADDITIONAL_ARGS,
    COLTYPES_TO_REVISE
)
from .timeout_decorator import timeout


@dataclass
class FilledTemplates:
    lfs: list
    syntax_ok: bool


def revise_coltypes(table, coltypes):
    first_row = table.cells[1]

    for i, (cell, coltype) in enumerate(zip(first_row, coltypes)):
        if coltype not in COLTYPES_TO_REVISE or cell.value['processed'] is None:
            continue

        assert isinstance(cell.value['processed'], tuple)  # same_n_nums are parsed into num tuples
        if len(cell.value['processed']) == 1:  # save columns where there's only one num in same_n_nums
            coltypes[i] = COLTYPES_TO_REVISE[coltype]

    return coltypes


def get_processed_cells(table, coltypes):
    processed_cells = []

    for row in table.cells[1:]:
        row_cells = []

        for coltype, cell in zip(coltypes, row):
            processed_val = cell.value['processed']

            # freshly changed coltype, keep only the num
            if coltype in COLTYPES_TO_REVISE.values() and isinstance(processed_val, tuple):
                processed_val = processed_val[0]

            # if not empty and more nums, treat them as mixed - raw string
            elif coltype in COLTYPES_TO_REVISE and processed_val is not None:
                processed_val = cell.value['raw']

            row_cells.append(processed_val)

        processed_cells.append(row_cells)

    return processed_cells


def check_template(template, header_with_coltypes, upper_func=None):
    func_config = FUNCS.get(template['func'])
    if func_config is None:
        return False  # error in func name

    len_func_args = len(func_config['args'])
    func_input_types = func_config['type']
    n_content_args = 0
    n_placeholders = 0

    for i, arg in enumerate(template['args']):
        if isinstance(arg, dict):
            arg_result = check_template(arg, header_with_coltypes, upper_func=template['func'])
            if not arg_result:
                return False  # error in nested template

        elif arg in header_with_coltypes:
            if template['func'] == 'hop':  # upper_func must be defined and it determines if col in hop is valid
                func_input_types = FUNCS[upper_func].get('type', [])

            if func_input_types != 'all' and header_with_coltypes[arg] not in func_input_types:
                return False  # coltype doesn't match the requirement, formula invalid

        elif arg not in [ALL_ROWS, PLACEHOLDER]:
            return False  # word hallucination

        if arg == PLACEHOLDER:
            n_placeholders += 1
        else:
            n_content_args += 1

    for additional_arg in ADDITIONAL_ARGS:
        if additional_arg in func_config['args']:
            n_content_args += 1

    # w o r k a r o u n d
    # the only function that can have a nested lf + placeholder or two nested lfs
    if template['func'] == 'eq' and not n_placeholders:
        len_func_args += 1

    if len_func_args != n_content_args:
        return False  # n expected args doesn't match expected, formula invalid

    if n_placeholders > 1:
        return False  # there can be no or only one placeholder

    return True


def get_nested_func_output(table, parent_template, nested_template, filled_args, anscestor_func_info):
    if anscestor_func_info is None:
        anscestor_func_info = []

    parent_func_info = {
        'func': parent_template['func'],
        'raw_args': parent_template['args'],
        'filled_args': [x['arg'] for x in filled_args],
        'arg_lfs': [x['lf'] for x in filled_args]
    }
    options = fill_template_part(
        table=table,
        template_part=nested_template,
        upper_func_info=anscestor_func_info + [parent_func_info],
    )
    nested_output_options = [
        {'arg': option['output'], 'lf': option['lf']}
        for option in options
    ]
    return nested_output_options


def get_simple_arg_output(arg):
    return [{'arg': arg, 'lf': None}]  # lf is none because this arg is not a nested func
                                       # and doesn't produce nested lf


def produce_new_arg_combinations(prev_filled_args, new_option_list):
    new_arg_combs = []
    for option in new_option_list:
        copied_prev = deepcopy(prev_filled_args)
        copied_new = deepcopy(option)
        copied_prev.append(copied_new)
        new_arg_combs.append(copied_prev)
    return new_arg_combs


def create_all_argument_combinations(table, template, upper_func_info):
    filled_arg_combinations = [[]]

    for i, arg in enumerate(template['args']):
        all_combinations_for_arg = []

        for partially_filled in filled_arg_combinations:
            if isinstance(arg, dict):  # nested template
                curr_arg_options = get_nested_func_output(
                    table=table,
                    parent_template=template,
                    nested_template=arg,
                    filled_args=partially_filled,
                    anscestor_func_info=upper_func_info
                )

                # couldn't fill the nested template -> formula with these prev args is invalid, skipping
                if not curr_arg_options:
                    continue

            else:
                curr_arg_options = get_simple_arg_output(arg)

            # compile new argument combinations based on
            # one particular set of previous arguments and current argument options
            new_arg_combs = produce_new_arg_combinations(
                prev_filled_args=partially_filled,
                new_option_list=curr_arg_options
            )
            # add new combinations to all combinations for this arg
            # for all sets of previous arguments
            all_combinations_for_arg.extend(new_arg_combs)

        # no valid combinations for this argument -> the whole current formula is invalid
        if not all_combinations_for_arg:
            return []

        # overwriting bc `all_combinations_for_arg` has all arg combinations so far
        filled_arg_combinations = all_combinations_for_arg

    return filled_arg_combinations


def calculate_current_func(table, template, lf_args, upper_func_info):
    current_template_option = deepcopy(template)
    filled_template_options = []

    func_config = FUNCS[template['func']]
    func_argnames = func_config['args']
    func_obj = func_config['func']

    # --- "parsing" arguments ---

    content_args = []
    placeholder_idx = None
    for i, arg_info in enumerate(lf_args):
        arg_value, arg_lf = arg_info['arg'], arg_info['lf']

        if isinstance(arg_value, str) and arg_value == PLACEHOLDER:
            placeholder_idx = i  # there can be only one placeholder (checked in `check_template`)
        else:
            if isinstance(arg_value, str) and arg_value == ALL_ROWS:
                arg_value = table
            content_args.append(arg_value)
            if arg_lf is not None:
                current_template_option['args'][i] = arg_lf

    lf_kwargs = dict(zip(func_argnames, content_args))
    if 'upper_funcs' in func_argnames:
        lf_kwargs['upper_funcs'] = upper_func_info
    if 'children' in func_argnames:
        lf_kwargs['children'] = {
            'args': deepcopy(current_template_option['args']),
            'placeholder_idx': placeholder_idx
        }

    # --- getting func results ---

    func_output_options = func_obj(**lf_kwargs)

    for option in func_output_options:
        filled_lf = deepcopy(current_template_option)
        if option['filled'] is None and placeholder_idx is not None:
            continue  # a technical check but not strict assert because of possible broken template generations
        elif placeholder_idx is not None:  # this condition is only for eq when there can be lf + placeholder or lf + lf
            filled_lf['args'][placeholder_idx] = option['filled']

        filled_template_options.append({
            'output': option['output'],
            'lf': filled_lf
        })

    return filled_template_options


def fill_template_part(table, template_part, upper_func_info=None):
    all_arg_combinations = create_all_argument_combinations(table, template_part, upper_func_info)
    final_template_part_options = []

    for arg_combination in all_arg_combinations:
        arg_result_options = calculate_current_func(
            table=table,
            template=template_part,
            lf_args=arg_combination,
            upper_func_info=upper_func_info
        )
        final_template_part_options.extend(arg_result_options)

    return final_template_part_options


# @timeout(30)
def fill_template(template_str, table_obj, coltypes):
    try:
        template = parse_str(template_str, func_map={})
    except:
        return FilledTemplates(lfs=[], syntax_ok=False)

    coltypes = revise_coltypes(table_obj, coltypes)

    header = [c.value for c in table_obj.get_cells()[0]]
    raw_header = [c.value + '_raw' for c in table_obj.get_cells()[0]]
    header_with_coltypes = {h: c for h, c in zip(header, coltypes)}

    if not check_template(template, header_with_coltypes):
        return FilledTemplates(lfs=[], syntax_ok=False)

    cells = get_processed_cells(table_obj, coltypes)
    raw_cells = [  # for taking raw values for template filling
        [c.value['raw'] for c in row]
        for row in table_obj.get_cells()[1:]
    ]

    table = pd.concat((
        pd.DataFrame(cells, columns=header),
        pd.DataFrame(raw_cells, columns=raw_header)
    ), axis=1)

    template_filling_options = fill_template_part(table, template_part=template)
    lfs = [linearize_lf(filled['lf']) for filled in template_filling_options]
    lfs = sorted(set(lfs))  # removing duplicates

    return FilledTemplates(lfs=lfs, syntax_ok=True)
