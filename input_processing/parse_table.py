import re

from pytimeparse import parse as timeparser_parse_time
from dateparser import parse as dateparser_parse_date
from dateparser.search import search_dates as dateparser_search_dates

from input_processing.detect_column_types import detect_column_types_in_header


EMPTY = re.compile(r'^\s*(|-|n\s*[\\/]\s*a)\s*$')
NUM_INSIDE_STR_REGEX = re.compile(r'\b\d+(?:\s*,\s*\d{3})*(?:\.\d+)?\b')
TOTAL_REGEX = re.compile(r'\b((o ?v ?e ?r ?)?a ?l ?l|t ?o ?t ?a ?l( ?s)?|s ?u ?m)\b')


def create_combined_values_and_rm_empty(table):
    for i, row in enumerate(table.get_cells()):
        if not i:  # for headers, we keep orig values
            continue

        for j, cell in enumerate(row):
            combined_value = {
                'raw': cell.value,
                'processed': cell.value if not EMPTY.match(cell.value) else None
            }
            table.cells[i][j].value = combined_value
    return table


def parse_number(num_str):
    num_str = num_str.strip(' %').replace(',', '').replace(' ', '')
    if '.' in num_str:
        return float(num_str)
    return int(num_str)


def parse_numbers(num_str):
    nums = NUM_INSIDE_STR_REGEX.findall(num_str)
    return tuple(parse_number(num) for num in nums)


def parse_date(date_str):
    parsed_date = dateparser_parse_date(
        date_str,
        settings={'PARSERS': ['absolute-time'], 'REQUIRE_PARTS': ['month']}
    )
    if parsed_date is None:  # date MUST be there, checked on prevous step
        parsed_date = dateparser_search_dates(date_str)[0][1]
    return parsed_date


def parse_time(time_str):
    parsed = timeparser_parse_time(time_str)
    if parsed is None:  # only seconds (checked with regex in coltype clf)
        parsed = float(time_str.strip())
    return parsed


PARSE_FUNCS = {
    'num': parse_number,
    'rank': parse_number,
    'year': parse_numbers,
    'same_n_nums': parse_numbers,
    'date': parse_date,
    'time': parse_time
}


def is_row_sum(first_cell):
    return (
        first_cell['processed'] is not None
        and TOTAL_REGEX.search(first_cell['raw'].lower()) is not None
    )


def is_row_repeating_header(row, header):
    return all(cell.value['raw'] == header_cell for cell, header_cell in zip(row, header))


def parse_table(table):
    table = create_combined_values_and_rm_empty(table)
    header = [c.value.lower().strip() for c in table.cells[0]]
    rows_to_remove = set()

    for i, row in enumerate(table.get_cells()):
        # remove rows which repeat header (e.g. train 66)
        if i and is_row_repeating_header(row, header):
            rows_to_remove.add(i)

    # remove summation row
    if is_row_sum(table.cells[-1][0].value):
        rows_to_remove.add(len(table.cells) - 1)

    table.cells = [
        row
        for i, row in enumerate(table.cells)
        if i not in rows_to_remove
    ]

    coltypes = detect_column_types_in_header(table)

    for i, row in enumerate(table.get_cells()):
        if not i:  # header
            continue

        for j, cell in enumerate(row):
            if coltypes[j] in PARSE_FUNCS and cell.value['processed'] is not None:
                table.cells[i][j].value['processed'] = PARSE_FUNCS[coltypes[j]](cell.value['raw'])

    return table, coltypes
