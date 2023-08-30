import re

from pytimeparse import parse as parse_time
from dateparser import parse as parse_date
from dateutil.parser import ParserError
from dateparser.search import search_dates


NUM_REGEX = re.compile(r'^\s*([\-+]\s*)?\d+(\s*,\s*\d{3})*(\.\d+)?(\s*%)?\s*$')  # the most strict definition
NUM_INSIDE_STR_REGEX = re.compile(r'\b\d+(\s*,\s*\d{3})*(\.\d+)?\b')
STR_REGEX = re.compile(r'^\D+$')

# https://github.com/microsoft/PLOG/blob/d7964e86bb7fbd93e3c6045326402e75ed6122d3/execute/APIs.py#L306
MONTH_REGEX = re.compile(
    r"""
    \b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?
    |aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b
    """,
    flags=re.VERBOSE | re.I
)
YEAR_REGEX = re.compile(r"\b((?:1[789]|20)\d\d)\b")
SECONDS_ONLY_TIME = re.compile(r"^\s*\d\d?\.\d\d\s*$")
DATE_HEADER_REGEX = re.compile(r'\bdate\b', flags=re.I)
MOST_COMMON_DATE_TEMPLATE_REGEX = re.compile(r'\d\s*/\s*\d+\s*/\s*\d+')


def check_num(column, **kwargs):
    return all(
        NUM_REGEX.search(cell) is not None
        for cell in column
    )


def check_same_n_nums(column, **kwargs):
    lens = {
        len(NUM_INSIDE_STR_REGEX.findall(cell))
        for cell in column
    }
    return len(lens) == 1 and 0 not in lens


def check_str(column, **kwargs):
    return all(
        STR_REGEX.search(cell) is not None
        for cell in column
    )


def check_one_date(val, is_header_date):
    if NUM_REGEX.search(val) is not None:
        return False

    try:
        parsed_date = parse_date(
            val,
            settings={'PARSERS': ['absolute-time'], 'REQUIRE_PARTS': ['month']}
        )
    except ParserError:
        return False

    is_month_present = MONTH_REGEX.search(val) is not None
    is_full_year_present = YEAR_REGEX.search(val) is not None
    has_date_features = is_month_present or is_header_date or is_full_year_present
    looks_like_most_common_date = MOST_COMMON_DATE_TEMPLATE_REGEX.search(val) is not None

    if parsed_date is not None and (has_date_features or looks_like_most_common_date):
        return True

    # it should be present but not in this data, too many false positives
    # that are hard to distinguish, e.g. record: 20 - 10 - 1
    # elif parsed_date is not None:
    #     strict_date = parse_date(
    #         val,
    #         settings={'STRICT_PARSING': True, 'PARSERS': ['absolute-time']}
    #     )
    #     return strict_date is not None

    elif parsed_date is None and is_month_present:
        found_dates = search_dates(val)
        return found_dates is not None  # WILL TAKE THE FIRST ONE IN THE FUTURE

    return False


def check_date(column, header, **kwargs):
    is_header_date = DATE_HEADER_REGEX.search(header.value) is not None
    return all(check_one_date(cell, is_header_date) for cell in column)


def check_rank(column, header, **kwargs):
    if len(set(column)) == 1:
        return False

    is_header_rank = header.value.startswith('rank')
    for i, cell in enumerate(column):
        if not cell.isdigit():
            return False

        if not i or is_header_rank:
            continue

        diff = int(cell) - int(column[i-1])
        if diff < 0 or diff > 1:  # diff must be either 0 or 1
            return False

    return True


def check_year(column, **kwargs):
    # same_n_nums will return True for cases like 1875 or 2003 - 04
    # later, we can process this column if number of numbers is 1
    if not check_same_n_nums(column):
        return False

    return all(
        YEAR_REGEX.search(cell) is not None
        for cell in column
    )


def check_time(column, **kwargs):
    n_cells_w_colon = sum(1 for cell in column if ':' in cell)  # most cells look like 33:45
    all_times = all(
        parse_time(cell) is not None or SECONDS_ONLY_TIME.match(cell) is not None  # seconds only
        for cell in column
    )
    return all_times and n_cells_w_colon > len(column) // 2


CHECKS = {  # order matters - from strictest to loosest
    'date': check_date,
    'year': check_year,
    'rank': check_rank,
    'num': check_num,
    'time': check_time,
    'same_n_nums': check_same_n_nums,
    'str': check_str
}


def detect_column_type(column, header):
    column = [cell.value['raw'] for cell in column if cell.value['processed'] is not None]
    if not column:
        return 'empty'

    for check_name, check_func in CHECKS.items():
        if check_func(column=column, header=header):
            return check_name

    return 'mixed'


def detect_column_types_in_header(table):
    coltypes = []
    for col in zip(*table.cells):
        header, col = col[0], col[1:]
        coltype = detect_column_type(col, header)
        coltypes.append(coltype)

    return coltypes
