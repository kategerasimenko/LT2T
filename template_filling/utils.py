import math
import numbers
from collections import defaultdict

import pandas as pd
import numpy as np

from .stopwords import STOPWORDS


def find_upper_func(lf, func):
    if lf['func'].startswith(func):
        return [lf]

    lfs = []
    for arg in lf['args']:
        if isinstance(arg, dict):
            lfs.extend(find_upper_func(arg, func))

    return lfs


def is_only_word_ngram(ngram):
    return (
        all(char.isalnum() for char in ''.join(ngram))
        and not any(w in STOPWORDS for w in ngram)
    )


def collect_ngrams_with_texts(texts):
    ngrams_with_texts = defaultdict(lambda: defaultdict(list))

    for text in texts:
        tokens = text.split(' ')

        for unigram in tokens:
            ngrams_with_texts[1][(unigram,)].append(text)

        for bigram in zip(tokens, tokens[1:]):
            ngrams_with_texts[2][bigram].append(text)

        for trigram in zip(tokens, tokens[1:], tokens[2:]):
            ngrams_with_texts[3][trigram].append(text)

    return ngrams_with_texts


def clean_up_and_calc_ngram_frequencies(ngrams_with_texts):
    ngram_frequencies = {
        n: {
            ' '.join(ngram): len(texts)  # save ngram frequency
            for ngram, texts in ngrams.items()
            if (
                is_only_word_ngram(ngram)  # leave out all ngrams with punctuation and stopwords
                and len(texts) >= 3  # number of cells with ngram >= 3
                and len(set(texts)) > 1  # if number of unique texts with ngram is 1, it's not valuable
            )                            # because the whole text can be taken instead
        }
        for n, ngrams in ngrams_with_texts.items()
    }

    return ngram_frequencies


def get_ngram_list_wo_repeats(ngram_frequencies):
    ngram_list = []

    for n, ngrams_freqs in ngram_frequencies.items():
        for ngram, freq in ngrams_freqs.items():
            is_substring_only = False
            for larger_ngram, lfreq in ngram_frequencies.get(n+1, {}).items():
                # skip if ngram is encountered only in larger_ngram (bc freqs are equal)
                if ngram in larger_ngram and freq == lfreq:
                    is_substring_only = True
                    break

            if not is_substring_only:
                ngram_list.extend([ngram] * freq)  # todo: why multiplying by freq?

    return ngram_list


def get_substrings(texts):
    ngrams_with_texts = collect_ngrams_with_texts(texts)
    ngram_frequencies = clean_up_and_calc_ngram_frequencies(ngrams_with_texts)
    ngram_list = get_ngram_list_wo_repeats(ngram_frequencies)
    return ngram_list


def get_processed_to_raw_mapping(table, col, preserve_all_raw=False):
    # todo: account for 1-to-many rels between proc and raw values
    col_values = table[col].replace([np.nan], [None])
    raw_col_values = table[col + '_raw']

    proc2raw = defaultdict(set)
    for p, r in zip(col_values, raw_col_values):
        proc2raw[p].add(r)

    for p, r in proc2raw.items():
        proc2raw[p] = sorted(r) if preserve_all_raw else sorted(r)[0]

    return proc2raw


def round_number(num):
    rounded = round(num, 2)
    if rounded == int(rounded):
        rounded = int(rounded)
    return rounded


def get_range_step(nums):
    """
    1. Get the most frequent number order (n digits)
    in the sequence of nums. If several orders have the same freq,
    return the largest one.

    2. Get the "rounding step" of num sequence
    based on the most frequent order and max number in that order:
        for 1-10, step is 1
        for 10-50 -       5
        for 10-100 -      10
        for 100-500 -     50
        for 100-1000 -    100
        etc.
    """
    orders = defaultdict(list)
    for num in nums:
        if pd.isna(num) or not isinstance(num, numbers.Number):
            continue

        if isinstance(num, float):
            num = round(num)

        num_order = len(str(abs(num))) - 1
        orders[num_order].append(abs(num))

    if not orders:
        return 1

    most_freq_order = max(orders.items(), key=lambda x: (len(x[1]), x[0]))[0]
    max_order_num = max(orders[most_freq_order])
    max_first_digit = int(str(max_order_num)[0])

    if most_freq_order == 0:
        step = 1
    elif max_first_digit <= 5:
        step = 10 ** (most_freq_order - 1) * 5
    else:
        step = 10 ** most_freq_order

    return step


def get_floor(number, step):
    if not isinstance(number, numbers.Number):  # date
        return number

    floor = step * math.floor(number / step)
    if number and floor == number:
        floor -= step
    return floor


def get_ceil(number, step):
    if not isinstance(number, numbers.Number):  # date
        return number

    ceil = step * math.ceil(number / step)
    if ceil == number:
        ceil += step
    return ceil
