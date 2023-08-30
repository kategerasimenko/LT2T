import random

import torch
import numpy as np

from .config import MAX_LENGTH


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def calc_truncated_and_unknown(df, tokenizer):
    n_truncated_inputs = 0
    n_truncated_outputs = 0

    unkn_inputs = 0
    total_inputs = 0
    unkn_outputs = 0
    total_outputs = 0

    for item in df:
        if len(item['input_ids']) == MAX_LENGTH:
            n_truncated_inputs += 1
        if len(item['labels']) == MAX_LENGTH:
            n_truncated_outputs += 1

        unkn_inputs += sum(1 for tok in item['input_ids'] if tok == tokenizer.unk_token_id)
        unkn_outputs += sum(1 for tok in item['labels'] if tok == tokenizer.unk_token_id)

        total_inputs += len(item['input_ids'])
        total_outputs += len(item['labels'])

    results = {
        'truncated_inputs': round(n_truncated_inputs / df.num_rows, 4),
        'truncated_outputs': round(n_truncated_outputs / df.num_rows, 4),
        'unknown_inputs': round(unkn_inputs / total_inputs, 4),
        'unknown_outputs': round(unkn_outputs / total_outputs, 4)
    }

    return results
