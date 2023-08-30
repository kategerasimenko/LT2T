import os
import sys
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(os.path.join(ROOT_DIR, 'BLEC'))

from BLEC.Logic2text import BLECLogic2text


BLEC_METRIC = BLECLogic2text()


def calc_blec(predictions, references, lfs):
    blec_scores = []
    for item_id, refs in tqdm(references.items()):
        # only one ref for lf2text, it's in the list for compat with bleu metric
        ref = refs[0]
        for pred in predictions[item_id]:
            lf = lfs[item_id]
            errors_tokens = BLEC_METRIC.evaluate(lf, pred, ref)
            is_correct = not bool(errors_tokens)
            blec_scores.append(int(is_correct))

    blec_score = sum(blec_scores) / len(blec_scores)
    return blec_score
