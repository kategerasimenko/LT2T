import os
from pathlib import Path


SEED = 42

ROOT_DIR = Path(__file__).parent.parent
TMP_DIR = os.environ.get('TMPDIR', ROOT_DIR)

EVAL_PARTS = ['dev', 'test']

MAX_LENGTH = 512
LABEL_PAD_TOKEN_ID = -100
