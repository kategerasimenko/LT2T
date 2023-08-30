import re
import os
import json
from collections import defaultdict

import gdown


JSON_VALUE_REGEX = re.compile(r'\s+".+?": "(.+?)"')


def save_generated_dataset(filename):
    tmp_filename = 'raw_generated_data.json'
    file_id = '1Q_1HuSUmNjbNSr-XrJCqbAVUy0QlGTpy'
    gdown.download(id=file_id, output=tmp_filename, quiet=False)

    table_ids = []
    logic_strs = []

    with open(tmp_filename) as f:
        for line in f:
            if '"url":' in line:
                val = JSON_VALUE_REGEX.search(line).group(1)
                table_id = val.rsplit('/', 1)[-1]
                table_ids.append(table_id)
            elif '"logic_str":' in line:
                val = JSON_VALUE_REGEX.search(line).group(1)
                logic_strs.append(val)

    assert len(table_ids) == len(logic_strs)

    dataset = defaultdict(list)
    for table_id, lf in zip(table_ids, logic_strs):
        dataset[table_id].append(lf)

    with open(filename, 'w') as f:
        json.dump(dataset, f, ensure_ascii=False)

    os.remove(tmp_filename)
