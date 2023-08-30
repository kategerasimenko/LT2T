import os
import pickle

import click

from config import DATA_DIR
from input_processing.parse_dataset import parse_tables


@click.command()
@click.option("--dataset", required=True, help="Dataset to parse: logicnlg or logic2text")
@click.option("--splits", help="Splits to load: train, dev, test. Default: all splits")
def parse_tables_cli(dataset, splits):
    if splits is not None:
        splits = splits.split(',')

    parsed = parse_tables(dataset, splits=splits)

    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, f'{dataset}_parsed.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(parsed, f)


if __name__ == '__main__':
    parse_tables_cli()
