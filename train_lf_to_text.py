import os
import pickle

import click

from config import ROOT_DIR, DATA_DIR, TMP_DIR
from training_utils.seq2seq_training import train_seq2seq
from lf_to_text import (
    prepare_lf2text_data,
    evaluate_preds
)


@click.command()
@click.option("--do-train", is_flag=True, help="Run training")
@click.option("--do-predict", is_flag=True, type=bool, help="Run prediction on the best model")
@click.option("--do-ckpt-predict", is_flag=True, type=bool, help="Run prediction for dev on available checkpoints.")
@click.option("--base-model", default="t5-small", type=str, help="Base model to finetune")
@click.option("--epochs", default=30, type=int, help="Maximum number of epochs")
@click.option("--batch-size", default=16, type=int, help="Path to the output directory")
@click.option("--learning-rate", default=1e-4, type=float, help="Learning rate")
@click.option("--eval-steps", default=0, type=int, help="Training steps before evaluation. If 0, evaluation after every epoch.")
@click.option("--num-beams", default=1, type=int, help="Number of beams for generation.")
@click.option("--temperature", default=1.0, type=float, help="Generation temperature.")
@click.option("--do-sample", is_flag=True, help="Whether to do sampling.")
@click.option("--top-k", default=50, type=int, help="Top k for sampling.")
@click.option("--n-generated", default=1, type=int, help="Number of sequences to be generated.")
@click.option("--ckpt-dir", default=os.path.join(TMP_DIR, "checkpoints"), type=str, help="Directory to store checkpoints")
@click.option("--output-dir", default=os.path.join(ROOT_DIR, "models"), type=str, help="Directory to store models and their outputs")
def run(
        do_train, do_predict, do_ckpt_predict, base_model,
        epochs, batch_size, learning_rate, eval_steps,
        num_beams, temperature, do_sample, top_k, n_generated,
        ckpt_dir, output_dir
):
    with open(os.path.join(DATA_DIR, f'logic2text_parsed.pkl'), 'rb') as f:
        parsed_tables = pickle.load(f)

    splits = set()

    if do_train:
        do_ckpt_predict = True  # will evaluate on all checkpoints and save the best one
        splits |= {'train', 'dev'}

    if do_ckpt_predict:
        splits.add('dev')

    if do_predict:
        splits |= {'dev', 'test'}

    data = prepare_lf2text_data(
        parsed_tables,
        splits=splits,
        mode='train'
    )

    if base_model.startswith('t5-'):
        additional_tokens = {'new_tokens': ['{', '}']}
    else:
        additional_tokens = None

    generation_params = {
        'num_beams': num_beams,
        'temperature': temperature,
        'do_sample': do_sample,
        'top_k': top_k,
        'num_return_sequences': n_generated
    }

    train_seq2seq(
        train_data=data,
        predict_data=data,
        task='lf-to-text',
        base_model=base_model,
        additional_tokens=additional_tokens,
        eval_func=evaluate_preds,
        score_to_maximize='sacrebleu',
        output_dir=output_dir,
        ckpt_dir=ckpt_dir,
        do_train=do_train,
        do_predict=do_predict,
        do_ckpt_predict=do_ckpt_predict,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        eval_steps=eval_steps,
        generation_params=generation_params
    )


if __name__ == '__main__':
    run()
