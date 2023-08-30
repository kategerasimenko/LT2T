import os
import json
import pickle

import click

from config import ROOT_DIR, DATA_DIR, TMP_DIR
from training_utils.seq2seq_training import train_seq2seq
from content_selection import (
    get_training_data,
    fix_tokenization,
    evaluate_preds
)


def pretrain(
        base_model,
        ckpt_dir,
        output_dir,
        parsed_tables,
        processing_params,
        add_main=False,
        additional_tokens=None

):
    train_data = get_training_data(
        parsed_tables,
        mode='train',
        splits=['train', 'dev'],
        add_main=add_main,
        add_generated=True,
        processing_params=processing_params
    )

    trainer, _ = train_seq2seq(
        train_data=train_data,
        task='content-selection',
        base_model=base_model,
        additional_tokens=additional_tokens,
        ckpt_dir=ckpt_dir,
        do_train=True,
        do_predict=False,
        epochs=1,
        batch_size=16,  # hardcoded for 40GB GPU mem
        learning_rate=1e-04,
        eval_steps=1000,
        model_name_suffix=f'_pretrain'
    )

    save_dir = os.path.join(output_dir, f'content-selection_{base_model}_pretrained')
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    return save_dir


@click.command()
@click.option("--do-train", is_flag=True, help="Run training")
@click.option("--do-predict", is_flag=True, type=bool, help="Run prediction on the best model")
@click.option("--do-ckpt-predict", is_flag=True, type=bool, help="Run prediction for dev on available checkpoints.")
@click.option("--do-pretrain", is_flag=True, type=bool, help="Whether to do pre-training on generated data.")
@click.option("--sources", default='main', type=str, help="Source of data - main, generated, or all. Default: main.")
@click.option("--balancing-strategy", type=str, help="Strategy of balancing: `oversample` or None (default).")
@click.option("--balancing-power", type=float, default=0.5, help="Power to apply to counts during balancing. Default: 0.5")
@click.option("--oversample-from-generated", is_flag=True, help="Whether to oversample from generated data.")
@click.option("--proc", default='v2', type=str, help="Data processing version. Default: v2.")
@click.option("--include-stats", is_flag=True, help="Whether to include column statistics in the input.")
@click.option("--include-num-stats", is_flag=True, help="Whether to include numerical column statistics in the input.")
@click.option("--include-value", is_flag=True, help="Whether to include value example in the input.")
@click.option("--base-model", default="t5-small", type=str, help="Base model to finetune")
@click.option("--epochs", default=30, type=int, help="Maximum number of epochs")
@click.option("--batch-size", default=16, type=int, help="Path to the output directory")
@click.option("--learning-rate", default=1e-4, type=float, help="Learning rate")
@click.option("--eval-steps", default=0, type=int, help="Training steps before evaluation. If 0, evaluation after every epoch.")
@click.option("--num-beams", default=1, type=int, help="Number of beams for generation.")
@click.option("--num-beam-groups", default=1, type=int, help="Number of beam groups for generation.")
@click.option("--temperature", default=1.0, type=float, help="Generation temperature.")
@click.option("--do-sample", is_flag=True, help="Whether to do sampling.")
@click.option("--top-k", default=50, type=int, help="Top k for sampling.")
@click.option("--top-p", default=1.0, type=float, help="Top p for sampling.")
@click.option("--n-generated", default=1, type=int, help="Number of sequences to be generated.")
@click.option("--ckpt-dir", default=os.path.join(TMP_DIR, "checkpoints"), type=str, help="Directory to store checkpoints")
@click.option("--output-dir", default=os.path.join(ROOT_DIR, "models"), type=str, help="Directory to store models and their outputs")
def run(
        do_train, do_predict, do_ckpt_predict, do_pretrain,
        sources, proc, include_stats, include_num_stats, include_value,
        balancing_strategy, balancing_power, oversample_from_generated,
        base_model, epochs, batch_size, learning_rate, eval_steps,
        num_beams, num_beam_groups, temperature, do_sample,
        top_k, top_p, n_generated, ckpt_dir, output_dir
):
    processing_params = {
        'processing_version': proc,
        'include_stats': include_stats,
        'include_num_stats': include_num_stats,
        'include_value': include_value
    }

    generation_params = {
        'num_beams': num_beams,
        'num_beam_groups': num_beam_groups,
        'temperature': temperature,
        'do_sample': do_sample,
        'top_k': top_k,
        'top_p': top_p,
        'num_return_sequences': n_generated
    }

    if num_beam_groups > 1:
        generation_params['diversity_penalty'] = 1.0

    neptune_params = {
        'sources': sources,
        'balance': f'{balancing_strategy}{balancing_power}_fromgen{oversample_from_generated}'
    }
    neptune_params.update(processing_params)

    model_suffix = f'_sources{sources}_proc{proc}'
    if include_stats:
        model_suffix += '_inclstats'
    if include_num_stats:
        model_suffix += '_inclnumstats'
    if include_value:
        model_suffix += '_inclval'

    model_suffix += f'_{balancing_strategy}{balancing_power}'
    if oversample_from_generated:
        model_suffix += '-fromgen'

    with open(os.path.join(DATA_DIR, f'logic2text_parsed.pkl'), 'rb') as f:
        parsed_tables = pickle.load(f)

    sources = sources.split(',')
    add_main = 'main' in sources or 'all' in sources
    add_generated = 'generated' in sources or 'all' in sources

    train_data = None
    predict_data = None
    additional_tokens = None

    if do_train:
        do_ckpt_predict = True  # will evaluate on all checkpoints and save the best one

        if base_model.startswith('t5-'):
            additional_tokens = {'new_tokens': ['{', '}']}

        if do_pretrain:
            base_model = pretrain(
                base_model=base_model,
                additional_tokens=additional_tokens,
                ckpt_dir=ckpt_dir,
                output_dir=output_dir,
                parsed_tables=parsed_tables,
                add_main=add_main,
                processing_params=processing_params
            )
            additional_tokens = None  # already in tokenizer

        train_data = get_training_data(
            parsed_tables,
            mode='train',
            splits=['train', 'dev'],
            add_main=add_main,
            add_generated=add_generated,
            balancing_strategy=balancing_strategy,
            balancing_power=balancing_power,
            oversample_from_generated=oversample_from_generated,
            processing_params=processing_params
        )

    # predict and evaluate
    if do_ckpt_predict or do_predict:
        predict_splits = ['dev']
        if do_predict:
            predict_splits.append('test')

        predict_data = get_training_data(
            parsed_tables,
            mode='eval',
            splits=predict_splits,
            processing_params=processing_params
        )

    trainer, save_dir = train_seq2seq(
        train_data=train_data,
        predict_data=predict_data,
        task='content-selection',
        base_model=base_model,
        additional_tokens=additional_tokens,
        eval_func=evaluate_preds,
        eval_kwargs={'parsed_tables': parsed_tables},
        score_to_maximize='exec_var_mean',
        output_dir=output_dir,
        ckpt_dir=ckpt_dir,
        do_train=do_train,
        do_predict=do_predict,
        do_ckpt_predict=do_ckpt_predict,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        eval_steps=eval_steps,
        generation_params=generation_params,
        postprocessing_func=fix_tokenization,
        model_name_suffix=model_suffix,
        neptune_params=neptune_params
    )

    with open(os.path.join(save_dir, 'processing_params.json'), 'w') as f:
        json.dump(processing_params, f, indent=2)


if __name__ == '__main__':
    run()
