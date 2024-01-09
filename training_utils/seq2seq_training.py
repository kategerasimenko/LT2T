import os
import re
import json
import random
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from unidecode import unidecode
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
    Seq2SeqTrainer, GenerationConfig, EarlyStoppingCallback
)
from transformers.integrations import NeptuneCallback

from .config import (
    SEED, ROOT_DIR, TMP_DIR,
    EVAL_PARTS, MAX_LENGTH, LABEL_PAD_TOKEN_ID
)
from .utils import set_seed, calc_truncated_and_unknown


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_REGEX = re.compile(r'\d+')


def preprocess_data(ds, tokenizer, mode):
    def tokenize(examples):
        inputs = [unidecode(inp) for inp in examples['input']]  # to reduce OOV issue (with T5 at least)
        model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)

        if mode == 'train':
            outs = [unidecode(inp) for inp in examples['output']]  # to reduce OOV issue with T5 (with T5 at least)
            labels = tokenizer(text_target=outs, max_length=MAX_LENGTH, truncation=True)
            model_inputs['labels'] = labels['input_ids']

        return model_inputs

    ds = ds.map(tokenize, batched=True)
    return ds


def train(
        ds,
        model_name,
        base_model,
        model_ckpt_dir,
        eval_steps,
        training_params,
        neptune_callback,
        additional_tokens=None,
        patience=0
):
    print(f'Fine-tuning {model_name}')
    neptune_callback.run['mode/train'] = True

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if additional_tokens is not None:
        tokenizer.add_tokens(**additional_tokens)
        model.resize_token_embeddings(len(tokenizer))

    ds = preprocess_data(ds, tokenizer, mode='train')
    ds['train'] = ds['train'].shuffle(seed=SEED)

    # to control OOV tokens and whether the data fits into the predefined length limit
    token_problem_stats = calc_truncated_and_unknown(ds['train'], tokenizer)
    print(json.dumps(token_problem_stats, indent=2))

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=LABEL_PAD_TOKEN_ID
    )

    if eval_steps > 0:
        eval_params = {
            'evaluation_strategy': 'steps',
            'eval_steps': eval_steps,
            'save_strategy': 'steps',
            'save_steps': eval_steps
        }
    else:
        eval_params = {
            'evaluation_strategy': 'epoch',
            'save_strategy': 'epoch',
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_ckpt_dir,
        report_to='none',
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        load_best_model_at_end=True,
        **training_params,
        **eval_params
    )

    callbacks = [neptune_callback]
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['dev'],
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks
    )

    trainer.train()
    return trainer


def predict(
        dataset,
        model,
        tokenizer,
        batch_size,
        generation_params,
        postprocessing_func=None,
        eval_func=None,
        eval_kwargs=None
):
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=LABEL_PAD_TOKEN_ID
    )

    if generation_params['do_sample']:
        batch_size = 1  # for deterministic generation when sampling

    tokens = dataset.remove_columns([c for c in dataset.column_names if c not in tokenizer.model_input_names])
    test_dataloader = DataLoader(tokens, batch_size=batch_size, collate_fn=collator)

    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # for deterministic generation when sampling
            set_seed(SEED)

            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            preds = model.generate(
                **batch,
                **generation_params,
                max_length=MAX_LENGTH
            )
            decoded_preds = tokenizer.batch_decode(
                preds,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False  # all refs are pre-tokenized
            )

            # group flattened preds
            n_generated = generation_params['num_return_sequences']
            for i in range(0, len(decoded_preds), n_generated):
                decoded_sample = decoded_preds[i:i + n_generated]
                if postprocessing_func is not None:
                    decoded_sample = [postprocessing_func(s) for s in decoded_sample]
                all_preds.append({'out': decoded_sample})

    print('Prediction sample')
    print(all_preds[0]['out'][:3])

    if eval_func is not None:
        if eval_kwargs is None:
            eval_kwargs = {}

        scores = eval_func(all_preds, dataset, **eval_kwargs)
        return all_preds, scores

    return all_preds, None


def run_inference(
        ds,
        model_path,
        data_part,
        batch_size,
        generation_params,
        postprocessing_func=None,
        eval_func=None,
        eval_kwargs=None,
        write_predictions=True,
        write_scores=True,
        preds_path=None
):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    ds = preprocess_data(ds, tokenizer, mode='inference')

    preds, scores = predict(
        dataset=ds,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        generation_params=generation_params,
        postprocessing_func=postprocessing_func,
        eval_func=eval_func,
        eval_kwargs=eval_kwargs
    )

    params_str = '_'.join(f'{k.replace("_", "-")}{v}' for k, v in generation_params.items())

    if write_predictions:
        with open(os.path.join(preds_path, f'preds_{params_str}_{data_part}.jsonl'), 'w') as f:
            f.write('\n'.join(json.dumps(pred, ensure_ascii=False) for pred in preds))

    if eval_func is not None:
        str_scores = '\n'.join(f'{k}: {v}' for k, v in scores.items())
        print(str_scores)

        if write_scores:
            with open(os.path.join(preds_path, f'score_{params_str}_{data_part}.txt'), 'w') as f:
                f.write(str_scores)

    return preds, scores


def save_best_checkpoint(
        score_vals,
        score_to_maximize,
        ckpts_dir,
        model_save_dir,
        neptune_callback,
        part='dev'
):
    main_scores = score_vals[score_to_maximize][part]
    max_ckpt, max_score = max(main_scores.items(), key=lambda x: x[1])
    ckpt_to_save = os.path.join(ckpts_dir, max_ckpt)

    best_ckpt_model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_to_save).to(DEVICE)
    best_ckpt_tokenizer = AutoTokenizer.from_pretrained(ckpt_to_save)

    best_ckpt_model.save_pretrained(model_save_dir)
    best_ckpt_tokenizer.save_pretrained(model_save_dir)

    neptune_callback.run[f'scores/{part}/max_ckpt_{score_to_maximize}'] = max_score
    neptune_callback.run[f'scores/{part}/argmax_ckpt_{score_to_maximize}'] = max_ckpt
    print(f'max score {part}: {max_score} in {max_ckpt}')


def evaluate_checkpoints(
        ds,
        ckpts_dir,
        generation_params,
        batch_size,
        eval_func,
        neptune_callback,
        eval_kwargs=None,
        postprocessing_func=None
):
    score_vals = defaultdict(lambda: defaultdict(dict))
    part = 'dev'  # for checkpoints, predict on dev only

    ckpts = os.listdir(ckpts_dir)
    ckpt_nums = [int(NUM_REGEX.search(ckpt).group()) for ckpt in ckpts]
    sorted_ckpts = sorted(zip(ckpts, ckpt_nums), key=lambda x: x[1])

    for ckpt_name, ckpt_num in sorted_ckpts:
        print(f'Running prediction of {ckpt_name} on {part}')

        _, scores = run_inference(
            ds,
            model_path=os.path.join(ckpts_dir, ckpt_name),
            data_part=part,
            batch_size=batch_size,
            generation_params=generation_params,
            postprocessing_func=postprocessing_func,
            eval_func=eval_func,
            eval_kwargs=eval_kwargs,
            write_predictions=False,
            write_scores=False
        )
        for score_name, score_val in scores.items():
            neptune_callback.run[f'scores/{part}/ckpt_{score_name}'].log(score_val, step=ckpt_num)
            score_vals[score_name][part][ckpt_name] = score_val

    return score_vals


def train_seq2seq(
        task,
        eval_func=None,
        score_to_maximize=None,
        eval_kwargs=None,
        train_data=None,
        predict_data=None,
        base_model='t5-small',
        additional_tokens=None,
        output_dir=os.path.join(ROOT_DIR, "models"),
        ckpt_dir=os.path.join(TMP_DIR, "checkpoints"),
        do_train=True,
        do_predict=True,
        do_ckpt_predict=False,
        epochs=30,
        batch_size=16,
        learning_rate=1e-4,
        eval_steps=0,
        patience=0,
        generation_params=None,
        postprocessing_func=None,
        model_name_suffix=None,
        neptune_params=None
):
    set_seed(SEED)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    model_name = f'{task}_{base_model.rsplit("/", 1)[-1]}_{epochs}e_{batch_size}bs_{learning_rate}lr'
    if model_name_suffix is not None:
        model_name += model_name_suffix

    save_dir = os.path.join(output_dir, model_name)
    model_ckpt_dir = os.path.join(ckpt_dir, model_name)
    model_save_dir = os.path.join(save_dir, "model")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "preds"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'scores'), exist_ok=True)

    if train_data is not None:
        print('TRAIN:')
        print(train_data)
        print('SAMPLE:')
        existing_key = list(train_data.keys())[0]
        for k, v in train_data[existing_key][0].items():
            print(f'{k}: {v}')

    if predict_data is not None:
        print('PREDICT:')
        print(predict_data)
        print('SAMPLE:')
        existing_key = list(predict_data.keys())[0]
        for k, v in predict_data[existing_key][0].items():
            print(f'{k}: {v}')

    training_params = {
        'per_device_train_batch_size': batch_size,
        'per_device_eval_batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_train_epochs': epochs
    }
    if generation_params is None:
        generation_params = {}

    params_str = '_'.join(f'{k.replace("_", "-")}{v}' for k, v in generation_params.items())

    neptune_callback = NeptuneCallback(
        tags=[model_name],
        project=f"{os.environ['NEPTUNE_ORG']}/{task}"  # env var must be defined for Neptune anyway
    )
    neptune_callback.run['model'] = model_name
    neptune_callback.run['base_model'] = base_model
    neptune_callback.run['parameters/batch_size'] = batch_size
    neptune_callback.run['parameters/learning_rate'] = learning_rate
    neptune_callback.run['parameters/eval_steps'] = eval_steps

    if neptune_params is not None:
        for k, v in neptune_params.items():
            neptune_callback.run[f'parameters/{k}'] = v

    for k, v in generation_params.items():
        neptune_callback.run[f'generation/{k}'] = v

    trainer = None

    if do_train:
        trainer = train(
            ds=train_data,
            model_name=model_name,
            base_model=base_model,
            additional_tokens=additional_tokens,
            model_ckpt_dir=model_ckpt_dir,
            eval_steps=eval_steps,
            patience=patience,
            training_params=training_params,
            neptune_callback=neptune_callback
        )

    if do_ckpt_predict:
        neptune_callback.run['mode/ckpt_predict'] = True

        scores_file = os.path.join(ROOT_DIR, 'scores', f'{model_name}_{params_str}.json')
        score_vals = evaluate_checkpoints(
            predict_data['dev'],
            ckpts_dir=model_ckpt_dir,
            generation_params=generation_params,
            batch_size=batch_size,
            eval_func=eval_func,
            eval_kwargs=eval_kwargs,
            postprocessing_func=postprocessing_func,
            neptune_callback=neptune_callback
        )

        save_best_checkpoint(score_vals, score_to_maximize, model_ckpt_dir, model_save_dir, neptune_callback)
        with open(scores_file, 'w') as f:
            json.dump(score_vals, f, indent=2)

    if do_predict:
        neptune_callback.run['mode/predict'] = True

        for part in EVAL_PARTS:
            print(f'Running prediction of best model on {part}')
            # scores and preds are saved in model folder
            preds, scores = run_inference(
                predict_data[part],
                model_path=model_save_dir,
                data_part=part,
                batch_size=batch_size,
                generation_params=generation_params,
                postprocessing_func=postprocessing_func,
                eval_func=eval_func,
                eval_kwargs=eval_kwargs,
                write_predictions=True,
                write_scores=True,
                preds_path=os.path.join(save_dir, "preds")
            )
            for k, v in scores.items():
                neptune_callback.run[f'scores/{part}/best_model_{k}'] = v

    if do_ckpt_predict or do_predict:
        with open(os.path.join(save_dir, f'generation_{params_str}.json'), 'w') as f:
            json.dump(generation_params, f, indent=2)

    return trainer, save_dir
