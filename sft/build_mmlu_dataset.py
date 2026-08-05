#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt/completion tokenization for the MMLU-Sinhala LoRA SFT run.

Records are single-turn {"prompt": str, "completion": str, ...} produced by
build_mmlu_sft.py (training) / build_mmlu_validation_proxy.py (eval proxy) —
`prompt` is already rendered byte-identical to the template
mmlu/evaluate_sinhala_mmlu.py scores against (ending in "පිළිතුර: "), and
`completion` is the gold answer digit as a string ("1".."4"). Unlike
build_dataset.py's chat-message path, there is no Llama-3 chat-template
wrapping here: training format must match eval format exactly, since the
whole point is that the digit token trained is the digit token scored.
"""
import logging
import os
from typing import Union, List

import datasets
import transformers
from datasets import load_dataset, concatenate_datasets

IGNORE_INDEX = -100

logger = logging.getLogger('__name__')


def build_mmlu_dataset(data_path: Union[List[str], str],
                       tokenizer: transformers.PreTrainedTokenizer,
                       max_seq_length: int, data_cache_dir=None,
                       preprocessing_num_workers=None,
                       ):

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    def tokenization(examples):
        prompt_ids_batch = tokenizer(examples["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids_batch = tokenizer(examples["completion"], add_special_tokens=False)["input_ids"]
        all_input_ids = []
        all_labels = []
        for prompt_ids, completion_ids in zip(prompt_ids_batch, completion_ids_batch):
            input_ids = []
            labels = []
            if bos_id is not None:
                input_ids.append(bos_id)
                labels.append(IGNORE_INDEX)
            input_ids += prompt_ids                     # masked prompt
            labels += [IGNORE_INDEX] * len(prompt_ids)
            input_ids += completion_ids                  # supervised answer digit
            labels += completion_ids
            if eos_id is not None:
                input_ids.append(eos_id)                 # supervised EOS
                labels.append(eos_id)

            all_input_ids.append(input_ids[:max_seq_length])
            all_labels.append(labels[:max_seq_length])

        return {"input_ids": all_input_ids, "labels": all_labels}

    logging.warning("building dataset...")
    all_datasets = []

    if not isinstance(data_path, (list, tuple)):
        data_path = [data_path]
    for file in data_path:
        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
        cache_path = os.path.join(
            data_cache_dir, os.path.basename(file).split('.')[0] + f"_{max_seq_length}")
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'training datasets-{file} has been loaded from disk')
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            tokenized_dataset = raw_dataset.map(
                tokenization,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=raw_dataset["train"].column_names,
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            # Drop examples left with no supervised (answer) tokens after truncation.
            tokenized_dataset = tokenized_dataset.filter(
                lambda ex: any(l != IGNORE_INDEX for l in ex["labels"]),
                num_proc=preprocessing_num_workers,
                desc="dropping examples with no answer tokens",
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets
