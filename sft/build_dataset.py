import logging
import os
from typing import Union, List

import datasets
import transformers
from datasets import load_dataset, concatenate_datasets

IGNORE_INDEX = -100

logger = logging.getLogger('__name__')

# UltraChat-Sinhala records are multi-turn:
#   {"prompt": str, "prompt_id": str,
#    "messages": [{"role": "user"|"assistant", "content": str}, ...]}
# We format every conversation with the Llama-3 chat template, mask the system
# and user turns (plus each assistant *header*), and compute the loss only on
# the assistant turns' content + their terminating <|eot_id|>. Loss is therefore
# trained across ALL assistant turns of a dialogue, not just the first.

# Short Sinhala assistant system prompt ("You are a helpful assistant.").
# Set to "" or None to train with no system turn (the original UltraChat-200k /
# zephyr SFT recipe injects no system message).
DEFAULT_SYSTEM_PROMPT = "ඔබ උපකාරශීලී සහායකයෙකි."

# Llama-3 chat formatting (the <|...|> markers are atomic special tokens).
system_format = '<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
user_format = '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>'
assistant_header = '<|start_header_id|>assistant<|end_header_id|>\n\n'
EOT = '<|eot_id|>'


def build_instruction_dataset(data_path: Union[List[str], str],
                              tokenizer: transformers.PreTrainedTokenizer,
                              max_seq_length: int, data_cache_dir=None,
                              preprocessing_num_workers=None,
                              ):

    # Precompute the pieces that never change per example.
    bos_id = tokenizer.bos_token_id
    assistant_header_ids = tokenizer(assistant_header, add_special_tokens=False)["input_ids"]
    if DEFAULT_SYSTEM_PROMPT:
        system_prefix_ids = tokenizer(
            system_format.format(content=DEFAULT_SYSTEM_PROMPT), add_special_tokens=False
        )["input_ids"]
    else:
        system_prefix_ids = []

    def tokenization(examples):
        all_input_ids = []
        all_labels = []
        for messages in examples["messages"]:
            input_ids = []
            labels = []
            if bos_id is not None:
                input_ids.append(bos_id)
                labels.append(IGNORE_INDEX)
            input_ids += system_prefix_ids
            labels += [IGNORE_INDEX] * len(system_prefix_ids)

            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "assistant":
                    # header is part of the prompt (masked); content + <|eot_id|> is supervised.
                    body_ids = tokenizer(content + EOT, add_special_tokens=False)["input_ids"]
                    input_ids += assistant_header_ids + body_ids
                    labels += [IGNORE_INDEX] * len(assistant_header_ids) + body_ids
                else:
                    fmt = system_format if role == "system" else user_format
                    turn_ids = tokenizer(fmt.format(content=content), add_special_tokens=False)["input_ids"]
                    input_ids += turn_ids
                    labels += [IGNORE_INDEX] * len(turn_ids)

            # Truncate from the end (keep the earliest turns). Examples whose
            # supervised tokens all fall past the budget are dropped below.
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
            # Drop dialogues left with no supervised (assistant) tokens after truncation.
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
