#!/usr/bin/env python
"""Plain causal-LM LoRA on the Sinhala textbook corpus -- no chat template.

    python textbook_cpt/build_textbook_cpt_dataset.py
    python textbook_cpt/run_textbook_cpt.py --config textbook_cpt/config.yaml
    python textbook_cpt/run_textbook_cpt.py --set train.learning_rate=5e-5

Every chapter document (build_textbook_cpt_dataset.py) is tokenized as
BOS + text + EOS, the documents are concatenated in a seeded random order and
cut into data.block_size blocks, and every token is a label -- the same
packing as cpt/run_clm_pt_with_peft.py, but over whole documents rather than
per line (no BOS before every Markdown line), and with eval on held-out
chapters instead of a random slice of the same blocks.

The run trains for the full train.num_train_epochs (no early stopping),
evaluates and checkpoints every eval_steps, always checkpoints the final step,
and loads the lowest-eval_loss checkpoint at the end (load_best_model_at_end).
"""

import argparse
import logging
import math
import os
import sys
from itertools import chain
from pathlib import Path

import datasets
import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sft"))
from run_sft_uc import load_config, resolve_path  # noqa: E402

logger = logging.getLogger(__name__)


class SaveFinalStep(TrainerCallback):
    """Evaluate + checkpoint the last step too, so the final adapter survives
    even when max_steps is not a multiple of save_steps."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control


def pack(path: str, tokenizer, block_size: int, seed: int, num_proc: int) -> datasets.Dataset:
    ds = datasets.load_dataset("parquet", data_files=path, split="train")
    ds = ds.shuffle(seed=seed)  # document order inside the packed stream
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id

    def tokenize(batch):
        ids = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
        return {"input_ids": [[bos] + x + [eos] for x in ids]}

    ds = ds.map(tokenize, batched=True, num_proc=num_proc, remove_columns=ds.column_names,
                desc="tokenize")
    # One batch over the whole stream: the corpus is small (~4.4M tokens), and
    # a single group keeps the only dropped remainder at the very end.
    stream = list(chain.from_iterable(ds["input_ids"]))
    n = len(stream) // block_size * block_size
    blocks = [stream[i:i + block_size] for i in range(0, n, block_size)]
    logger.info(f"{path}: {len(ds):,} docs, {len(stream):,} tokens -> {len(blocks):,} blocks "
                f"of {block_size} ({len(stream) - n:,} tail tokens dropped)")
    return datasets.Dataset.from_dict({"input_ids": blocks, "attention_mask": [[1] * block_size] * len(blocks),
                                       "labels": blocks})


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    return {"token_accuracy": float((preds[:, :-1] == labels[:, 1:]).mean())}


def preprocess_logits_for_metrics(logits, labels):
    return (logits[0] if isinstance(logits, tuple) else logits).argmax(dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "textbook_cpt" / "config.yaml"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    model_cfg, data_cfg, lora_cfg = cfg["model"], cfg["data"], cfg["lora"]
    train_cfg = dict(cfg["train"])
    train_cfg.pop("wandb_project", None)
    train_cfg["output_dir"] = resolve_path(train_cfg["output_dir"])

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
                        handlers=[logging.StreamHandler(sys.stdout)])
    transformers.utils.logging.set_verbosity_info()

    training_args = TrainingArguments(
        **train_cfg,
        do_train=True, do_eval=True,
        eval_strategy="steps", save_strategy="steps",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_strategy="steps", logging_first_step=True,
        label_names=["labels"],
    )
    if training_args.save_steps % training_args.eval_steps:
        raise ValueError("train.save_steps must be a multiple of train.eval_steps")
    set_seed(training_args.seed)

    model_path = resolve_path(model_cfg["path"])
    tokenizer = AutoTokenizer.from_pretrained(resolve_path(model_cfg.get("tokenizer_path") or model_cfg["path"]))
    assert tokenizer.bos_token_id is not None and tokenizer.eos_token_id is not None

    block_size = data_cfg["block_size"]
    num_proc = data_cfg.get("preprocessing_num_workers")
    with training_args.main_process_first(desc="pack"):
        train_ds = pack(resolve_path(data_cfg["train_file"]), tokenizer, block_size, training_args.seed, num_proc)
        eval_ds = pack(resolve_path(data_cfg["eval_file"]), tokenizer, block_size, training_args.seed, num_proc)
    logger.info("train block 0 (head): " + tokenizer.decode(train_ds[0]["input_ids"][:200]))

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        low_cpu_mem_usage=model_cfg.get("low_cpu_mem_usage", True),
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
    )
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()  # frozen embeddings + checkpointing need this
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["rank"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        modules_to_save=lora_cfg.get("modules_to_save") or None,
    ))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer, data_collator=default_data_collator,
        compute_metrics=compute_metrics, preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[SaveFinalStep()],
    )

    last = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last = get_last_checkpoint(training_args.output_dir)
        if last:
            logger.info(f"resuming from {last}")
    result = trainer.train(resume_from_checkpoint=last)
    trainer.save_model()  # the best checkpoint (load_best_model_at_end)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    metrics["perplexity"] = math.exp(metrics["eval_loss"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    logger.info(f"best checkpoint: {trainer.state.best_model_checkpoint} "
                f"(eval_loss={trainer.state.best_metric})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
