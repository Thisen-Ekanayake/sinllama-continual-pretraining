"""
LoRA SFT finetuning of the merged SinLlama model on the Sinhala
Writing-Style-Classification task.

Dataset: data/Writing/writing_style_{train,val,test}.csv
  columns (whitespace-padded in the header): "comments", "labels", "length"
  labels: ACADEMIC, CREATIVE, NEWS, BLOG

The model is trained as a causal LM: the task prompt is masked (labels=-100)
and the loss is only computed on the answer tokens. Logs to Weights & Biases.

Example:
    python finetune/finetune_writing_style.py \
        --model_name_or_path models/SinLlama_cpt_merged \
        --data_dir data/Writing \
        --output_dir runs/writing_style_lora
"""
import argparse
import os

# Avoid the "tokenizers fork after parallelism" warning/deadlock with DataLoader workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# ----------------------------------------------------------------------------
# Task definition
# ----------------------------------------------------------------------------
TASK = "writing"
TEXT_COL = "comments"
LABEL_COL = "labels"
LABELS = ["ACADEMIC", "CREATIVE", "NEWS", "BLOG"]

PROMPT_PREFIX = (
    "You are an NLP assistant whose purpose is to classify Sinhala comments "
    "into predefined categories. The categories are as follows: ACADEMIC, "
    "CREATIVE, NEWS, BLOG. Given a Sinhala comment, your task is to determine "
    "which category it belongs to. Choose one category: ACADEMIC, CREATIVE, "
    "NEWS, or BLOG. Answer must match exactly in capitalization and formatting.\n"
    "Comment: "
)
PROMPT_SUFFIX = "\nAnswer:"


def build_answer(label_value) -> str:
    # leading space so the completion is " ACADEMIC"
    return " " + str(label_value).strip()


def parse_prediction(generated_text: str):
    t = generated_text.strip().upper()
    for lab in LABELS:
        if t.startswith(lab):
            return lab
    for lab in LABELS:
        if lab in t:
            return lab
    return None


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class SFTDataset(Dataset):
    """Tokenizes prompt+answer; masks the prompt so loss is on the answer only."""

    def __init__(self, df, tokenizer, max_seq_length):
        self.examples = []
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        prefix_ids = tokenizer(PROMPT_PREFIX, add_special_tokens=False)["input_ids"]
        suffix_ids = tokenizer(PROMPT_SUFFIX, add_special_tokens=False)["input_ids"]

        for text, label in zip(df[TEXT_COL].tolist(), df[LABEL_COL].tolist()):
            answer_ids = tokenizer(build_answer(label), add_special_tokens=False)["input_ids"]
            # budget for the (potentially very long) comment
            budget = max_seq_length - len(prefix_ids) - len(suffix_ids) - len(answer_ids) - 2
            budget = max(budget, 0)
            comment_ids = tokenizer(str(text), add_special_tokens=False)["input_ids"][:budget]

            prompt_ids = [bos] + prefix_ids + comment_ids + suffix_ids
            input_ids = prompt_ids + answer_ids + [eos]
            labels = [-100] * len(prompt_ids) + answer_ids + [eos]
            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DataCollator:
    """Right-pads input_ids/labels to the longest sequence in the batch."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_split(data_dir, name):
    df = pd.read_csv(os.path.join(data_dir, name))
    df.columns = [c.strip() for c in df.columns]            # header has padding
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()   # values have padding
    df = df[df[LABEL_COL].isin(LABELS)].reset_index(drop=True)
    return df


@torch.no_grad()
def evaluate_accuracy(model, tokenizer, df, max_seq_length, batch_size, device):
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"

    prefix_ids = tokenizer(PROMPT_PREFIX, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(PROMPT_SUFFIX, add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id

    preds, golds = [], []
    texts = df[TEXT_COL].tolist()
    labels = df[LABEL_COL].tolist()
    for start in range(0, len(texts), batch_size):
        batch_prompts = []
        for text in texts[start:start + batch_size]:
            budget = max_seq_length - len(prefix_ids) - len(suffix_ids) - 8
            comment_ids = tokenizer(str(text), add_special_tokens=False)["input_ids"][:max(budget, 0)]
            batch_prompts.append([bos] + prefix_ids + comment_ids + suffix_ids)

        max_len = max(len(p) for p in batch_prompts)
        input_ids, attn = [], []
        for p in batch_prompts:
            pad = max_len - len(p)
            input_ids.append([tokenizer.pad_token_id] * pad + p)
            attn.append([0] * pad + [1] * len(p))
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        attn = torch.tensor(attn, dtype=torch.long, device=device)

        out = model.generate(
            input_ids=input_ids, attention_mask=attn,
            max_new_tokens=6, do_sample=False, num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, input_ids.shape[1]:]
        for g in gen:
            preds.append(parse_prediction(tokenizer.decode(g, skip_special_tokens=True)))
        golds.extend(labels[start:start + batch_size])

    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    acc = correct / len(golds) if golds else 0.0
    # per-label accuracy
    per_label = {}
    for lab in LABELS:
        idx = [i for i, g in enumerate(golds) if g == lab]
        if idx:
            per_label[lab] = sum(1 for i in idx if preds[i] == lab) / len(idx)
    return acc, per_label, preds, golds


def write_results_report(path, args, train_result, trainer, acc, per_label, samples):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    best_eval_loss = getattr(trainer.state, "best_metric", None)
    global_step = getattr(trainer.state, "global_step", None)
    tm = train_result.metrics if train_result is not None else {}
    lines = []
    lines.append("=" * 70)
    lines.append(f"TASK: {TASK}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("[CONFIG]")
    lines.append(f"  model_name_or_path      : {args.model_name_or_path}")
    lines.append(f"  data_dir                : {args.data_dir}")
    lines.append(f"  train/val/test files    : {args.train_file} / {args.val_file} / {args.test_file}")
    lines.append(f"  max_seq_length          : {args.max_seq_length}")
    lines.append(f"  lora_rank/alpha/dropout : {args.lora_rank} / {args.lora_alpha} / {args.lora_dropout}")
    lines.append(f"  lora_target_modules     : {args.lora_target_modules}")
    lines.append(f"  num_train_epochs        : {args.num_train_epochs}")
    lines.append(f"  learning_rate           : {args.learning_rate}")
    lines.append(f"  train/eval batch size   : {args.per_device_train_batch_size} / {args.per_device_eval_batch_size}")
    lines.append(f"  gradient_accum_steps    : {args.gradient_accumulation_steps}")
    lines.append(f"  warmup_ratio            : {args.warmup_ratio}")
    lines.append(f"  weight_decay            : {args.weight_decay}")
    lines.append(f"  load_in_4bit            : {args.load_in_4bit}")
    lines.append(f"  seed                    : {args.seed}")
    lines.append("")
    lines.append("[TRAINING]")
    lines.append(f"  total_train_steps       : {global_step}")
    lines.append(f"  best_eval_loss          : {best_eval_loss}")
    lines.append(f"  train_runtime_sec       : {tm.get('train_runtime')}")
    lines.append(f"  train_loss              : {tm.get('train_loss')}")
    lines.append("")
    lines.append("[TEST RESULTS]")
    lines.append(f"  overall_accuracy        : {acc:.4f}")
    for lab, a in per_label.items():
        lines.append(f"  acc[{lab}]".ljust(26) + f": {a:.4f}")
    lines.append("")
    lines.append("[SAMPLE PREDICTIONS]  (gold | pred | text)")
    for text, gold, pred in samples:
        snippet = str(text).replace("\n", " ")[:90]
        lines.append(f"  gold={str(gold):10s} pred={str(pred):10s} | {snippet}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote results report -> {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="models/SinLlama_cpt_merged")
    p.add_argument("--data_dir", default="data/Writing")
    p.add_argument("--train_file", default="writing_style_train.csv")
    p.add_argument("--val_file", default="writing_style_val.csv")
    p.add_argument("--test_file", default="writing_style_test.csv")
    p.add_argument("--output_dir", default="runs/writing_style_lora")
    p.add_argument("--results_file", default=None,
                   help="Where to write the final report (default: <output_dir>/results_writing.txt).")
    p.add_argument("--max_seq_length", type=int, default=1024)
    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj")
    # training
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)
    # ON by default: an 8B model + bs8/seq512 + the large 139k-vocab fp32 logits does
    # NOT fit on a 44GB A40 without checkpointing (OOMs in the MLP). Disable with
    # --no-gradient_checkpointing only on a bigger GPU or with a smaller batch.
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    # wandb
    p.add_argument("--wandb_project", default="sinllama-finetune")
    p.add_argument("--run_name", default="writing-style-lora")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["WANDB_PROJECT"] = args.wandb_project
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing)
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.lora_target_modules.split(","),
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        inference_mode=False,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_df = load_split(args.data_dir, args.train_file)
    val_df = load_split(args.data_dir, args.val_file)
    test_df = load_split(args.data_dir, args.test_file)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_ds = SFTDataset(train_df, tokenizer, args.max_seq_length)
    val_ds = SFTDataset(val_df, tokenizer, args.max_seq_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=True,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["wandb"],
        run_name=args.run_name,
        seed=args.seed,
        dataloader_num_workers=4,
        # Batch sequences of similar length together to cut padding waste.
        group_by_length=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollator(tokenizer.pad_token_id),
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Final generation-based accuracy on the test set
    print("Evaluating exact-match accuracy on the test set ...")
    acc, per_label, preds, golds = evaluate_accuracy(
        model, tokenizer, test_df, args.max_seq_length,
        args.per_device_eval_batch_size, model.device)
    print(f"TEST accuracy = {acc:.4f}")
    for lab, a in per_label.items():
        print(f"  {lab:10s}: {a:.4f}")

    samples = list(zip(test_df[TEXT_COL].tolist()[:15], golds[:15], preds[:15]))
    results_file = args.results_file or os.path.join(args.output_dir, f"results_{TASK}.txt")
    write_results_report(results_file, args, train_result, trainer, acc, per_label, samples)

    try:
        import wandb
        if wandb.run is not None:
            wandb.log({"test/accuracy": acc,
                       **{f"test/acc_{lab}": a for lab, a in per_label.items()}})
            wandb.run.summary["test_accuracy"] = acc
    except Exception as e:
        print(f"wandb logging skipped: {e}")


if __name__ == "__main__":
    main()
