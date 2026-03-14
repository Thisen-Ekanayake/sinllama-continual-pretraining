"""
SinLlama LoRA Fine-tuning — News Classification (5-class)
==========================================================

Architecture / methods ported from sentiment_context_aware_finetune.py:
  • Sliding-window chunking  — long articles are split into overlapping
    token chunks; each chunk is formatted as a separate training example
    (the label is repeated for every chunk so the model learns to classify
    from any window of the article).
  • compute_metrics          — accuracy, macro-precision/recall/F1,
    weighted-F1 computed at every eval epoch.
  • Per-class metrics CSV    — precision / recall / F1 / support per
    label saved after training.
  • Predictions CSV          — full test-set predictions saved with
    confidence scores.
  • Rich W&B config logging  — hyperparams + label distribution histograms.

Labels (from dataset):
  Political: 0 | Business: 1 | Technology: 2 | Sports: 3 | Entertainment: 4

Dataset expected at:
  data_new/news/{train,val,test}.jsonl
  Each line: {"text": "...", "label": <int>, "label_name": "..."}
"""

import os
import json
import math
import random
import traceback
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import wandb

from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ============================================================
# CONFIG  (all overridable via environment variables)
# ============================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/model/SinLlama_CPT")
DATA_DIR   = os.environ.get("DATA_DIR",   "data_new/news")
OUT_DIR    = os.environ.get("OUT_DIR",    "outputs/news_lora")

SEQ_LEN    = int(os.environ.get("SEQ_LEN",   "512"))
MICRO_BS   = int(os.environ.get("MICRO_BS",  "4"))
GRAD_ACC   = int(os.environ.get("GRAD_ACC",  "4"))   # effective BS = 16

EPOCHS     = float(os.environ.get("EPOCHS",  "3"))
LR         = float(os.environ.get("LR",      "1e-5"))

LORA_R     = int(os.environ.get("LORA_R",    "64"))
LORA_ALPHA = LORA_R

# ---- Sliding-window (ported from sentiment_context_aware_finetune.py) ----
# If an article is shorter than CHUNK_SIZE tokens it becomes one chunk.
# Longer articles are split with CHUNK_STRIDE overlap so every part
# of the text generates a training example.
CHUNK_SIZE   = int(os.environ.get("CHUNK_SIZE",   "400"))  # tokens per chunk
CHUNK_STRIDE = int(os.environ.get("CHUNK_STRIDE", "200"))  # 50 % overlap
MAX_CHUNKS   = int(os.environ.get("MAX_CHUNKS",   "8"))    # cap per article

RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
USE_WANDB   = os.environ.get("USE_WANDB", "true").lower() != "false"

# ---- Label map (fixed for this task) ----
# NOTE: label names must each tokenise to a single distinct token in the
# model vocabulary.  "entertainment" splits into multiple tokens in the
# LLaMA tokeniser so we use "entertain" as the prompt/answer keyword and
# map it back to the canonical name for display / CSV outputs.
LABEL_NAMES        = ["political", "business", "technology", "sports", "entertainment"]
LABEL_PROMPT_WORDS = ["political", "business", "technology", "sports", "entertain"]
NUM_LABELS         = len(LABEL_NAMES)
ID_TO_LABEL        = {i: n      for i, n in enumerate(LABEL_NAMES)}
ID_TO_PROMPT_WORD  = {i: w      for i, w in enumerate(LABEL_PROMPT_WORDS)}
PROMPT_WORD_TO_ID  = {w: i      for i, w in enumerate(LABEL_PROMPT_WORDS)}

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# SEEDS
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("=" * 70)
print("SinLlama — NEWS CLASSIFICATION  (LoRA, sliding-window)")
print("=" * 70)
print(f"MODEL   : {MODEL_PATH}")
print(f"DATA    : {DATA_DIR}")
print(f"OUT     : {OUT_DIR}")
print(f"SEQ_LEN : {SEQ_LEN}   CHUNK {CHUNK_SIZE}/{CHUNK_STRIDE} (max {MAX_CHUNKS})")
print(f"LR      : {LR}   EPOCHS {EPOCHS}   BS {MICRO_BS}×{GRAD_ACC}={MICRO_BS*GRAD_ACC}")
print(f"LoRA    : r={LORA_R}  alpha={LORA_ALPHA}")
print("=" * 70)

# ============================================================
# ENVIRONMENT CHECK
# ============================================================

print(f"\nPyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM    : {props.total_memory / 1e9:.1f} GB")

# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"✓ Tokenizer  vocab={tokenizer.vocab_size}  pad='{tokenizer.pad_token}'")

# ---- Verify every prompt word is a single token ----
# This must run after the tokenizer is loaded but before training starts.
print("\nVerifying label prompt words tokenise to single tokens:")
for lid, word in ID_TO_PROMPT_WORD.items():
    toks = tokenizer(word, add_special_tokens=False)["input_ids"]
    decoded = tokenizer.decode(toks)
    status = "✓" if len(toks) == 1 else "✗ MULTI-TOKEN — change this word!"
    print(f"  [{lid}] {LABEL_NAMES[lid]:14s} → prompt word '{word}' → {toks} ('{decoded}') {status}")
    if len(toks) != 1:
        raise ValueError(
            f"Label prompt word '{word}' tokenises to {len(toks)} tokens {toks}. "
            f"Replace it with a single-token word in LABEL_PROMPT_WORDS."
        )

# ============================================================
# MODEL  (LoRA only — Stage 2)
# ============================================================

print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
)
model.config.use_cache = False

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    modules_to_save=["embed_tokens", "lm_head"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ============================================================
# SLIDING-WINDOW CHUNKING
# (ported from sentiment_context_aware_finetune.py → tokenize_chunks)
#
# Instead of encoding [CLS] vectors (BERT style), here each chunk is
# turned into a full causal-LM prompt+label string.  This lets the
# Llama model learn to predict the correct label from any window of
# the article text.
# ============================================================

def sliding_window_token_ids(
    token_ids: List[int],
    chunk_size: int,
    stride: int,
    max_chunks: int,
) -> List[List[int]]:
    """
    Split a list of token IDs into overlapping chunks.
    Returns up to max_chunks lists, each of length ≤ chunk_size.
    """
    chunks = []
    start  = 0
    while start < len(token_ids) and len(chunks) < max_chunks:
        end = min(start + chunk_size, len(token_ids))
        chunks.append(token_ids[start:end])
        if end == len(token_ids):
            break
        start += stride
    return chunks if chunks else [token_ids[:chunk_size]]


def build_prompt(text: str, label_id: int) -> str:
    """Full prompt+answer string used for causal-LM training."""
    word = ID_TO_PROMPT_WORD[label_id]
    return (
        f"Classify the following Sinhala news article into one of: "
        f"political, business, technology, sports, entertain.\n\n"
        f"Article: {text}\n"
        f"Category: {word}{tokenizer.eos_token}"
    )


def build_prompt_no_label(text: str) -> str:
    """Prompt without answer — used to compute the label-only loss mask."""
    return (
        f"Classify the following Sinhala news article into one of: "
        f"political, business, technology, sports, entertain.\n\n"
        f"Article: {text}\n"
        f"Category:"
    )


# ============================================================
# LOAD JSONL DATA
# ============================================================

def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    assert "text"       in df.columns, "Missing 'text' column"
    assert "label"      in df.columns, "Missing 'label' column"
    assert "label_name" in df.columns, "Missing 'label_name' column"
    df["label_name"] = df["label_name"].str.lower().str.strip()
    return df.reset_index(drop=True)


print("\nLoading data...")
train_df = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
val_df   = load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))
test_df  = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
print(f"✓  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

# Label distribution (ported from sentiment_context_aware_finetune.py)
print("\nLabel distribution:")
for i, name in ID_TO_LABEL.items():
    tr = (train_df["label"] == i).sum()
    va = (val_df["label"]   == i).sum()
    te = (test_df["label"]  == i).sum()
    print(f"  [{i}] {name:14s}  train={tr:5d}  val={va:4d}  test={te:4d}")

# ============================================================
# TOKENISE WITH SLIDING WINDOW
# ============================================================

def tokenise_df(df: pd.DataFrame, split_name: str) -> HFDataset:
    """
    For each row, tokenise the article text, apply sliding-window chunking,
    and build one training example per chunk.  The label answer token is
    masked out of the loss for the prompt prefix (only the answer token
    contributes to the loss).
    """
    all_input_ids      = []
    all_attention_mask = []
    all_labels         = []
    all_label_ids      = []   # integer class ids kept for compute_metrics

    for _, row in df.iterrows():
        text       = str(row["text"])
        label_id   = int(row["label"])

        # Tokenise just the article body to decide chunking boundaries.
        # We leave room for the prompt wrapper + answer.
        prompt_prefix = build_prompt_no_label("")   # overhead estimate
        overhead      = len(tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]) + 8
        body_budget   = CHUNK_SIZE - overhead        # tokens available for article body

        body_ids = tokenizer(
            text, add_special_tokens=False, truncation=False
        )["input_ids"]

        # Sliding window over body token IDs
        body_chunks = sliding_window_token_ids(
            body_ids,
            chunk_size=max(body_budget, 32),
            stride=max(body_budget // 2, 16),
            max_chunks=MAX_CHUNKS,
        )

        for chunk_ids in body_chunks:
            # Decode chunk back to text so we can build the prompt cleanly
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            full_text  = build_prompt(chunk_text, label_id)

            enc = tokenizer(
                full_text,
                truncation=True,
                max_length=SEQ_LEN,
                padding="max_length",
            )
            input_ids      = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            # Build labels: mask the prompt prefix with -100 so only the
            # answer token contributes to the loss
            # (ported idea: only supervise the output token(s))
            prefix_text = build_prompt_no_label(chunk_text)
            prefix_len  = len(
                tokenizer(
                    prefix_text,
                    truncation=True,
                    max_length=SEQ_LEN,
                    add_special_tokens=True,
                )["input_ids"]
            )

            labels = [-100] * prefix_len + input_ids[prefix_len:]
            # Mask padding
            labels = [
                tok if mask == 1 else -100
                for tok, mask in zip(labels, attention_mask)
            ]
            # Pad labels to SEQ_LEN
            labels = labels + [-100] * (SEQ_LEN - len(labels))
            labels = labels[:SEQ_LEN]

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)
            all_label_ids.append(label_id)

    dataset = HFDataset.from_dict({
        "input_ids":      all_input_ids,
        "attention_mask": all_attention_mask,
        "labels":         all_labels,
        "label_id":       all_label_ids,
    })
    dataset.set_format(type="torch")

    orig  = len(df)
    expnd = len(dataset)
    print(f"  {split_name}: {orig:,} articles → {expnd:,} chunks "
          f"(×{expnd/orig:.1f} expansion)")
    return dataset


print("\nTokenising datasets with sliding window...")
train_ds = tokenise_df(train_df, "train")
val_ds   = tokenise_df(val_df,   "val")
test_ds  = tokenise_df(test_df,  "test")

# ============================================================
# COMPUTE METRICS
# (ported directly from sentiment_context_aware_finetune.py)
#
# For a causal-LM model the predictions tensor is logits over the
# vocabulary at every token position.  We care only about the
# FIRST non-masked label position (i.e., the answer token) to
# compare predicted vs true label name token.
# ============================================================

# Pre-encode each prompt word to its first token id
LABEL_TOKEN_IDS = {
    label_id: tokenizer(
        word, add_special_tokens=False
    )["input_ids"][0]
    for label_id, word in ID_TO_PROMPT_WORD.items()
}
LABEL_TOKEN_ID_TO_LABEL = {v: k for k, v in LABEL_TOKEN_IDS.items()}
print(f"\nLabel → token mapping:")
for lid, tid in LABEL_TOKEN_IDS.items():
    print(f"  [{lid}] {LABEL_NAMES[lid]:14s} ('{ID_TO_PROMPT_WORD[lid]}') → token_id={tid}  "
          f"'{tokenizer.decode([tid])}'")


def compute_metrics(eval_pred: EvalPrediction):
    """
    Ported from sentiment_context_aware_finetune.py compute_metrics.
    Extracts the predicted label from the first unmasked answer position.
    """
    logits_all = eval_pred.predictions   # [N, SEQ_LEN, vocab]
    labels_all = eval_pred.label_ids     # [N, SEQ_LEN]

    y_pred = []
    y_true = []

    for logits, labels in zip(logits_all, labels_all):
        # Find the first position the model should predict (non -100)
        valid_positions = np.where(labels != -100)[0]
        if len(valid_positions) == 0:
            continue
        pos = valid_positions[0]

        # True label: decode label token id → class id
        true_tok = int(labels[pos])
        true_cls = LABEL_TOKEN_ID_TO_LABEL.get(true_tok, -1)
        if true_cls == -1:
            continue

        # Predicted: argmax over the label-token subset only
        label_tok_ids = list(LABEL_TOKEN_IDS.values())
        subset_logits = logits[pos, label_tok_ids]
        pred_idx      = int(np.argmax(subset_logits))
        pred_cls      = list(LABEL_TOKEN_IDS.keys())[pred_idx]

        y_true.append(true_cls)
        y_pred.append(pred_cls)

    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "f1_weighted": 0.0}

    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "recall":      recall_score(y_true, y_pred,    average="macro",    zero_division=0),
        "f1":          f1_score(y_true, y_pred,        average="macro",    zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred,        average="weighted", zero_division=0),
    }

# ============================================================
# WANDB
# (ported rich config logging from sentiment_context_aware_finetune.py)
# ============================================================

if USE_WANDB:
    run = wandb.init(
        project="sinllama-news-classification",
        name=f"news_lora_lr{LR}_r{LORA_R}_ep{int(EPOCHS)}",
        resume="allow",
        config={
            "model":          MODEL_PATH,
            "task":           "news_classification",
            "lora_r":         LORA_R,
            "lora_alpha":     LORA_ALPHA,
            "learning_rate":  LR,
            "epochs":         EPOCHS,
            "seq_len":        SEQ_LEN,
            "chunk_size":     CHUNK_SIZE,
            "chunk_stride":   CHUNK_STRIDE,
            "max_chunks":     MAX_CHUNKS,
            "micro_bs":       MICRO_BS,
            "grad_acc":       GRAD_ACC,
            "effective_bs":   MICRO_BS * GRAD_ACC,
            "num_labels":     NUM_LABELS,
            "label_names":    LABEL_NAMES,
            "train_articles": len(train_df),
            "train_chunks":   len(train_ds),
            "val_articles":   len(val_df),
            "val_chunks":     len(val_ds),
            "seed":           RANDOM_SEED,
        },
    )
    wandb.log({
        "train_label_dist": wandb.Histogram(train_df["label"].tolist()),
        "val_label_dist":   wandb.Histogram(val_df["label"].tolist()),
    })
    print(f"\n✓ W&B run : {run.get_url()}")

# ============================================================
# TRAINING ARGS
# ============================================================

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    run_name=f"news_lora_lr{LR}_r{LORA_R}",
    bf16=True,
    per_device_train_batch_size=MICRO_BS,
    per_device_eval_batch_size=MICRO_BS,
    gradient_accumulation_steps=GRAD_ACC,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=LR,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=EPOCHS,
    logging_steps=20,
    logging_first_step=True,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",          # macro-F1 (ported from BERT script)
    greater_is_better=True,
    optim="adamw_torch_fused",
    weight_decay=0.01,
    max_grad_norm=1.0,
    seed=RANDOM_SEED,
    report_to="wandb" if USE_WANDB else "none",
)

# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

approx_steps = int(len(train_ds) * EPOCHS / (MICRO_BS * GRAD_ACC))
print(f"\n✓ Trainer ready  |  ~{approx_steps:,} optimiser steps")

# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

try:
    train_result = trainer.train()
    print("\n✓ Training complete")
    for k, v in train_result.metrics.items():
        print(f"  {k}: {v}")
except KeyboardInterrupt:
    print("\nInterrupted — saving checkpoint...")
    model.save_pretrained(os.path.join(OUT_DIR, "interrupted"))
    tokenizer.save_pretrained(os.path.join(OUT_DIR, "interrupted"))
except Exception as e:
    print(f"\nTraining error: {e}")
    print(traceback.format_exc())

# ============================================================
# SAVE ADAPTERS + MERGED MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

adapter_path = os.path.join(OUT_DIR, "adapters")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"✓ Adapters saved to {adapter_path}")

print("Merging LoRA weights...")
merged      = model.merge_and_unload()
merged_path = os.path.join(OUT_DIR, "merged_bf16")
merged.save_pretrained(merged_path, safe_serialization=True)
tokenizer.save_pretrained(merged_path)
print(f"✓ Merged model saved to {merged_path}")

# ============================================================
# FULL TEST-SET EVALUATION
# (ported from sentiment_context_aware_finetune.py)
# ============================================================

print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

test_output = trainer.predict(test_ds)
logits_all  = test_output.predictions  # [N_chunks, SEQ_LEN, vocab]
labels_all  = test_output.label_ids

y_pred_chunks = []
y_true_chunks = []

for logits, labels in zip(logits_all, labels_all):
    valid_positions = np.where(labels != -100)[0]
    if len(valid_positions) == 0:
        continue
    pos = valid_positions[0]

    true_tok = int(labels[pos])
    true_cls = LABEL_TOKEN_ID_TO_LABEL.get(true_tok, -1)
    if true_cls == -1:
        continue

    label_tok_ids = list(LABEL_TOKEN_IDS.values())
    subset_logits  = logits[pos, label_tok_ids]
    pred_idx       = int(np.argmax(subset_logits))
    pred_cls       = list(LABEL_TOKEN_IDS.keys())[pred_idx]

    confidence = float(torch.softmax(
        torch.tensor(subset_logits, dtype=torch.float32), dim=0
    )[pred_idx].item())

    y_true_chunks.append(true_cls)
    y_pred_chunks.append(pred_cls)

y_true = np.array(y_true_chunks)
y_pred = np.array(y_pred_chunks)

test_metrics = {
    "accuracy":    accuracy_score(y_true, y_pred),
    "precision":   precision_score(y_true, y_pred, average="macro",    zero_division=0),
    "recall":      recall_score(y_true, y_pred,    average="macro",    zero_division=0),
    "f1":          f1_score(y_true, y_pred,        average="macro",    zero_division=0),
    "f1_weighted": f1_score(y_true, y_pred,        average="weighted", zero_division=0),
}

print("\nTest Metrics:")
for k, v in test_metrics.items():
    print(f"  {k:18s}: {v:.4f}")

if USE_WANDB:
    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

# Classification report
print("\n" + classification_report(
    y_true, y_pred,
    target_names=LABEL_NAMES,
    zero_division=0,
))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
cm_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix_test.csv"))
print(f"✓ Confusion matrix saved")

# ============================================================
# PER-CLASS METRICS CSV
# (ported from sentiment_context_aware_finetune.py)
# ============================================================

prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
    y_true, y_pred, average=None, zero_division=0, labels=list(range(NUM_LABELS))
)
per_class_df = pd.DataFrame({
    "Label ID":  range(NUM_LABELS),
    "Category":  LABEL_NAMES,
    "Precision": prec_pc,
    "Recall":    rec_pc,
    "F1-Score":  f1_pc,
    "Support":   sup_pc,
})
per_class_path = os.path.join(OUT_DIR, "per_class_metrics_test.csv")
per_class_df.to_csv(per_class_path, index=False)
print(f"\nPer-class metrics:")
print(per_class_df.to_string(index=False))
print(f"\n✓ Saved to {per_class_path}")

if USE_WANDB:
    wandb.log({"test/per_class_metrics": wandb.Table(dataframe=per_class_df)})
    for metric in ["Precision", "Recall", "F1-Score"]:
        data = [[LABEL_NAMES[i], per_class_df.loc[i, metric]] for i in range(NUM_LABELS)]
        tbl  = wandb.Table(data=data, columns=["Category", metric])
        wandb.log({
            f"test/per_class_{metric.lower().replace('-','_')}":
            wandb.plot.bar(tbl, "Category", metric, title=f"Per-Category {metric}")
        })

# ============================================================
# SAVE PREDICTIONS CSV
# (ported from sentiment_context_aware_finetune.py)
# ============================================================

# Re-run to get confidence per chunk
confidences   = []
pred_names    = []
true_names    = []
chunk_indices = []   # which original article index each chunk came from

# Reconstruct article index mapping
chunk_label_ids = test_ds["label_id"].tolist()

for logits, labels, true_lbl_id in zip(
    logits_all, labels_all, chunk_label_ids
):
    valid_positions = np.where(labels != -100)[0]
    if len(valid_positions) == 0:
        confidences.append(0.0)
        pred_names.append("unknown")
        true_names.append(ID_TO_LABEL.get(int(true_lbl_id), "unknown"))
        continue

    pos           = valid_positions[0]
    label_tok_ids = list(LABEL_TOKEN_IDS.values())
    subset_logits  = logits[pos, label_tok_ids]
    pred_idx       = int(np.argmax(subset_logits))
    pred_cls       = list(LABEL_TOKEN_IDS.keys())[pred_idx]
    conf           = float(torch.softmax(
        torch.tensor(subset_logits, dtype=torch.float32), dim=0
    )[pred_idx].item())

    confidences.append(conf)
    pred_names.append(ID_TO_LABEL[pred_cls])
    true_names.append(ID_TO_LABEL.get(int(true_lbl_id), "unknown"))

pred_path = os.path.join(OUT_DIR, "predictions_test.csv")
pd.DataFrame({
    "true_label_name":      true_names,
    "predicted_label_name": pred_names,
    "correct":              [t == p for t, p in zip(true_names, pred_names)],
    "confidence":           confidences,
}).to_csv(pred_path, index=False)
print(f"\n✓ Predictions saved to {pred_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"Outputs in: {OUT_DIR}/")
print(f"  adapters/                 — LoRA adapter weights")
print(f"  merged_bf16/              — merged full model")
print(f"  per_class_metrics_test.csv")
print(f"  predictions_test.csv")
print(f"  confusion_matrix_test.csv")

if USE_WANDB:
    print(f"\nW&B run: {wandb.run.get_url()}")
    wandb.finish()