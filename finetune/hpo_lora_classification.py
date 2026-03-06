"""
train_lora_classification.py
MI300X-optimised LoRA fine-tuning + Ray Tune ASHA hyperparameter search
Logs per-epoch eval metrics (loss, accuracy, F1) to JSON.

Fixes applied:
  - tf32=False          (tf32 is NVIDIA Ampere only, crashes on ROCm)
  - modules_to_save     removed (caused OOM via tied weight duplication)
  - resources_per_trial {"gpu": 0.5} → 2 concurrent trials (~45GB each)
  - micro_bs            capped at 64 (removes 128 from search space)
  - device_map          {"": torch.cuda.current_device()} for Ray isolation
"""

import os
import json
import time
import datetime
import numpy as np
import torch
import wandb

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, accuracy_score

# Ray Tune
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune import CLIReporter


# ============================================================
# CONFIG  (override via env vars)
# ============================================================

MODEL_PATH  = os.environ.get("MODEL_PATH",  "/workspace/model/SinLlama_CPT")
DATA_DIR    = os.environ.get("DATA_DIR",    "/workspace/data/classification")
OUT_DIR     = os.environ.get("OUT_DIR",     "/workspace/classification_out")
LOG_DIR     = os.environ.get("LOG_DIR",     os.path.join(OUT_DIR, "logs"))

TASK        = os.environ.get("TASK",   "sentiment")   # sentiment | writing | news
STAGE       = int(os.environ.get("STAGE",   "2"))

SEQ_LEN     = int(os.environ.get("SEQ_LEN", "512"))

# HPO mode: set HPO=1 to run Ray Tune search, HPO=0 for single run
HPO_MODE    = bool(int(os.environ.get("HPO", "0")))
HPO_TRIALS  = int(os.environ.get("HPO_TRIALS", "20"))

# Single-run defaults (ignored when HPO=1)
MICRO_BS     = int(os.environ.get("MICRO_BS",     "64"))
GRAD_ACC     = int(os.environ.get("GRAD_ACC",     "1"))
EPOCHS       = float(os.environ.get("EPOCHS",     "3"))
LR           = float(os.environ.get("LR",         "1e-5"))
LORA_R       = int(os.environ.get("LORA_R",       "16"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.1"))

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sinllama-classification")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# TASK METADATA
# ============================================================

TASK_META = {
    "sentiment": {
        "prompt_fn": lambda text: (
            f"Does the following Sinhala sentence have a POSITIVE, NEGATIVE or NEUTRAL sentiment?\n\n"
            f"{text}\n\nAnswer:"
        ),
        "labels": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
    },
    "writing": {
        "prompt_fn": lambda text: (
            f"Classify the Sinhala comment into ACADEMIC, CREATIVE, NEWS, BLOG.\n\n"
            f"Comment: {text}\nAnswer:"
        ),
        "labels": ["ACADEMIC", "CREATIVE", "NEWS", "BLOG"],
    },
    "news": {
        "prompt_fn": lambda text: (
            f"Classify into:\nPolitical: 0, Business: 1, Technology: 2, Sports: 3, Entertainment: 4.\n\n"
            f"Comment: {text}\nAnswer:"
        ),
        "labels": ["0", "1", "2", "3", "4"],
    },
}


# ============================================================
# JSON EPOCH LOGGER  (callback)
# ============================================================

class EpochJSONLogger(TrainerCallback):
    """
    Writes one JSON record per epoch to  LOG_DIR/<run_name>_epochs.jsonl
    Final summary goes to LOG_DIR/<run_name>_summary.json
    """

    def __init__(self, run_name: str, log_dir: str):
        self.run_name   = run_name
        self.log_dir    = log_dir
        self.epoch_file = os.path.join(log_dir, f"{run_name}_epochs.jsonl")
        self.history    = []
        self._train_loss_accum = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._train_loss_accum.append(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        epoch_record = {
            "epoch":          round(state.epoch, 2),
            "step":           state.global_step,
            "timestamp":      datetime.datetime.utcnow().isoformat(),
            "train_loss_avg": round(float(np.mean(self._train_loss_accum)), 6)
                              if self._train_loss_accum else None,
            **{k: round(float(v), 6) if isinstance(v, float) else v
               for k, v in metrics.items()},
        }

        self.history.append(epoch_record)
        self._train_loss_accum = []

        with open(self.epoch_file, "a") as f:
            f.write(json.dumps(epoch_record) + "\n")

        print(f"\n[EpochLogger] epoch {epoch_record['epoch']} → {epoch_record}")

    def on_train_end(self, args, state, control, **kwargs):
        if not self.history:
            return

        best = max(self.history, key=lambda r: r.get("eval_f1", 0))
        summary = {
            "run_name":   self.run_name,
            "task":       TASK,
            "stage":      STAGE,
            "epochs_run": len(self.history),
            "best_epoch": best,
            "all_epochs": self.history,
        }
        summary_path = os.path.join(self.log_dir, f"{self.run_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[EpochLogger] Summary saved → {summary_path}")


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ============================================================
# DATASET
# ============================================================

def build_dataset(tokenizer, seq_len: int):
    meta = TASK_META[TASK]
    prompt_fn = meta["prompt_fn"]

    raw = load_dataset(
        "json",
        data_files={
            "train":      os.path.join(DATA_DIR, "train.jsonl"),
            "validation": os.path.join(DATA_DIR, "val.jsonl"),
        }
    )

    def build_example(example):
        text   = example["text"]
        label  = str(example["label"])
        prompt = prompt_fn(text)
        full   = prompt + " " + label + tokenizer.eos_token
        return {"full_text": full, "prompt": prompt}

    raw = raw.map(build_example)

    def tokenize_fn(example):
        tokens = tokenizer(
            example["full_text"],
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )

        # Mask prompt tokens — loss only on label token(s)
        prompt_ids = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=seq_len,
            add_special_tokens=False,
        )["input_ids"]
        prompt_len = len(prompt_ids)

        labels = tokens["input_ids"].copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        tokens["labels"] = labels
        return tokens

    tokenized = raw.map(
        tokenize_fn,
        batched=False,
        remove_columns=["full_text", "prompt", "text", "label"],
    )
    tokenized.set_format(type="torch")
    return tokenized["train"], tokenized["validation"]


# ============================================================
# COMPUTE METRICS
# ============================================================

def make_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred   # (N, seq, vocab), (N, seq)

        pred_ids = []
        true_ids = []

        for i in range(logits.shape[0]):
            label_row = labels[i]
            valid_pos = np.where(label_row != -100)[0]
            if len(valid_pos) == 0:
                continue
            last_pos = valid_pos[-1]

            pred_tok = int(np.argmax(logits[i, last_pos - 1]))
            true_tok = int(label_row[last_pos])

            pred_ids.append(pred_tok)
            true_ids.append(true_tok)

        if not pred_ids:
            return {"eval_f1": 0.0, "eval_accuracy": 0.0}

        f1  = f1_score(true_ids, pred_ids, average="macro", zero_division=0)
        acc = accuracy_score(true_ids, pred_ids)

        return {
            "eval_f1":       round(float(f1),  4),
            "eval_accuracy": round(float(acc), 4),
        }

    return compute_metrics


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(lora_r: int, lora_dropout: float):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        # FIX: safe for Ray worker GPU isolation
        device_map={"": torch.cuda.current_device()},
        attn_implementation="eager",   # ROCm-safe; swap to "sdpa" if supported
    )
    model.config.use_cache = False

    if STAGE == 1:
        print("\n>>> STAGE 1: embed_tokens + lm_head ONLY")
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True

    else:
        print(f"\n>>> STAGE 2: LoRA r={lora_r}, dropout={lora_dropout}")
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            # FIX: modules_to_save removed — model has tie_word_embeddings=True
            # which causes PEFT to duplicate tied weights per trial → OOM on HPO
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model


# ============================================================
# CORE TRAIN FUNCTION
# ============================================================

def train(config: dict, run_name: str, report_to: str = "wandb"):
    lr           = config["lr"]
    lora_r       = config["lora_r"]
    lora_dropout = config["lora_dropout"]
    weight_decay = config["weight_decay"]
    micro_bs     = config["micro_bs"]
    grad_acc     = config["grad_acc"]
    epochs       = config["epochs"]

    tokenizer        = load_tokenizer()
    train_ds, val_ds = build_dataset(tokenizer, SEQ_LEN)
    model            = build_model(lora_r, lora_dropout)

    stage_out = os.path.join(OUT_DIR, run_name)
    os.makedirs(stage_out, exist_ok=True)

    with open(os.path.join(stage_out, "run_config.json"), "w") as f:
        json.dump({"run_name": run_name, "task": TASK, "stage": STAGE, **config}, f, indent=2)

    training_args = TrainingArguments(
        output_dir=stage_out,
        run_name=run_name,

        # ── dtype ──────────────────────────────────────────────────
        bf16=True,
        tf32=False,       # FIX: tf32 is NVIDIA Ampere only, not ROCm

        # ── batch / accumulation ───────────────────────────────────
        per_device_train_batch_size=micro_bs,
        per_device_eval_batch_size=micro_bs,
        gradient_accumulation_steps=grad_acc,

        # ── MI300X: 192 GB → NO gradient checkpointing needed ──────
        gradient_checkpointing=False,

        # ── optimiser ──────────────────────────────────────────────
        optim="adamw_torch",   # not fused — ROCm compatible
        learning_rate=lr,
        weight_decay=weight_decay,
        max_grad_norm=1.0,

        # ── schedule ───────────────────────────────────────────────
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        num_train_epochs=epochs,

        # ── eval / save ────────────────────────────────────────────
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,

        # ── reporting ──────────────────────────────────────────────
        report_to=report_to,
    )

    epoch_logger = EpochJSONLogger(run_name=run_name, log_dir=LOG_DIR)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[
            epoch_logger,
            EarlyStoppingCallback(early_stopping_patience=2),
        ],
    )

    trainer.train()

    # ── save ───────────────────────────────────────────────────────
    if STAGE == 1:
        model.save_pretrained(stage_out)
        tokenizer.save_pretrained(stage_out)
    else:
        adapter_path = os.path.join(stage_out, "adapters")
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        print("Merging LoRA weights...")
        merged      = model.merge_and_unload()
        merged_path = os.path.join(stage_out, "merged_bf16")
        merged.save_pretrained(merged_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_path)

    best_metric = max(
        (r.get("eval_f1", 0) for r in epoch_logger.history),
        default=0.0,
    )
    return best_metric


# ============================================================
# RAY TUNE WRAPPER
# ============================================================

def ray_train_fn(ray_config):
    ts = int(time.time())
    run_name = (
        f"{TASK}_s{STAGE}"
        f"_lr{ray_config['lr']:.0e}"
        f"_r{ray_config['lora_r']}"
        f"_bs{ray_config['micro_bs']}"
        f"_{ts}"
    )
    best_f1 = train(ray_config, run_name=run_name, report_to="none")
    tune.report({"eval_f1": best_f1})


# ============================================================
# HPO  (Hyperband + Bayesian proposals via Optuna)
# ============================================================

def run_hpo():
    print(f"\n{'='*60}")
    print(f"HPO MODE  |  TASK={TASK}  STAGE={STAGE}  TRIALS={HPO_TRIALS}")
    print(f"{'='*60}\n")

    search_space = {
        "lr":           tune.loguniform(5e-6, 5e-4),
        "lora_r":       tune.choice([8, 16, 32, 64]),
        "lora_dropout": tune.choice([0.0, 0.05, 0.1]),
        "weight_decay": tune.choice([0.01, 0.05, 0.1]),
        # FIX: max batch size 64 — safe for 2 concurrent trials on 192 GB
        "micro_bs":     tune.choice([16, 32, 64]),
        "grad_acc":     tune.choice([1, 2]),
        "epochs":       tune.choice([2, 3, 5]),
    }

    scheduler = ASHAScheduler(
        metric="eval_f1",
        mode="max",
        max_t=5,
        grace_period=1,
        reduction_factor=2,
    )

    search_algo = OptunaSearch(metric="eval_f1", mode="max")

    reporter = CLIReporter(
        metric_columns=["eval_f1", "training_iteration"],
        max_progress_rows=10,
    )

    analysis = tune.run(
        ray_train_fn,
        config=search_space,
        num_samples=HPO_TRIALS,
        scheduler=scheduler,
        search_alg=search_algo,
        progress_reporter=reporter,
        # FIX: 0.5 GPU per trial → 2 concurrent trials (~45 GB each)
        resources_per_trial={"gpu": 0.5, "cpu": 4},
        storage_path=os.path.join(OUT_DIR, "ray_results"),
        name=f"hpo_{TASK}_stage{STAGE}",
        verbose=1,
    )

    best_cfg   = analysis.get_best_config(metric="eval_f1", mode="max")
    best_trial = analysis.get_best_trial(metric="eval_f1", mode="max")

    hpo_summary = {
        "task":         TASK,
        "stage":        STAGE,
        "num_trials":   HPO_TRIALS,
        "best_config":  best_cfg,
        "best_eval_f1": best_trial.last_result["eval_f1"],
        "timestamp":    datetime.datetime.utcnow().isoformat(),
    }

    hpo_path = os.path.join(LOG_DIR, f"hpo_{TASK}_stage{STAGE}_results.json")
    with open(hpo_path, "w") as f:
        json.dump(hpo_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"HPO COMPLETE")
    print(f"Best config  : {best_cfg}")
    print(f"Best eval_f1 : {best_trial.last_result['eval_f1']:.4f}")
    print(f"Results saved: {hpo_path}")
    print(f"{'='*60}\n")

    return best_cfg


# ============================================================
# SINGLE RUN
# ============================================================

def run_single():
    config = {
        "lr":           LR,
        "lora_r":       LORA_R,
        "lora_dropout": LORA_DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "micro_bs":     MICRO_BS,
        "grad_acc":     GRAD_ACC,
        "epochs":       EPOCHS,
    }

    ts       = int(time.time())
    run_name = f"{TASK}_s{STAGE}_lr{LR:.0e}_r{LORA_R}_{ts}"

    print(f"\n{'='*60}")
    print(f"SINGLE RUN  |  TASK={TASK}  STAGE={STAGE}")
    print(f"Config: {config}")
    print(f"Run name: {run_name}")
    print(f"{'='*60}\n")

    wandb.init(project=WANDB_PROJECT, name=run_name)
    best_f1 = train(config, run_name=run_name, report_to="wandb")
    print(f"\nBest eval F1: {best_f1:.4f}")
    wandb.finish()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if HPO_MODE:
        best_config = run_hpo()

        retrain = bool(int(os.environ.get("RETRAIN_BEST", "1")))
        if retrain:
            print("\nRe-training with best config at full budget...")
            best_config["epochs"] = float(os.environ.get("RETRAIN_EPOCHS", "5"))
            ts       = int(time.time())
            run_name = f"{TASK}_s{STAGE}_best_{ts}"
            wandb.init(project=WANDB_PROJECT, name=run_name)
            best_f1 = train(best_config, run_name=run_name, report_to="wandb")
            print(f"Final best model F1: {best_f1:.4f}")
            wandb.finish()
    else:
        run_single()