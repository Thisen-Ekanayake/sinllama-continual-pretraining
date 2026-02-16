import os
import math
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/models/SinLlama_merged_bf16")
TXT_PATH   = os.environ.get("TXT_PATH", "/workspace/data/All-Text_8696658_147190824.normalized.txt")
OUT_DIR    = os.environ.get("OUT_DIR", "/workspace/sinllama_lora_cpt")

SEQ_LEN    = int(os.environ.get("SEQ_LEN", 1024))
MICRO_BS   = int(os.environ.get("MICRO_BS", 8))
GRAD_ACC   = int(os.environ.get("GRAD_ACC", 2))

LR         = float(os.environ.get("LR", 1.5e-4))
WARMUP     = float(os.environ.get("WARMUP_RATIO", 0.02))
EPOCHS     = float(os.environ.get("EPOCHS", 1.0))

LORA_R     = int(os.environ.get("LORA_R", 32))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 64))
LORA_DROP  = float(os.environ.get("LORA_DROPOUT", 0.05))

LOG_STEPS   = int(os.environ.get("LOG_STEPS", 50))
SAVE_STEPS  = int(os.environ.get("SAVE_STEPS", 1000))
EVAL_STEPS  = 1000   # 🔥 requested validation frequency

PROJECT_NAME = os.environ.get("WANDB_PROJECT", "sinllama-cpt")

torch.backends.cuda.matmul.allow_tf32 = True

# ============================================================
# WandB Init
# ============================================================

wandb.init(
    project=PROJECT_NAME,
    name=f"lora_cpt_seq{SEQ_LEN}_bs{MICRO_BS}x{GRAD_ACC}_lr{LR}",
)

# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# Load & Split Dataset
# ============================================================

dataset = load_dataset(
    "text",
    data_files={"train": TXT_PATH},
)["train"]

dataset = dataset.shuffle(seed=42)

split = dataset.train_test_split(test_size=0.02)  # 2% validation
train_ds = split["train"]
val_ds   = split["test"]

# ============================================================
# Tokenization
# ============================================================

def tokenize_fn(example):
    text = example["text"].strip()
    if not text:
        return {"input_ids": []}
    ids = tokenizer(text + "\n", add_special_tokens=False).input_ids
    return {"input_ids": ids}

train_tok = train_ds.map(tokenize_fn, remove_columns=["text"], num_proc=4)
val_tok   = val_ds.map(tokenize_fn, remove_columns=["text"], num_proc=4)

# ============================================================
# Token Packing
# ============================================================

def pack_tokens(examples):
    all_ids = []
    for ids in examples["input_ids"]:
        all_ids.extend(ids)

    total_len = (len(all_ids) // SEQ_LEN) * SEQ_LEN
    all_ids = all_ids[:total_len]

    return {
        "input_ids": [
            all_ids[i:i+SEQ_LEN]
            for i in range(0, total_len, SEQ_LEN)
        ]
    }

train_packed = train_tok.map(
    pack_tokens,
    batched=True,
    batch_size=1000,
    remove_columns=["input_ids"],
)

val_packed = val_tok.map(
    pack_tokens,
    batched=True,
    batch_size=1000,
    remove_columns=["input_ids"],
)

train_packed.set_format(type="torch")
val_packed.set_format(type="torch")

# ============================================================
# Model Load
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
)

model.config.use_cache = False

# ============================================================
# Attach LoRA
# ============================================================

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROP,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ============================================================
# Perplexity Callback
# ============================================================

class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                logs["train_perplexity"] = math.exp(logs["loss"])
            if "eval_loss" in logs:
                logs["eval_perplexity"] = math.exp(logs["eval_loss"])

# ============================================================
# TrainingArguments
# ============================================================

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    bf16=True,
    tf32=True,
    per_device_train_batch_size=MICRO_BS,
    per_device_eval_batch_size=MICRO_BS,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    warmup_ratio=WARMUP,
    lr_scheduler_type="cosine",
    num_train_epochs=EPOCHS,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",
    run_name=wandb.run.name,
    optim="adamw_torch_fused",
    weight_decay=0.0,
    max_grad_norm=1.0,
    dataloader_num_workers=4,
)

# ============================================================
# Trainer
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_packed,
    eval_dataset=val_packed,
    data_collator=data_collator,
    callbacks=[PerplexityCallback()],
)

# ============================================================
# Train
# ============================================================

trainer.train()

# ============================================================
# Save LoRA adapters
# ============================================================

model.save_pretrained(os.path.join(OUT_DIR, "adapters"))
tokenizer.save_pretrained(os.path.join(OUT_DIR, "tokenizer"))

wandb.finish()

print("Training complete. Adapters saved.")