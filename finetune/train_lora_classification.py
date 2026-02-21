import os
import torch
import wandb

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ============================================================
# CONFIG (ENV-STYLE)
# ============================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/model/SinLlama_CPT")
DATA_DIR   = os.environ.get("DATA_DIR", "/workspace/data/classification")
OUT_DIR    = os.environ.get("OUT_DIR",  "/workspace/classification_out")

TASK       = os.environ.get("TASK", "sentiment")  # sentiment | writing | news
STAGE      = int(os.environ.get("STAGE", "2"))

SEQ_LEN    = int(os.environ.get("SEQ_LEN", "512"))
MICRO_BS   = int(os.environ.get("MICRO_BS", "8"))
GRAD_ACC   = int(os.environ.get("GRAD_ACC", "2"))

EPOCHS     = float(os.environ.get("EPOCHS", "3"))
LR         = float(os.environ.get("LR", "1e-5"))

LORA_R     = int(os.environ.get("LORA_R", "64"))
LORA_ALPHA = LORA_R

os.makedirs(OUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"TASK={TASK} | STAGE={STAGE}")
print(f"LR={LR} | EPOCHS={EPOCHS}")
print(f"SEQ={SEQ_LEN} | BS={MICRO_BS}x{GRAD_ACC}")
print(f"{'='*60}\n")

# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# MODEL
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
)

model.config.use_cache = False

# ============================================================
# STAGE CONTROL
# ============================================================

if STAGE == 1:
    print("\n>>> STAGE 1: embed_tokens + lm_head ONLY")

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = True

else:
    print(f"\n>>> STAGE 2: LoRA r={LORA_R}")

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
# DATASET
# ============================================================

dataset = load_dataset(
    "json",
    data_files={
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "validation": os.path.join(DATA_DIR, "val.jsonl"),
    }
)

def build_prompt(example):
    text  = example["text"]
    label = str(example["label"])

    if TASK == "writing":
        prompt = f"""Classify the Sinhala comment into ACADEMIC, CREATIVE, NEWS, BLOG.

Comment: {text}
Answer:"""

    elif TASK == "sentiment":
        prompt = f"""Does the following Sinhala sentence have a POSITIVE, NEGATIVE or NEUTRAL sentiment?

{text}

Answer:"""

    elif TASK == "news":
        prompt = f"""Classify into:
Political: 0, Business: 1, Technology: 2, Sports: 3, Entertainment: 4.

Comment: {text}
Answer:"""

    full_text = prompt + " " + label + tokenizer.eos_token
    return {"text": full_text}

dataset = dataset.map(build_prompt)

# ============================================================
# TOKENIZATION (FIXED VERSION)
# ============================================================

def tokenize_fn(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=SEQ_LEN,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text", "label"]  # <-- THIS FIXES YOUR ERROR
)

dataset.set_format(type="torch")

train_ds = dataset["train"]
val_ds   = dataset["validation"]

# ============================================================
# TRAINING ARGS
# ============================================================

stage_out = os.path.join(OUT_DIR, f"{TASK}_stage{STAGE}")

training_args = TrainingArguments(
    output_dir=stage_out,
    run_name=f"{TASK}_stage{STAGE}_lr{LR}",
    bf16=True,
    tf32=True,
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
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_torch_fused",
    weight_decay=0.01,
    max_grad_norm=1.0,
    report_to="wandb",
)

# ============================================================
# WANDB
# ============================================================

wandb.init(
    project="sinllama-classification",
    name=f"{TASK}_stage{STAGE}_lr{LR}",
)

# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()

# ============================================================
# SAVE
# ============================================================

print(f"\nSaving to {stage_out}...")

if STAGE == 1:
    model.save_pretrained(stage_out)
    tokenizer.save_pretrained(stage_out)
else:
    adapter_path = os.path.join(stage_out, "adapters")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print("Merging LoRA...")
    merged = model.merge_and_unload()
    merged_path = os.path.join(stage_out, "merged_bf16")
    merged.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)

wandb.finish()