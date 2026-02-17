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

MODEL_PATH  = os.environ.get("MODEL_PATH", "/workspace/model/SinLlama_merged_bf16")
TXT_PATH    = os.environ.get("TXT_PATH",   "/workspace/data/All-Text_8696658_147190824.normalized.txt")
OUT_DIR     = os.environ.get("OUT_DIR",    "/workspace/sinllama_cpt_out")
STAGE       = int(os.environ.get("STAGE",  "1"))

SEQ_LEN     = int(os.environ.get("SEQ_LEN",   "1024"))
MICRO_BS    = int(os.environ.get("MICRO_BS",  "4"))
GRAD_ACC    = int(os.environ.get("GRAD_ACC",  "4"))

# Stage 1: embed warmup — higher LR, short run
# Stage 2: LoRA+embeds  — lower LR, full run
EPOCHS      = float(os.environ.get("EPOCHS", "0.5" if STAGE == 1 else "1.0"))
LR          = float(os.environ.get("LR",     "1e-4" if STAGE == 1 else "2e-5"))

LORA_R      = int(os.environ.get("LORA_R",   "128"))
LORA_ALPHA  = LORA_R   # always keep alpha == r for CPT

os.makedirs(OUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"  STAGE={STAGE} | LR={LR} | EPOCHS={EPOCHS} | SEQ={SEQ_LEN} | BS={MICRO_BS}x{GRAD_ACC}")
print(f"{'='*60}\n")

# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Vocab size: {len(tokenizer)}")

# ============================================================
# Model
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
)
model.config.use_cache = False

# Resize if tokenizer vocab doesn't match model embeddings
embed_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) != embed_size:
    print(f"Resizing embeddings: {embed_size} -> {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ============================================================
# Stage 1: Freeze everything except embed_tokens + lm_head
# ============================================================

if STAGE == 1:
    print("\n>>> STAGE 1: Training embed_tokens + lm_head ONLY")

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = True
            print(f"  Trainable: {name} | {param.shape}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)\n")

# ============================================================
# Stage 2: LoRA on attention+MLP + embed_tokens + lm_head
# ============================================================

else:
    print(f"\n>>> STAGE 2: LoRA r={LORA_R} + embed_tokens + lm_head")

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,          # 0 dropout recommended for CPT
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=["embed_tokens", "lm_head"],  # critical for extended vocab
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

# ============================================================
# Dataset — line-by-line with token packing
# ============================================================

print(f"\nLoading dataset...")
dataset = load_dataset(
    "text",
    data_files={"train": TXT_PATH},
)["train"]

dataset = dataset.shuffle(seed=42)
split   = dataset.train_test_split(test_size=0.02, seed=42)
train_ds = split["train"]
val_ds   = split["test"]

print(f"Train lines: {len(train_ds):,} | Val lines: {len(val_ds):,}")

def tokenize_fn(example):
    text = example["text"].strip()
    if not text:
        return {"input_ids": []}
    return {"input_ids": tokenizer(
        text + tokenizer.eos_token,
        add_special_tokens=False,
    ).input_ids}

def pack_tokens(examples):
    all_ids = []
    for ids in examples["input_ids"]:
        all_ids.extend(ids)
    total = (len(all_ids) // SEQ_LEN) * SEQ_LEN
    all_ids = all_ids[:total]
    return {
        "input_ids": [all_ids[i:i+SEQ_LEN] for i in range(0, total, SEQ_LEN)]
    }

print("Tokenizing...")
train_tok = train_ds.map(tokenize_fn, remove_columns=["text"], num_proc=8)
val_tok   = val_ds.map(tokenize_fn,   remove_columns=["text"], num_proc=8)

print("Packing...")
train_packed = train_tok.map(pack_tokens, batched=True, batch_size=2000, remove_columns=["input_ids"])
val_packed   = val_tok.map(pack_tokens,   batched=True, batch_size=2000, remove_columns=["input_ids"])

train_packed.set_format(type="torch")
val_packed.set_format(type="torch")

print(f"Packed train sequences: {len(train_packed):,}")
print(f"Packed val sequences:   {len(val_packed):,}")

# ============================================================
# Perplexity logging callback
# ============================================================

class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                logs["train_perplexity"] = math.exp(min(logs["loss"], 20))
            if "eval_loss" in logs:
                logs["eval_perplexity"] = math.exp(min(logs["eval_loss"], 20))

# ============================================================
# Training arguments
# ============================================================

stage_out = os.path.join(OUT_DIR, f"stage{STAGE}")

training_args = TrainingArguments(
    output_dir=stage_out,
    bf16=True,
    tf32=True,
    per_device_train_batch_size=MICRO_BS,
    per_device_eval_batch_size=MICRO_BS,
    gradient_accumulation_steps=GRAD_ACC,
    gradient_checkpointing=True,             # saves ~30% VRAM
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=LR,
    warmup_ratio=0.05,
    lr_scheduler_type="linear" if STAGE == 1 else "cosine",
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_steps=1000,
    eval_strategy="steps",
    eval_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_torch_fused",
    weight_decay=0.01,
    max_grad_norm=1.0,
    dataloader_num_workers=4,
    report_to="wandb",
)

# ============================================================
# Trainer
# ============================================================

wandb.init(
    project="sinllama-cpt",
    name=f"stage{STAGE}_r{LORA_R}_lr{LR}_ep{EPOCHS}_seq{SEQ_LEN}",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
# Save
# ============================================================

print(f"\nSaving to {stage_out}...")

if STAGE == 1:
    save_path = os.path.join(stage_out, "model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n>>> Stage 1 complete. Now run Stage 2 with:")
    print(f"    MODEL_PATH={save_path} STAGE=2 python /workspace/train_cpt.py")

else:
    # Save LoRA adapters
    adapter_path = os.path.join(stage_out, "adapters")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # Merge and save full model
    print("Merging LoRA weights into base model...")
    merged = model.merge_and_unload()
    merged_path = os.path.join(stage_out, "merged_bf16")
    merged.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    print(f"\n>>> Stage 2 complete. Merged model at: {merged_path}")

wandb.finish()