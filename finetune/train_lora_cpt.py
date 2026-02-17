import os
import math
import wandb
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

# ============================================================
# CONFIG — edit these if needed, or pass as env vars
# ============================================================

MODEL_PATH     = os.environ.get("MODEL_PATH",  "/workspace/model/SinLlama_merged_bf16")
TOKENIZER_PATH = os.environ.get("TOK_PATH",    "/workspace/model/SinLlama_merged_bf16")
TXT_PATH       = os.environ.get("TXT_PATH",    "/workspace/data/All-Text_8696658_147190824.normalized.txt")
OUT_DIR        = os.environ.get("OUT_DIR",     "/workspace/sinllama_cpt_out")
STAGE          = int(os.environ.get("STAGE",   "1"))
VOCAB_SIZE     = int(os.environ.get("VOCAB_SIZE", "139336"))

SEQ_LEN        = int(os.environ.get("SEQ_LEN",    "512"))
MICRO_BS       = int(os.environ.get("MICRO_BS",   "4"))
GRAD_ACC       = int(os.environ.get("GRAD_ACC",   "4"))
EPOCHS         = float(os.environ.get("EPOCHS",   "0.5" if STAGE == 1 else "3.0"))
LR             = float(os.environ.get("LR",       "1e-4" if STAGE == 1 else "2e-5"))
LORA_R         = int(os.environ.get("LORA_R",     "128"))

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Load model
# ============================================================

print(f"\n{'='*50}")
print(f"  STAGE {STAGE} | LR={LR} | EPOCHS={EPOCHS} | SEQ={SEQ_LEN}")
print(f"{'='*50}\n")

model, _ = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=SEQ_LEN,
    dtype=torch.bfloat16,
    load_in_4bit=False,       # A40 48GB — no need for quantization
)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)

# Resize only if vocab doesn't already match
if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
    print(f"Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} -> {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))
else:
    print(f"Vocab size already matches: {len(tokenizer)}")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.use_cache = False

# ============================================================
# Stage 1: Embeddings only
# ============================================================

if STAGE == 1:
    print(">>> STAGE 1: Training embed_tokens + lm_head only")

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ============================================================
# Stage 2: LoRA + Embeddings
# ============================================================

else:
    print(f">>> STAGE 2: LoRA r={LORA_R} + embed_tokens + lm_head")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_R,       # alpha == r, scaling = 1.0
        lora_dropout=0.0,         # 0 dropout recommended for CPT
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=["embed_tokens", "lm_head"],  # critical for extended vocab
    )
    model.print_trainable_parameters()

# ============================================================
# Dataset
# ============================================================

print(f"\nLoading dataset from {TXT_PATH}")
dataset = load_dataset("text", data_files={"train": TXT_PATH})["train"]
dataset = dataset.shuffle(seed=42)
split   = dataset.train_test_split(test_size=0.02, seed=42)

def format_text(example):
    text = example["text"].strip()
    if not text:
        return {"text": ""}
    return {"text": text + tokenizer.eos_token}

train_ds = split["train"].map(format_text, num_proc=4)
val_ds   = split["test"].map(format_text,  num_proc=4)

print(f"Train: {len(train_ds):,} lines | Val: {len(val_ds):,} lines")

# ============================================================
# WandB
# ============================================================

wandb.init(
    project="sinllama-cpt",
    name=f"stage{STAGE}_r{LORA_R}_lr{LR}_ep{EPOCHS}_seq{SEQ_LEN}",
)

# ============================================================
# Trainer
# ============================================================

stage_out = os.path.join(OUT_DIR, f"stage{STAGE}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    dataset_text_field="text",
    max_seq_length=SEQ_LEN,
    dataset_num_proc=4,
    packing=True,
    args=SFTConfig(
        output_dir=stage_out,
        bf16=True,
        per_device_train_batch_size=MICRO_BS,
        per_device_eval_batch_size=MICRO_BS,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        warmup_ratio=0.05,
        lr_scheduler_type="linear" if STAGE == 1 else "cosine",
        num_train_epochs=EPOCHS,
        logging_steps=50,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="wandb",
    ),
)

# ============================================================
# Train
# ============================================================

trainer.train()

# ============================================================
# Save
# ============================================================

print(f"\nSaving to {stage_out}")

if STAGE == 1:
    # Save full model with warmed-up embeddings — used as base for stage 2
    model.save_pretrained(os.path.join(stage_out, "model"))
    tokenizer.save_pretrained(os.path.join(stage_out, "model"))
    print(f"\n>>> Stage 1 complete.")
    print(f">>> Run Stage 2 with:")
    print(f">>> MODEL_PATH={os.path.join(stage_out, 'model')} STAGE=2 python /workspace/train_cpt.py")

else:
    # Save LoRA adapters
    model.save_pretrained(os.path.join(stage_out, "adapters"))
    tokenizer.save_pretrained(os.path.join(stage_out, "adapters"))

    # Save merged full model
    model.save_pretrained_merged(
        os.path.join(stage_out, "merged_bf16"),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"\n>>> Stage 2 complete. Merged model at: {stage_out}/merged_bf16")

wandb.finish()