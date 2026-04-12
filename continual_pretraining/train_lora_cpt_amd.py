"""
Continual Pre-Training script for AMD MI300I (192 GB VRAM).
Runs Stage 1 (embed warmup) then Stage 2 (LoRA + embeds) in one invocation.

    python train_lora_cpt_amd.py

Environment overrides (all optional):
    MODEL_PATH      Original merged model directory
    TXT_PATH        Sinhala corpus .txt file
    OUT_DIR         Root output directory
    STAGE1_DIR      Override Stage 1 output location  (default: OUT_DIR/stage1)
    SEQ_LEN         Token sequence length             (default: 1024)
    MICRO_BS        Per-device batch size             (default: 8)
    GRAD_ACC        Gradient accumulation steps       (default: 2)
    DATA_PERC       Fraction of corpus to use         (default: 1.0)
    S1_EPOCHS       Stage 1 epochs                   (default: 0.5)
    S2_EPOCHS       Stage 2 epochs                   (default: 1.0)
    S1_LR           Stage 1 learning rate             (default: 1e-4)
    S2_LR           Stage 2 learning rate             (default: 5e-5)
    LORA_R          LoRA rank                         (default: 128)
    EMBED_LR_SCALE  embed/lm_head LR multiplier in S2 (default: 10.0)
    ATTN_IMPL       Attention backend                 (default: eager)
    SKIP_STAGE1     Set to "1" to skip Stage 1        (default: 0)
    SKIP_STAGE2     Set to "1" to skip Stage 2        (default: 0)
"""

import gc
import math
import os

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH     = os.environ.get("MODEL_PATH",  "/workspace/model/SinLlama_merged_bf16")
TXT_PATH       = os.environ.get("TXT_PATH",    "/workspace/data/All-Text_8696658_147190824.normalized.txt")
OUT_DIR        = os.environ.get("OUT_DIR",     "/workspace/sinllama_cpt_out")
STAGE1_DIR     = os.environ.get("STAGE1_DIR",  os.path.join(OUT_DIR, "stage1"))
STAGE2_DIR     = os.path.join(OUT_DIR, "stage2")

SEQ_LEN        = int(os.environ.get("SEQ_LEN",   "1024"))
MICRO_BS       = int(os.environ.get("MICRO_BS",  "8"))
GRAD_ACC       = int(os.environ.get("GRAD_ACC",  "2"))
DATA_PERC      = float(os.environ.get("DATA_PERC", "1.0"))

S1_EPOCHS      = float(os.environ.get("S1_EPOCHS", "0.5"))
S2_EPOCHS      = float(os.environ.get("S2_EPOCHS", "1.0"))
S1_LR          = float(os.environ.get("S1_LR",     "1e-4"))
S2_LR          = float(os.environ.get("S2_LR",     "5e-5"))

LORA_R         = int(os.environ.get("LORA_R",         "128"))
EMBED_LR_SCALE = float(os.environ.get("EMBED_LR_SCALE", "10.0"))

# "eager" is safest on ROCm.
# Set ATTN_IMPL=flash_attention_2 if aotriton / ROCm FA is installed.
ATTN_IMPL      = os.environ.get("ATTN_IMPL", "eager")

SKIP_STAGE1    = os.environ.get("SKIP_STAGE1", "0") == "1"
SKIP_STAGE2    = os.environ.get("SKIP_STAGE2", "0") == "1"

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Perplexity callback (shared)
# ============================================================

class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                logs["train_perplexity"] = math.exp(min(logs["loss"], 20))
            if "eval_loss" in logs:
                logs["eval_perplexity"] = math.exp(min(logs["eval_loss"], 20))


# ============================================================
# Custom Trainer — per-group LR + no weight decay on embeddings
# ============================================================

class CPTTrainer(Trainer):
    """
    Parameter groups:
      - LoRA / attention weights  →  base_lr, weight_decay
      - Biases / LayerNorm        →  base_lr, no weight_decay
      - embed_tokens / lm_head    →  base_lr × embed_lr_scale, no weight_decay

    adamw_torch is used instead of the fused variant because fused kernels
    require CUDA and are not available on ROCm.
    """

    def __init__(self, *args, embed_lr_scale: float = 1.0, **kwargs):
        self.embed_lr_scale = embed_lr_scale
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        base_lr = self.args.learning_rate
        wd      = self.args.weight_decay

        embed_params   = []
        lora_wd_params = []
        lora_no_wd     = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "embed_tokens" in name or "lm_head" in name:
                embed_params.append(param)
            elif any(nd in name for nd in ["bias", "layer_norm", "layernorm"]):
                lora_no_wd.append(param)
            else:
                lora_wd_params.append(param)

        param_groups = [
            {"params": lora_wd_params, "lr": base_lr,                        "weight_decay": wd},
            {"params": lora_no_wd,     "lr": base_lr,                        "weight_decay": 0.0},
            {"params": embed_params,   "lr": base_lr * self.embed_lr_scale,  "weight_decay": 0.0},
        ]
        param_groups = [g for g in param_groups if g["params"]]

        self.optimizer = torch.optim.AdamW(param_groups)
        return self.optimizer


# ============================================================
# Dataset (built once, reused by both stages)
# ============================================================

def build_datasets(tokenizer):
    print("\nLoading dataset...")
    dataset = load_dataset("text", data_files={"train": TXT_PATH})["train"]
    dataset = dataset.shuffle(seed=42)

    if DATA_PERC < 1.0:
        subset_size = int(len(dataset) * DATA_PERC)
        dataset = dataset.select(range(subset_size))
        print(f"Using {subset_size:,} lines ({DATA_PERC*100:.1f}%)")
    else:
        print(f"Using full dataset ({len(dataset):,} lines)")

    split    = dataset.train_test_split(test_size=0.02, seed=42)
    train_ds = split["train"]
    val_ds   = split["test"]
    print(f"Train lines: {len(train_ds):,} | Val lines: {len(val_ds):,}")

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    def tokenize_fn(example):
        text = example["text"].strip()
        if not text:
            return {"input_ids": []}
        ids = tokenizer(text, add_special_tokens=False).input_ids
        return {"input_ids": ids + [eos_id]}  # EOS = document boundary

    def pack_tokens(examples):
        all_ids = []
        for ids in examples["input_ids"]:
            all_ids.extend(ids)
        # Each chunk: BOS + (SEQ_LEN - 1) content tokens
        step   = SEQ_LEN - 1
        chunks = [
            [bos_id] + all_ids[i : i + step]
            for i in range(0, len(all_ids) - step + 1, step)
        ]
        return {"input_ids": chunks}

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

    return train_packed, val_packed


# ============================================================
# Stage 1 — embed_tokens + lm_head warmup
# ============================================================

def run_stage1(tokenizer, train_packed, val_packed):
    print(f"\n{'='*60}")
    print(f"STAGE 1 — embed_tokens + lm_head warmup")
    print(f"LR={S1_LR} | EPOCHS={S1_EPOCHS} | SEQ={SEQ_LEN} | BS={MICRO_BS}×{GRAD_ACC}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=ATTN_IMPL,
    )
    model.config.use_cache = False

    embed_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embed_size:
        print(f"Resizing embeddings: {embed_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Freeze everything, then unfreeze embeddings
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = True
            print(f"  Trainable: {name}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    training_args = TrainingArguments(
        output_dir=STAGE1_DIR,
        bf16=True,
        per_device_train_batch_size=MICRO_BS,
        per_device_eval_batch_size=MICRO_BS,
        gradient_accumulation_steps=GRAD_ACC,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=S1_LR,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        num_train_epochs=S1_EPOCHS,
        logging_steps=50,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=False,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    run = wandb.init(
        project="sinllama-cpt",
        name=f"stage1_lr{S1_LR}_data{DATA_PERC}",
        reinit=True,
    )

    trainer = CPTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_packed,
        eval_dataset=val_packed,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[PerplexityCallback()],
        embed_lr_scale=1.0,  # no scale needed in Stage 1 (embeds are the only trainable params)
    )

    trainer.train()

    print(f"\nSaving Stage 1 checkpoint to {STAGE1_DIR}...")
    model.save_pretrained(STAGE1_DIR)
    tokenizer.save_pretrained(STAGE1_DIR)

    run.finish()

    # Free VRAM before Stage 2
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"VRAM after Stage 1 cleanup: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# ============================================================
# Stage 2 — LoRA + embed_tokens + lm_head
# ============================================================

def run_stage2(tokenizer, train_packed, val_packed):
    print(f"\n{'='*60}")
    print(f"STAGE 2 — LoRA r={LORA_R} + embed_tokens + lm_head")
    print(f"LR={S2_LR} | embed_LR={S2_LR*EMBED_LR_SCALE:.2e} | EPOCHS={S2_EPOCHS}")
    print(f"SEQ={SEQ_LEN} | BS={MICRO_BS}×{GRAD_ACC}")
    print(f"Loading from: {STAGE1_DIR}")
    print(f"{'='*60}")

    if not os.path.isdir(STAGE1_DIR):
        raise FileNotFoundError(
            f"Stage 1 output not found at {STAGE1_DIR}.\n"
            "Either let Stage 1 run first, or set STAGE1_DIR to an existing "
            "embed-warmed checkpoint and set SKIP_STAGE1=1."
        )

    model = AutoModelForCausalLM.from_pretrained(
        STAGE1_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=ATTN_IMPL,
    )
    model.config.use_cache = False

    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_R,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=STAGE2_DIR,
        bf16=True,
        per_device_train_batch_size=MICRO_BS,
        per_device_eval_batch_size=MICRO_BS,
        gradient_accumulation_steps=GRAD_ACC,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=S2_LR,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        num_train_epochs=S2_EPOCHS,
        logging_steps=50,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        # Disabled: PEFT modules_to_save + load_best_model_at_end can corrupt
        # the embed_tokens wrapper state on checkpoint reload.
        load_best_model_at_end=False,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    run = wandb.init(
        project="sinllama-cpt",
        name=f"stage2_lr{S2_LR}_data{DATA_PERC}",
        reinit=True,
    )

    trainer = CPTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_packed,
        eval_dataset=val_packed,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[PerplexityCallback()],
        embed_lr_scale=EMBED_LR_SCALE,
    )

    trainer.train()

    # Save adapters
    adapter_path = os.path.join(STAGE2_DIR, "adapters")
    print(f"\nSaving adapters to {adapter_path}...")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # Merge and save full model
    print("Merging LoRA into base weights...")
    merged = model.merge_and_unload()
    merged_path = os.path.join(STAGE2_DIR, "merged_bf16")
    merged.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    print(f"Merged model saved to {merged_path}")

    run.finish()


# ============================================================
# Main
# ============================================================

def main():
    print(f"\n{'='*60}")
    print(f"SinLlama Continual Pre-Training — AMD MI300I")
    print(f"Model:  {MODEL_PATH}")
    print(f"Corpus: {TXT_PATH}")
    print(f"Output: {OUT_DIR}")
    print(f"ATTN:   {ATTN_IMPL} | SEQ={SEQ_LEN} | BS={MICRO_BS}×{GRAD_ACC}")
    print(f"{'='*60}\n")

    # Load tokenizer once; both stages share it
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Vocab size: {len(tokenizer)}")

    # Build packed datasets once; both stages share them
    train_packed, val_packed = build_datasets(tokenizer)

    if not SKIP_STAGE1:
        run_stage1(tokenizer, train_packed, val_packed)
    else:
        print("\nSkipping Stage 1 (SKIP_STAGE1=1)")

    if not SKIP_STAGE2:
        run_stage2(tokenizer, train_packed, val_packed)
    else:
        print("\nSkipping Stage 2 (SKIP_STAGE2=1)")

    print("\nAll stages complete.")
    print(f"Final merged model: {os.path.join(STAGE2_DIR, 'merged_bf16')}")


if __name__ == "__main__":
    main()
