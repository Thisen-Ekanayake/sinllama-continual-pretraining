# train_lora_cpt.py — Documentation

> **Continual Pre-Training (CPT) Script for SinLlama using LoRA**  
> A two-stage fine-tuning pipeline that incrementally adapts a Llama-based causal language model to new text data, first warming up embeddings and then applying low-rank adaptation (LoRA) to transformer layers.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Architecture and Stage Design](#architecture-and-stage-design)
4. [Configuration Reference](#configuration-reference)
5. [Detailed Walkthrough](#detailed-walkthrough)
   - [Tokenizer Setup](#tokenizer-setup)
   - [Model Loading](#model-loading)
   - [Stage 1: Embedding Warmup](#stage-1-embedding-warmup)
   - [Stage 2: LoRA + Embeddings](#stage-2-lora--embeddings)
   - [Dataset Pipeline](#dataset-pipeline)
   - [Tokenization and Packing](#tokenization-and-packing)
   - [Perplexity Callback](#perplexity-callback)
   - [Training Arguments](#training-arguments)
   - [Model Saving and Merging](#model-saving-and-merging)
6. [Usage](#usage)
7. [Output Structure](#output-structure)
8. [Monitoring with Weights & Biases](#monitoring-with-weights--biases)
9. [Design Rationale](#design-rationale)
10. [Limitations and Considerations](#limitations-and-considerations)

---

## Overview

`train_lora_cpt.py` implements a **two-stage continual pre-training** workflow for a causal language model (specifically SinLlama, a Llama-based model). The goal is to inject new domain vocabulary and knowledge while keeping most of the original model weights frozen or minimally altered, preserving previously learned capabilities.

The two stages are:
- **Stage 1**: Freeze all weights except the token embedding matrix (`embed_tokens`) and the language model head (`lm_head`), allowing the model to learn new token representations.
- **Stage 2**: Apply LoRA adapters to all major projection layers while also keeping embeddings trainable, enabling richer knowledge integration at a fraction of the full fine-tuning cost.

---

## Requirements

The script depends on the following Python packages:

| Package | Purpose |
|---|---|
| `torch` | Deep learning framework (CUDA required) |
| `transformers` | Model loading, tokenizer, training utilities |
| `peft` | LoRA adapter support (`LoraConfig`, `get_peft_model`) |
| `datasets` | Efficient dataset loading and mapping |
| `wandb` | Experiment tracking and logging |

A CUDA-capable GPU is required. The script uses `bfloat16` precision and `tf32` matrix operations, so an Ampere or newer GPU (e.g., A100, 3090, 4090) is recommended.

---

## Architecture and Stage Design

### Why Two Stages?

When adapting a pre-trained LLM to a new language or domain corpus, a common failure mode is **catastrophic forgetting** — aggressively updating all weights destroys the original model's capabilities.

The two-stage approach mitigates this:

1. **Stage 1 (Embedding Warmup)** warms up only the input/output embedding layers. New tokens or vocabulary can be introduced, and the model learns to map them to useful vector representations before any structural weight updates occur. This is low-risk and computationally cheap.

2. **Stage 2 (LoRA Fine-Tuning)** uses Low-Rank Adaptation to inject trainable rank-decomposition matrices into the attention and feed-forward layers. Only a small number of new parameters are added, while the original weights are preserved. The embeddings remain trainable in this stage to continue refining token representations.

### LoRA Configuration

LoRA is applied to all seven major projection modules:

```
q_proj, k_proj, v_proj, o_proj       ← Attention
gate_proj, up_proj, down_proj         ← Feed-Forward (SwiGLU)
```

The `modules_to_save` field ensures that `embed_tokens` and `lm_head` are saved as full copies (not adapters) alongside the LoRA weights, preserving any embedding updates from Stage 1 or Stage 2.

---

## Configuration Reference

All configuration is managed through environment variables, with sensible defaults.

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/workspace/model/SinLlama_merged_bf16` | Path to the base model |
| `TXT_PATH` | `/workspace/data/All-Text_*.normalized.txt` | Path to the training text file |
| `OUT_DIR` | `/workspace/sinllama_cpt_out` | Root output directory |
| `STAGE` | `1` | Training stage (`1` or `2`) |
| `SEQ_LEN` | `1024` | Token sequence length for packed batches |
| `MICRO_BS` | `4` | Per-device micro batch size |
| `GRAD_ACC` | `4` | Gradient accumulation steps |
| `DATA_PERC` | `0.2` | Fraction of dataset to use (0.0–1.0) |
| `EPOCHS` | `0.5` (Stage 1) / `1.0` (Stage 2) | Training epochs |
| `LR` | `1e-4` (Stage 1) / `5e-6` (Stage 2) | Learning rate |
| `LORA_R` | `128` | LoRA rank (Stage 2 only) |

**Effective batch size** = `MICRO_BS × GRAD_ACC × number_of_GPUs`  
With defaults: `4 × 4 × 1 = 16` sequences of 1024 tokens = 16,384 tokens per step.

### Overriding via Shell

```bash
export STAGE=2
export LR=3e-6
export LORA_R=64
export DATA_PERC=0.5
python train_lora_cpt.py
```

---

## Detailed Walkthrough

### Tokenizer Setup

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

The fast Rust-based tokenizer is loaded for speed. Since the script uses **causal language modeling** (not masked or padded batches), the `pad_token` is set to `eos_token` as a safe default — it won't affect training because the data collator generates causal labels, and padding is not used in packed sequences.

---

### Model Loading

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
)
```

- **`torch_dtype=bfloat16`**: Halves memory usage vs. fp32 while maintaining stability (wider dynamic range than fp16).
- **`device_map="cuda"`**: Places the entire model on the GPU.
- **`attn_implementation="sdpa"`**: Uses PyTorch's Scaled Dot-Product Attention (Flash Attention-like, fused kernel) for efficient attention computation.
- **`use_cache=False`**: Disables the KV-cache (not needed during training, and incompatible with gradient checkpointing).

The script then checks if the tokenizer vocabulary size matches the model's embedding matrix. If they differ (e.g., new tokens were added), the embedding matrix is resized accordingly:

```python
if len(tokenizer) != embed_size:
    model.resize_token_embeddings(len(tokenizer))
```

---

### Stage 1: Embedding Warmup

All model parameters are frozen, then only `embed_tokens` and `lm_head` are re-enabled for gradient computation:

```python
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "embed_tokens" in name or "lm_head" in name:
        param.requires_grad = True
```

This is a targeted, low-risk update that teaches the model to handle new vocabulary or language statistics without disturbing the transformer's internal representations.

**Trainable parameters in Stage 1**: typically ~0.5–1% of total parameters (two weight matrices of size `vocab_size × hidden_dim`).

---

### Stage 2: LoRA + Embeddings

In Stage 2, a PEFT `LoraConfig` is applied:

```python
lora_cfg = LoraConfig(
    r=LORA_R,          # Rank of the low-rank matrices
    lora_alpha=LORA_R, # Scaling factor (alpha == r → scale = 1.0)
    lora_dropout=0.0,  # No dropout (common for CPT)
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[...],
    modules_to_save=["embed_tokens", "lm_head"],
)
model = get_peft_model(model, lora_cfg)
```

The `lora_alpha = lora_r` convention results in a scaling factor of `alpha/r = 1.0`, meaning LoRA outputs are not additionally rescaled. This is a common choice when using large ranks like 128.

**Trainable parameters in Stage 2** (with r=128): roughly 5–15% of total parameters depending on model size, covering all attention and FFN projection layers plus full embedding matrices.

---

### Dataset Pipeline

The dataset is a plain text file where each line is treated as a document:

```python
dataset = load_dataset("text", data_files={"train": TXT_PATH})["train"]
dataset = dataset.shuffle(seed=42)
```

A subset is then selected based on `DATA_PERC`, and a 98/2 train/validation split is created. The 2% validation split is used for `eval_loss` tracking and best-model selection.

---

### Tokenization and Packing

**Tokenization** converts each text line to token IDs and appends an `eos_token` as a document boundary marker:

```python
def tokenize_fn(example):
    return {
        "input_ids": tokenizer(
            text + tokenizer.eos_token,
            add_special_tokens=False,
        ).input_ids
    }
```

**Packing** concatenates all token IDs from a batch of lines into one long sequence and then slices it into fixed-length chunks of `SEQ_LEN` tokens:

```python
def pack_tokens(examples):
    all_ids = []
    for ids in examples["input_ids"]:
        all_ids.extend(ids)
    total = (len(all_ids) // SEQ_LEN) * SEQ_LEN
    # Split into SEQ_LEN-length chunks
```

This **eliminates padding waste** and ensures every token in every batch is a real training signal. The trade-off is that some document boundaries may be invisible to the model mid-sequence, but the `eos_token` delimiter between documents partially mitigates this.

Mapping is done with `num_proc=8` for parallel CPU processing.

---

### Perplexity Callback

A custom `TrainerCallback` computes and logs perplexity alongside loss:

```python
class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            logs["train_perplexity"] = math.exp(min(logs["loss"], 20))
        if "eval_loss" in logs:
            logs["eval_perplexity"] = math.exp(min(logs["eval_loss"], 20))
```

The `min(..., 20)` cap prevents numerical overflow from `math.exp()` during early training when losses can be large. This is logged to W&B automatically.

---

### Training Arguments

Key training configuration choices:

| Argument | Value | Rationale |
|---|---|---|
| `bf16=True` | Stage 1 & 2 | Memory efficient, stable |
| `tf32=True` | Stage 1 & 2 | ~2× matmul speedup on Ampere GPUs |
| `gradient_checkpointing=True` | Stage 1 & 2 | Reduces activation memory at slight compute cost |
| `use_reentrant=False` | Stage 1 & 2 | Required for compatibility with PEFT/LoRA |
| `warmup_ratio=0.05` | Both | 5% of steps used for LR warmup |
| `lr_scheduler_type` | `linear` (S1) / `cosine` (S2) | Cosine decay smoother for longer LoRA training |
| `optim="adamw_torch_fused"` | Both | Fused AdamW kernel for faster optimizer step |
| `load_best_model_at_end=True` | Both | Saves best checkpoint by `eval_loss` |
| `save_total_limit=2` | Both | Keeps only 2 checkpoints to save disk |

---

### Model Saving and Merging

**Stage 1** saves the full model (with updated embeddings) and tokenizer:

```python
model.save_pretrained(stage_out)
tokenizer.save_pretrained(stage_out)
```

**Stage 2** performs two saves:

1. **Adapter-only save** — saves only the LoRA weights and modified embedding matrices (compact, reusable):

```python
model.save_pretrained(adapter_path)
```

2. **Merged save** — merges LoRA weights into the base model and saves a standalone full-precision model (suitable for inference or further fine-tuning without PEFT):

```python
merged = model.merge_and_unload()
merged.save_pretrained(merged_path, safe_serialization=True)
```

`safe_serialization=True` saves in `.safetensors` format, which is safer and faster to load than `.bin` pickle files.

---

## Usage

### Step 1: Run Stage 1 (Embedding Warmup)

```bash
STAGE=1 \
MODEL_PATH=/path/to/base/model \
TXT_PATH=/path/to/corpus.txt \
OUT_DIR=/path/to/output \
DATA_PERC=0.2 \
EPOCHS=0.5 \
python train_lora_cpt.py
```

### Step 2: Run Stage 2 (LoRA Fine-Tuning)

Point `MODEL_PATH` to the Stage 1 output to build on the warmed-up embeddings:

```bash
STAGE=2 \
MODEL_PATH=/path/to/output/stage1 \
TXT_PATH=/path/to/corpus.txt \
OUT_DIR=/path/to/output \
DATA_PERC=0.2 \
EPOCHS=1.0 \
LORA_R=128 \
python train_lora_cpt.py
```

### Tip: Full Data Run

```bash
DATA_PERC=1.0 EPOCHS=1.0 STAGE=2 python train_lora_cpt.py
```

---

## Output Structure

```
OUT_DIR/
├── stage1/
│   ├── config.json
│   ├── tokenizer.json
│   ├── model.safetensors          ← Full model with updated embeddings
│   └── checkpoint-XXXX/           ← Intermediate checkpoints
│
└── stage2/
    ├── adapters/
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors  ← LoRA weights only
    │
    └── merged_bf16/
        ├── config.json
        ├── tokenizer.json
        └── model.safetensors          ← Full merged model (inference-ready)
```

The `merged_bf16` directory is the final artifact for deployment or further training.

---

## Monitoring with Weights & Biases

The script initializes a W&B run with:

```python
wandb.init(
    project="sinllama-cpt",
    name=f"stage{STAGE}_lr{LR}_data{DATA_PERC}",
)
```

The following metrics are tracked:

| Metric | Description |
|---|---|
| `loss` | Training cross-entropy loss |
| `train_perplexity` | `exp(train_loss)` |
| `eval_loss` | Validation cross-entropy loss |
| `eval_perplexity` | `exp(eval_loss)` |
| `learning_rate` | Current LR from scheduler |
| `grad_norm` | Gradient norm (clipped at 1.0) |

Runs are named by stage, learning rate, and data fraction for easy comparison across experiments.

---

## Design Rationale

### Why LoRA rank 128?
A rank of 128 is on the high end of typical LoRA configurations (common values are 8–64). This is intentional for CPT: higher rank gives the adapter more expressive capacity to absorb new language patterns. The trade-off is more trainable parameters and memory, but LoRA is still far cheaper than full fine-tuning.

### Why `lora_alpha = lora_r`?
Setting `alpha = r` means the effective scaling factor `alpha/r = 1.0`. This avoids any additional rescaling of the LoRA output and makes the learning rate's effect more predictable. Some practitioners prefer `alpha = 2r` for faster adaptation.

### Why pack sequences instead of padding?
Padding wastes compute on tokens that don't contribute to the loss. Packing maximizes GPU utilization and training signal density, which is especially important for CPT where efficiency matters.

### Why `lora_dropout=0.0`?
Dropout in LoRA is typically used to prevent overfitting in task-specific fine-tuning on small datasets. For continual pre-training on large corpora, overfitting is less of a concern, so dropout is disabled to preserve training signal.

---

## Limitations and Considerations

- **Single-GPU design**: `device_map="cuda"` maps the entire model to one GPU. For multi-GPU setups, replace with `device_map="auto"` or use `accelerate` / `torchrun`.
- **No evaluation metric beyond loss**: The script tracks only cross-entropy loss and perplexity. Downstream task evaluation must be done separately.
- **Document boundary leakage**: Token packing concatenates documents without masking cross-document attention. For strict isolation, a custom collator with document-level attention masks would be needed.
- **Stage 2 requires Stage 1 output**: The design assumes Stage 1 is run first. If using the original base model directly for Stage 2, embedding warmup is skipped, which may be suboptimal for vocabularies with new tokens.
- **Fixed `DATA_PERC` subset**: The subsample is taken as the first N rows after shuffling, which is deterministic but does not stratify by domain or document length.