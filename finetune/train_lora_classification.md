# train_lora_classification.py — Documentation

> **Sinhala Text Classification Fine-Tuning Script using LoRA**  
> A two-stage fine-tuning pipeline that adapts a continually pre-trained Llama-based causal language model (SinLlama) to perform generative text classification across multiple Sinhala NLP tasks.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Supported Tasks](#supported-tasks)
4. [Architecture and Stage Design](#architecture-and-stage-design)
5. [Configuration Reference](#configuration-reference)
6. [Detailed Walkthrough](#detailed-walkthrough)
   - [Tokenizer Setup](#tokenizer-setup)
   - [Model Loading](#model-loading)
   - [Stage 1: Embedding Warmup](#stage-1-embedding-warmup)
   - [Stage 2: LoRA Fine-Tuning](#stage-2-lora-fine-tuning)
   - [Dataset Loading](#dataset-loading)
   - [Prompt Construction](#prompt-construction)
   - [Tokenization](#tokenization)
   - [Training Arguments](#training-arguments)
   - [Model Saving and Merging](#model-saving-and-merging)
7. [Data Format](#data-format)
8. [Usage](#usage)
9. [Output Structure](#output-structure)
10. [Monitoring with Weights & Biases](#monitoring-with-weights--biases)
11. [Relationship to train_lora_cpt.py](#relationship-to-train_lora_cptpy)
12. [Design Rationale](#design-rationale)
13. [Limitations and Considerations](#limitations-and-considerations)

---

## Overview

`train_lora_classification.py` implements **generative text classification** fine-tuning for SinLlama — a Sinhala-adapted Llama-based causal language model. Rather than attaching a classification head, the script frames each classification task as a text generation problem: the model is trained to generate the correct label token(s) given a structured natural-language prompt.

The script supports three Sinhala NLP tasks out of the box:
- **Sentiment Analysis** (positive / negative / neutral)
- **Writing Style Classification** (academic / creative / news / blog)
- **News Category Classification** (political / business / technology / sports / entertainment)

Like its companion script `train_lora_cpt.py`, training is organized into two optional stages: an embedding warmup (Stage 1) and a LoRA adapter stage (Stage 2).

---

## Requirements

| Package | Purpose |
|---|---|
| `torch` | Deep learning framework (CUDA required) |
| `transformers` | Model loading, tokenizer, Trainer API |
| `peft` | LoRA adapter support (`LoraConfig`, `get_peft_model`) |
| `datasets` | Dataset loading and preprocessing |
| `wandb` | Experiment tracking and logging |

A CUDA-compatible GPU is required. `bfloat16` precision and `tf32` matrix operations are enabled, so an Ampere-class GPU (e.g., A100, 3090, 4090) or newer is strongly recommended.

---

## Supported Tasks

The `TASK` environment variable selects the classification task and controls prompt formatting.

### Sentiment Analysis (`TASK=sentiment`)

Performs three-class sentiment classification on Sinhala text.

**Prompt format:**
```
Does the following Sinhala sentence have a POSITIVE, NEGATIVE or NEUTRAL sentiment?

{text}

Answer: {label}
```

**Expected labels:** `POSITIVE`, `NEGATIVE`, `NEUTRAL`

---

### Writing Style Classification (`TASK=writing`)

Classifies Sinhala comments into four writing style categories.

**Prompt format:**
```
Classify the Sinhala comment into ACADEMIC, CREATIVE, NEWS, BLOG.

Comment: {text}
Answer: {label}
```

**Expected labels:** `ACADEMIC`, `CREATIVE`, `NEWS`, `BLOG`

---

### News Category Classification (`TASK=news`)

Classifies Sinhala news text into one of five topical categories, using integer labels.

**Prompt format:**
```
Classify into:
Political: 0, Business: 1, Technology: 2, Sports: 3, Entertainment: 4.

Comment: {text}
Answer: {label}
```

**Expected labels:** `0`, `1`, `2`, `3`, `4`

---

## Architecture and Stage Design

### Generative vs. Discriminative Classification

Traditional classification fine-tuning adds a linear classification head on top of a pooled encoder representation and trains with cross-entropy over class logits. This script instead uses a **generative approach**: the causal language model is trained to predict the label token(s) that follow the prompt. This approach:

- Requires no architectural modification (no new head layers)
- Is naturally compatible with instruction-tuned or chat-style models
- Allows the model to express uncertainty via generation probabilities
- Enables zero-shot transfer by simply changing the prompt

The trade-off is that evaluation requires text matching or constrained decoding rather than a simple argmax over a fixed-size output vector.

### Two-Stage Training

The same two-stage philosophy as `train_lora_cpt.py` applies here:

**Stage 1 — Embedding Warmup**: All transformer weights are frozen. Only `embed_tokens` and `lm_head` are updated. This allows the model to align task-specific vocabulary (label tokens, prompt keywords) with its internal representations before any structural changes.

**Stage 2 — LoRA Fine-Tuning**: LoRA adapters are applied to all major attention and feed-forward projection layers. Embeddings remain trainable. This is the primary stage for learning the classification mapping.

In practice, Stage 2 alone (using the CPT-trained model as the base) is the common entry point, since the upstream continual pre-training has already warmed up the embeddings.

---

## Configuration Reference

All settings are controlled via environment variables with defaults.

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/workspace/model/SinLlama_CPT` | Base model path (output of CPT pipeline) |
| `DATA_DIR` | `/workspace/data/classification` | Directory containing `train.jsonl` and `val.jsonl` |
| `OUT_DIR` | `/workspace/classification_out` | Root output directory |
| `TASK` | `sentiment` | Classification task: `sentiment`, `writing`, or `news` |
| `STAGE` | `2` | Training stage: `1` (embed warmup) or `2` (LoRA) |
| `SEQ_LEN` | `512` | Maximum tokenized sequence length |
| `MICRO_BS` | `8` | Per-device micro batch size |
| `GRAD_ACC` | `2` | Gradient accumulation steps |
| `EPOCHS` | `3` | Number of training epochs |
| `LR` | `1e-5` | Learning rate |
| `LORA_R` | `64` | LoRA rank (Stage 2 only) |

**Effective batch size** = `MICRO_BS × GRAD_ACC × number_of_GPUs`  
With defaults: `8 × 2 × 1 = 16` sequences of up to 512 tokens per step.

**Note:** The default stage is `2` (unlike `train_lora_cpt.py` which defaults to `1`), reflecting that classification fine-tuning typically starts from an already CPT-adapted model.

---

## Detailed Walkthrough

### Tokenizer Setup

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

Unlike the CPT script, the classification script uses **padded batches** (not packed sequences), so the `pad_token` assignment is functionally important here. The tokenizer must produce consistent-length sequences for batched training, and padding is used to fill shorter sequences to `SEQ_LEN`.

---

### Model Loading

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
)
model.config.use_cache = False
```

The model is expected to be the output of `train_lora_cpt.py` — specifically the `merged_bf16` Stage 2 artifact from the CPT pipeline. This ensures the model has already been adapted to the Sinhala language before task-specific fine-tuning begins.

- **`attn_implementation="sdpa"`**: Uses PyTorch's fused Scaled Dot-Product Attention for efficiency.
- **`use_cache=False`**: KV-cache is disabled during training; it's incompatible with gradient checkpointing and not needed for forward passes.

---

### Stage 1: Embedding Warmup

```python
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "embed_tokens" in name or "lm_head" in name:
        param.requires_grad = True
```

All parameters are frozen, then only the input embedding matrix and the output projection are enabled for training. This is useful if the task labels or prompt vocabulary contain tokens that need re-alignment before LoRA training.

**Trainable in Stage 1**: typically ~0.5–1% of total parameters.

---

### Stage 2: LoRA Fine-Tuning

```python
lora_cfg = LoraConfig(
    r=LORA_R,             # Default: 64
    lora_alpha=LORA_ALPHA, # = LORA_R → scaling factor 1.0
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
```

LoRA adapters are injected into all seven projection modules across both attention and feed-forward layers. The `modules_to_save` list ensures the full embedding and output projection weight matrices are saved alongside LoRA weights — critical because these layers may have been modified during CPT.

**Comparison with CPT script:**

| Setting | CPT (train_lora_cpt.py) | Classification (this script) |
|---|---|---|
| Default LORA_R | 128 | 64 |
| Default Stage | 1 | 2 |
| Default Epochs | 0.5 / 1.0 | 3 |
| Default LR | 1e-4 / 5e-6 | 1e-5 |
| Scheduler | linear / cosine | cosine |

The lower rank (64 vs 128) and fewer epochs reflect the smaller scale of classification datasets compared to CPT corpora.

---

### Dataset Loading

```python
dataset = load_dataset(
    "json",
    data_files={
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "validation": os.path.join(DATA_DIR, "val.jsonl"),
    }
)
```

The dataset is loaded from JSONL files with pre-defined train/validation splits. Unlike the CPT script (which creates its own split), this script expects both files to exist in `DATA_DIR`. See the [Data Format](#data-format) section for the required schema.

---

### Prompt Construction

```python
def build_prompt(example):
    text  = example["text"]
    label = str(example["label"])
    # ... construct task-specific prompt ...
    full_text = prompt + " " + label + tokenizer.eos_token
    return {"text": full_text}

dataset = dataset.map(build_prompt)
```

Each example is converted into a single string: `[prompt] [label][EOS]`. The label is always appended at the end, and an `eos_token` marks the end of the sequence. This formulation means:

- The model learns to **complete** the prompt by generating the correct label.
- During inference, you would truncate at `"Answer:"` and let the model generate the next token(s).
- The loss is computed over the **entire sequence** including the prompt tokens (see Tokenization below).

**Note:** For more precise fine-tuning, you may want to mask the prompt tokens in the loss (set `labels = -100` for prompt token positions). The current implementation computes loss over all tokens.

---

### Tokenization

```python
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
    remove_columns=["text", "label"]
)
```

Key behaviors:

- **`padding="max_length"`**: All sequences are padded to exactly `SEQ_LEN` (512 by default). This is necessary for uniform batch sizes without a custom collator.
- **`labels = input_ids.copy()`**: The labels are identical to the input IDs, which is standard causal language modeling (each token predicts the next). The `DataCollatorForLanguageModeling` is not used here; instead, labels are set manually.
- **`remove_columns=["text", "label"]`**: The original string columns are removed after tokenization. This is a critical fix noted in the source code (`# <-- THIS FIXES YOUR ERROR`), as leaving non-tensor columns causes the Trainer to fail when converting the dataset to tensors.
- **`batched=True`**: Processes multiple examples at once for faster mapping.

---

### Training Arguments

Key differences from the CPT training arguments:

| Argument | Value | Rationale |
|---|---|---|
| `save_strategy="epoch"` | Per epoch | Classification datasets are smaller; epoch-level checkpointing is appropriate |
| `eval_strategy="epoch"` | Per epoch | Matches save strategy; avoids overly frequent evaluation |
| `warmup_ratio=0.1` | 10% of steps | Longer warmup (vs 5% in CPT) for fine-tuning stability on smaller data |
| `lr_scheduler_type="cosine"` | Cosine | Smooth decay over 3 epochs |
| `num_train_epochs=3` | 3 epochs | Standard for classification fine-tuning |
| `run_name` | `{TASK}_stage{STAGE}_lr{LR}` | W&B run name set explicitly |

Shared settings with the CPT script include `bf16=True`, `tf32=True`, `gradient_checkpointing=True`, `optim="adamw_torch_fused"`, `weight_decay=0.01`, and `max_grad_norm=1.0`.

---

### Model Saving and Merging

**Stage 1** saves the full model and tokenizer directly:

```python
model.save_pretrained(stage_out)
tokenizer.save_pretrained(stage_out)
```

**Stage 2** performs a two-step save:

1. **Adapter save** — saves only the LoRA weights and `modules_to_save` matrices:
```python
model.save_pretrained(adapter_path)
```

2. **Merged save** — merges LoRA weights into the base model weights for a standalone inference-ready artifact:
```python
merged = model.merge_and_unload()
merged.save_pretrained(merged_path, safe_serialization=True)
```

The `merged_bf16` directory is the primary output for downstream inference or deployment.

---

## Data Format

Both `train.jsonl` and `val.jsonl` must be newline-delimited JSON files where each line contains at least two fields:

```json
{"text": "...", "label": "POSITIVE"}
{"text": "...", "label": "NEGATIVE"}
```

For the `news` task, labels should be integers (or integer strings):

```json
{"text": "...", "label": 2}
{"text": "...", "label": 0}
```

**Important:** Labels should be stored as the final token(s) the model should generate — exactly the string or number appended after `"Answer: "` in the prompt. Consistency between the label values in your data and the label strings in `build_prompt` is critical.

---

## Usage

### Stage 2 Only (Recommended Starting Point)

```bash
TASK=sentiment \
STAGE=2 \
MODEL_PATH=/path/to/sinllama_cpt/stage2/merged_bf16 \
DATA_DIR=/path/to/sentiment/data \
OUT_DIR=/path/to/classification_out \
EPOCHS=3 \
LR=1e-5 \
python train_lora_classification.py
```

### Stage 1 Then Stage 2

```bash
# Stage 1: Embed warmup
TASK=sentiment STAGE=1 MODEL_PATH=/path/to/cpt_model \
DATA_DIR=/data/sentiment OUT_DIR=/output python train_lora_classification.py

# Stage 2: LoRA fine-tuning on Stage 1 output
TASK=sentiment STAGE=2 MODEL_PATH=/output/sentiment_stage1 \
DATA_DIR=/data/sentiment OUT_DIR=/output python train_lora_classification.py
```

### Training All Three Tasks

```bash
for TASK in sentiment writing news; do
    TASK=$TASK STAGE=2 \
    DATA_DIR=/data/$TASK \
    OUT_DIR=/output/$TASK \
    python train_lora_classification.py
done
```

---

## Output Structure

```
OUT_DIR/
├── sentiment_stage2/
│   ├── adapters/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors   ← LoRA weights only
│   └── merged_bf16/
│       ├── config.json
│       ├── tokenizer.json
│       └── model.safetensors           ← Full merged model (inference-ready)
│
├── writing_stage2/
│   └── ...
│
└── news_stage2/
    └── ...
```

Each task produces its own subdirectory under `OUT_DIR`, named `{TASK}_stage{STAGE}`.

---

## Monitoring with Weights & Biases

W&B is initialized with:

```python
wandb.init(
    project="sinllama-classification",
    name=f"{TASK}_stage{STAGE}_lr{LR}",
)
```

Tracked metrics:

| Metric | Description |
|---|---|
| `loss` | Training cross-entropy loss |
| `eval_loss` | Validation cross-entropy loss |
| `learning_rate` | Current scheduler LR |
| `grad_norm` | Gradient norm (clipped at 1.0) |
| `epoch` | Current training epoch |

Unlike the CPT script, there is no `PerplexityCallback` in this script. If perplexity tracking is desired, it can be added following the same pattern as in `train_lora_cpt.py`.

---

## Relationship to train_lora_cpt.py

This script is the **downstream fine-tuning step** in a two-script pipeline:

```
Raw Sinhala Corpus
        │
        ▼
train_lora_cpt.py  (Stage 1 → Stage 2)
        │
        ▼
SinLlama_CPT / merged_bf16
        │
        ▼
train_lora_classification.py  (Stage 1 → Stage 2)
        │
        ▼
Task-Specific Classifier (sentiment / writing / news)
```

The classification script expects `MODEL_PATH` to point to the output of the CPT pipeline. Using the CPT model ensures the base model has already been adapted to Sinhala language patterns, vocabulary, and morphology before task-specific label learning begins.

---

## Design Rationale

### Why generative classification instead of a classification head?

Attaching a classification head requires the full model to be frozen or fine-tuned with a new random head, which introduces instability and requires careful learning rate balancing between the backbone and head. The generative approach reuses the existing `lm_head` (already adapted during CPT) and frames classification as a natural continuation of the prompt, making training more stable and architecture-free.

### Why `padding="max_length"` instead of dynamic padding?

Dynamic padding (with a data collator) would require removing the manual `labels` assignment, since the collator handles label shifting internally. Using fixed padding keeps the tokenization and label setup simple and explicit. The memory overhead is acceptable given the smaller batch sizes typical of classification training.

### Why is `lora_dropout=0.0`?

LoRA dropout is typically used to prevent overfitting on small supervised datasets. With moderately sized classification datasets and only 3 epochs of training, dropout adds noise without significant regularization benefit. Weight decay (`0.01`) is used instead for regularization.

### Why is `LORA_R=64` instead of 128 (as in CPT)?

Classification fine-tuning requires less adapter capacity than continual pre-training. The task is more focused (predict one label token), the dataset is smaller, and the model has already learned rich Sinhala representations from CPT. A rank of 64 provides sufficient expressiveness while reducing memory and training time.

### Why no prompt loss masking?

The labels are set to `input_ids.copy()`, meaning the loss is computed over every token including the prompt. A more precise approach would mask all prompt tokens (`labels[prompt_positions] = -100`) so the loss only reflects the model's ability to predict the label. This is a known limitation of the current implementation and can meaningfully improve performance by preventing the model from "wasting" gradient signal on learning to reproduce the static prompt text.

---

## Limitations and Considerations

- **No prompt loss masking**: Loss is computed over the full sequence including the prompt, which may dilute the classification learning signal. Consider masking prompt tokens in `labels` for better results.
- **Padding inefficiency**: `padding="max_length"` pads all sequences to 512 tokens, which wastes compute on short texts. Dynamic padding with `DataCollatorWithPadding` would be more efficient.
- **Single-GPU design**: `device_map="cuda"` targets one GPU. For multi-GPU setups, use `device_map="auto"` or launch with `accelerate`.
- **No accuracy metric**: Only `eval_loss` is tracked. For classification tasks, accuracy, F1, and per-class metrics would be more informative. These can be added via the `compute_metrics` argument to `Trainer`.
- **No inference utilities**: The script covers training only. Inference requires separate decoding logic with constrained generation or greedy decoding followed by label extraction from the generated text.
- **Task-specific label format must be consistent**: The label strings in `build_prompt` must exactly match those in the JSONL files. A mismatch (e.g., lowercase `"positive"` vs `"POSITIVE"`) will cause the model to learn to predict incorrect tokens.