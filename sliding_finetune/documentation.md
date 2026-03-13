# SinLlama Classification Fine-tuning — Documentation

Three LoRA fine-tuning scripts for SinLlama (LLaMA 3 8B, Sinhala CPT) covering news, sentiment, and writing style classification. All three share the same architecture and training pipeline.

---

## Scripts Overview

| Script | Task | Labels |
|---|---|---|
| `train_news_classification.py` | News topic classification | political, business, technology, sports, entertainment |
| `train_sentiment_classification.py` | Sentiment analysis | positive, negative |
| `train_writing_classification.py` | Writing style classification | academic, creative, news, blog |

---

## Dataset Format

Each script expects three JSONL files under its data directory:

```
data_new/<task>/
├── train.jsonl
├── val.jsonl
└── test.jsonl
```

**News** — `label` is an integer, `label_name` is a lowercase string:
```json
{"text": "...", "label": 3, "label_name": "sports"}
```

**Sentiment** — `label` is an uppercase string (`POSITIVE` / `NEGATIVE`), no `label_name` field:
```json
{"text": "...", "label": "POSITIVE"}
```

**Writing** — `label` is an uppercase string (`ACADEMIC`, `CREATIVE`, `NEWS`, `BLOG`), no `label_name` field:
```json
{"text": "...", "label": "CREATIVE"}
```

---

## Key Methods

### Sliding-Window Chunking
Long texts are tokenised and split into overlapping chunks (default: 400-token windows, 200-token stride, max 8 chunks per sample). Each chunk becomes an independent training example with the label repeated, so the model learns to classify from any segment of the text.

```
article (1200 tokens)
  → chunk 1: tokens   0–400  → prompt + label
  → chunk 2: tokens 200–600  → prompt + label
  → chunk 3: tokens 400–800  → prompt + label
  → chunk 4: tokens 600–1000 → prompt + label
```

This expands the effective dataset size (printed at startup as `×N expansion`) and helps with long Sinhala articles that exceed the sequence length.

### Label-Only Loss Masking
The prompt prefix is masked with `-100` so only the answer token(s) contribute to the cross-entropy loss. This prevents the model from being penalised for the instruction text and focuses training on the classification decision.

### Label Token Scoring at Inference
Rather than free-form generation, predictions are made by scoring only the first subword tokens of each label name against each other (e.g. `▁sports` vs `▁political`). This guarantees a valid prediction every time and makes `compute_metrics` reliable.

> **Important for writing classification:** the script asserts all four label first-tokens are distinct. If the assertion fails, rename a label in `build_prompt` (e.g. `"journalistic"` instead of `"news"`) to avoid collision.

### `compute_metrics`
Ported directly from `sentiment_context_aware_finetune.py`. Computed at every eval epoch and used to select the best checkpoint:

- Accuracy
- Macro Precision / Recall / F1
- Weighted F1

Best model is saved by **macro F1** (`metric_for_best_model="f1"`, `greater_is_better=True`).

---

## Training Setup

All scripts use **LoRA Stage 2 only** (no embed/lm_head warmup stage).

| Hyperparameter | Default | Override via |
|---|---|---|
| LoRA rank | 64 | `LORA_R` |
| Learning rate | 1e-5 | `LR` |
| Epochs | 3 | `EPOCHS` |
| Sequence length | 512 | `SEQ_LEN` |
| Micro batch size | 4 | `MICRO_BS` |
| Gradient accumulation | 4 (eff. BS=16) | `GRAD_ACC` |
| Chunk size / stride | 400 / 200 | `CHUNK_SIZE`, `CHUNK_STRIDE` |
| Max chunks per sample | 8 | `MAX_CHUNKS` |

LoRA targets all attention and MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`) with `embed_tokens` and `lm_head` saved as full modules.

---

## Running

```bash
# News
MODEL_PATH=/workspace/model/SinLlama_CPT \
DATA_DIR=data_new/news \
OUT_DIR=outputs/news_lora \
python train_news_classification.py

# Sentiment
MODEL_PATH=/workspace/model/SinLlama_CPT \
DATA_DIR=data_new/sentiment \
OUT_DIR=outputs/sentiment_lora \
python train_sentiment_classification.py

# Writing style
MODEL_PATH=/workspace/model/SinLlama_CPT \
DATA_DIR=data_new/writing \
OUT_DIR=outputs/writing_lora \
python train_writing_classification.py
```

All hyperparameters are configurable via environment variables — no need to edit the scripts.

---

## Outputs

Each run saves the following under `OUT_DIR`:

```
OUT_DIR/
├── adapters/                  # LoRA adapter weights (load with PEFT)
├── merged_bf16/               # Full merged model in bfloat16
├── predictions_test.csv       # Per-chunk predictions + confidence scores
├── per_class_metrics_test.csv # Precision / Recall / F1 / Support per label
└── confusion_matrix_test.csv  # N×N confusion matrix
```

The merged model can be loaded like any standard HuggingFace model without PEFT installed.

---

## W&B Logging

Set `USE_WANDB=false` to disable. When enabled, the following are logged:

- Full hyperparameter config on run init
- Train/val label distribution histograms
- Per-epoch eval metrics (accuracy, F1, precision, recall)
- Test-set metrics and per-class bar charts
- Per-class metrics table

Projects are named `sinllama-news-classification`, `sinllama-sentiment-classification`, and `sinllama-writing-classification` respectively.