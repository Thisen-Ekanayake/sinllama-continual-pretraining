# evaluate.py — Documentation

> **Inference & Evaluation Script for SinLlama Classification Tasks**  
> A standalone evaluation pipeline that runs greedy inference on a fine-tuned SinLlama causal language model, parses generated label tokens, and produces a comprehensive classification report with confusion matrices, per-class accuracy, and UNK diagnostics.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Relationship to the Training Pipeline](#relationship-to-the-training-pipeline)
4. [Configuration Reference](#configuration-reference)
5. [Detailed Walkthrough](#detailed-walkthrough)
   - [Model Loading](#model-loading)
   - [Prompt Construction](#prompt-construction)
   - [Inference Loop](#inference-loop)
   - [Prediction Parsing](#prediction-parsing)
   - [Metrics Computation](#metrics-computation)
   - [Report Generation](#report-generation)
6. [Output File Format](#output-file-format)
7. [UNK Prediction Diagnostics](#unk-prediction-diagnostics)
8. [Usage](#usage)
9. [Design Rationale](#design-rationale)
10. [Limitations and Considerations](#limitations-and-considerations)

---

## Overview

`evaluate.py` performs **zero-shot greedy inference** on a held-out test set using a fine-tuned SinLlama causal language model. It mirrors the prompt format used during training (`train_lora_classification.py`) and applies task-specific regex/keyword parsing to extract clean label predictions from the model's generated text.

The script produces a structured plain-text report saved to disk containing:

- True and predicted label distributions
- Full `sklearn` classification report (precision, recall, F1, support)
- Raw count confusion matrix
- Row-normalized confusion matrix
- Per-class accuracy breakdown
- UNK (unparseable) prediction diagnostics with example outputs

It is the **final stage** of the three-script SinLlama pipeline:

```
train_lora_cpt.py  →  train_lora_classification.py  →  evaluate.py
```

---

## Requirements

| Package | Purpose |
|---|---|
| `torch` | Inference on GPU |
| `transformers` | Model and tokenizer loading, text generation |
| `sklearn` | `classification_report`, `confusion_matrix` |
| `tqdm` | Progress bar over test examples |
| `json` | JSONL test file parsing |
| `re` | Regex-based label extraction for news and writing tasks |

A CUDA-capable GPU is recommended. The model is loaded in `bfloat16` with `device_map="auto"`, allowing it to span multiple GPUs or fall back to CPU if needed.

---

## Relationship to the Training Pipeline

`evaluate.py` is designed to evaluate models produced by `train_lora_classification.py`. The expected `MODEL_PATH` is the `merged_bf16` artifact from Stage 2 of classification fine-tuning:

```
train_lora_cpt.py
    └── stage2/merged_bf16/           ← SinLlama CPT model
            │
            ▼
train_lora_classification.py
    └── {task}_stage2/merged_bf16/    ← Task-specific classifier
            │
            ▼
evaluate.py
    └── results/{task}_classification.txt  ← Evaluation report
```

The prompt templates in `evaluate.py` are **identical** to those in `train_lora_classification.py`, which is critical — any mismatch between training prompts and evaluation prompts will degrade performance.

---

## Configuration Reference

All settings are controlled via environment variables.

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/workspace/model/SinLlama_CPT` | Path to the merged fine-tuned model |
| `TEST_FILE` | `/workspace/data/classification/test.jsonl` | Path to the JSONL test set |
| `TASK` | `sentiment` | Evaluation task: `sentiment`, `writing`, or `news` |
| `MAX_NEW_TOKENS` | `5` | Maximum tokens to generate after the prompt |

The `RESULTS_DIR` is hardcoded to `/workspace/results` and created automatically. The output file is named `{TASK}_classification.txt`.

### MAX_NEW_TOKENS

A value of `5` is intentionally small. Because labels are short (e.g., `"POSITIVE"`, `"0"`, `"ACADEMIC"`), generating more tokens wastes compute and increases the chance of the model producing verbose output that confuses the parser. If labels in your dataset are longer, increase this value accordingly.

---

## Detailed Walkthrough

### Model Loading

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
```

Key differences from the training scripts:

- **`device_map="auto"`** (vs `"cuda"` during training): Automatically distributes model layers across all available GPUs, or falls back to CPU. This is better suited for inference where memory pressure may vary.
- **`model.eval()`**: Disables dropout and sets the model to evaluation mode. This is required for deterministic inference.
- **No `use_cache=False`**: Unlike training, KV-cache is enabled (default) during inference, which significantly speeds up autoregressive generation.
- **No LoRA/PEFT wrapper**: The script loads a fully merged model, so no PEFT imports are needed.

---

### Prompt Construction

```python
def build_prompt(text: str) -> str:
    if TASK == "sentiment":
        return f"""Does the following Sinhala sentence have a POSITIVE, NEGATIVE or NEUTRAL sentiment?

{text}

Answer:"""
    ...
```

The prompt ends with `"Answer:"` and no trailing space, so the model's first generated token is the start of the label. This matches the training format exactly: during training, `"Answer: " + label + eos_token` was appended to form the full sequence.

All three task prompts are identical to those in `train_lora_classification.py`. This consistency is essential — the model was trained to respond to these exact phrasings.

---

### Inference Loop

```python
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for idx, line in enumerate(tqdm(f), start=1):
        row = json.loads(line)
        text = row["text"]
        label = str(row["label"])

        prompt = build_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = parse_prediction(decoded)
```

Key design choices:

- **`torch.no_grad()`**: Disables gradient computation, reducing memory usage and speeding up inference.
- **`do_sample=False`**: Greedy decoding — always picks the highest-probability next token. This ensures deterministic, reproducible results.
- **`temperature=None, top_p=None`**: Explicitly disables sampling parameters to avoid warnings when the model's `generation_config` has these set from training. Without this, HuggingFace may warn about conflicting settings.
- **`skip_special_tokens=True`**: Removes `<pad>`, `<eos>`, and other special tokens from the decoded output for cleaner parsing.
- **One example at a time**: The loop processes each JSONL line individually (batch size 1). This simplifies parsing and avoids padding issues, at the cost of slower throughput compared to batched inference.

---

### Prediction Parsing

The `parse_prediction` function extracts a clean label from the model's full decoded output. Because the decoded string includes the prompt, the function first isolates the generated portion by splitting on `"Answer:"`:

```python
raw = decoded.split("Answer:")[-1].strip()
```

Then task-specific extraction is applied:

#### News Task (`TASK=news`)

```python
m = re.search(r"\b([0-4])\b", raw)
return m.group(1) if m else "UNK"
```

Uses a regex with word boundaries (`\b`) to find a digit between 0 and 4. This fixes the "0," issue mentioned in the source — without word boundaries, a string like `"0, Political"` might match incorrectly or fail if the digit is part of a longer number. The first match is taken.

#### Sentiment Task (`TASK=sentiment`)

```python
ru = raw.upper()
if "POSITIVE" in ru:   return "POSITIVE"
if "NEGATIVE" in ru:   return "NEGATIVE"
if "NEUTRAL"  in ru:   return "NEUTRAL"
return "UNK"
```

Case-insensitive substring search. Note that `"POSITIVE"` is checked before `"NEGATIVE"` — this ordering matters if both appear in the output (unlikely but possible). The priority order is `POSITIVE → NEGATIVE → NEUTRAL`.

#### Writing Task (`TASK=writing`)

```python
ru = raw.upper()
if "ACADEMIC" in ru:               return "ACADEMIC"
if "CREATIVE" in ru:               return "CREATIVE"
if re.search(r"\bNEWS\b", ru):     return "NEWS"
if "BLOG" in ru:                   return "BLOG"
return "UNK"
```

`"NEWS"` uses a word-boundary regex rather than a plain `in` check to avoid false positives — for example, the word `"NEWSWORTHY"` or the phrase `"CLASSIFY INTO ... NEWS ..."` appearing in the prompt could incorrectly match. `"ACADEMIC"`, `"CREATIVE"`, and `"BLOG"` are less ambiguous and use simple substring checks.

#### UNK Handling

If no label can be extracted, `"UNK"` is returned. Up to 5 UNK examples are captured with their decoded output tail for debugging, and a summary count is reported both in the output file and as a console warning.

---

### Metrics Computation

After all predictions are collected, three metric structures are computed:

#### 1. Classification Report

```python
report = classification_report(y_true, y_pred, digits=4, zero_division=0)
```

Produces per-class and macro/weighted averages for precision, recall, and F1-score. `zero_division=0` prevents warnings for classes with no predicted instances. `digits=4` gives 4 decimal places for finer comparison between models.

#### 2. Confusion Matrix

```python
cm = confusion_matrix(y_true, y_pred, labels=label_order)
```

The `labels=label_order` argument enforces a consistent ordering of rows and columns regardless of which classes appear in the test set. Label orders are:

| Task | Order |
|---|---|
| `news` | `0, 1, 2, 3, 4` |
| `sentiment` | `NEGATIVE, NEUTRAL, POSITIVE` |
| `writing` | `ACADEMIC, CREATIVE, NEWS, BLOG` |

#### 3. Per-Class Accuracy

```python
for i, lab in enumerate(label_order):
    total = cm[i].sum()
    correct = cm[i, i]
    acc = safe_div(correct, total)
```

Computed as the diagonal element divided by the row sum for each class. This is the recall (sensitivity) per class — the fraction of true instances of each class that were correctly predicted. The `safe_div` helper prevents division-by-zero for missing classes.

#### 4. Row-Normalized Confusion Matrix

```python
cm_norm = [[round(v / row_sum, 4) for v in row] for row in cm]
```

Each row is divided by its sum, converting counts to proportions. This makes it easier to see error patterns relative to class frequency — especially important when classes are imbalanced.

---

### Report Generation

The output file is written in plain text with ASCII-art formatting for terminal readability and easy sharing. The `format_matrix` helper constructs monospaced confusion matrix tables:

```python
def format_matrix(labels, matrix, title=None):
    col_width = max(6, max(len(l) for l in labels) + 2)
    header = " " * col_width + "".join(l.rjust(col_width) for l in labels)
    ...
```

Column width auto-adapts to the longest label name, ensuring alignment is preserved for all three tasks. Both the raw count matrix and the row-normalized matrix use this formatter.

---

## Output File Format

The output file `results/{TASK}_classification.txt` contains the following sections in order:

```
Classification Evaluation Report
========================================
Timestamp      : 2025-02-23 14:32:01.123456
Task           : sentiment
Model Path     : /workspace/model/...
Test File      : /workspace/data/.../test.jsonl
Max New Tokens : 5
----------------------------------------

Label Distribution (True)
-------------------------
NEGATIVE: 412
NEUTRAL: 308
POSITIVE: 280

Label Distribution (Pred)
-------------------------
NEGATIVE: 401
NEUTRAL: 315
POSITIVE: 278
UNK: 6

Classification Report
---------------------
              precision    recall  f1-score   support
    NEGATIVE     0.9123    0.8932    0.9027       412
     NEUTRAL     0.8754    0.8994    0.8872       308
    POSITIVE     0.9021    0.9107    0.9064       280
    accuracy                         0.8999      1000
   macro avg     0.8966    0.9011    0.8988      1000
weighted avg     0.9003    0.8999    0.9000      1000

Confusion Matrix (Counts)
-------------------------
            NEGATIVE  NEUTRAL  POSITIVE
    NEGATIVE      368       28       16
     NEUTRAL       22      277        9
    POSITIVE       12       13      255

Per-Class Accuracy (Correct/Total)
----------------------------------
    NEGATIVE :  368/412   acc=0.8932
     NEUTRAL :  277/308   acc=0.8994
    POSITIVE :  255/280   acc=0.9107

Confusion Matrix (Row-normalized)
---------------------------------
            NEGATIVE  NEUTRAL  POSITIVE
    NEGATIVE   0.8932   0.0680   0.0388
     NEUTRAL   0.0714   0.8994   0.0292
    POSITIVE   0.0429   0.0464   0.9107

UNK Predictions
---------------
UNK count: 6

UNK Examples (first 5)
----------------------

[#47] true=NEUTRAL
text: ...
decoded_tail:
...
```

---

## UNK Prediction Diagnostics

`UNK` predictions indicate that the model generated output from which no valid label could be extracted. Common causes include:

- The model generating verbose explanations instead of a single label token (e.g., `"This sentence appears to express..."`)
- Label token appearing in an unexpected format or language
- `MAX_NEW_TOKENS` being too small to reach the label token
- Prompt mismatch between training and evaluation

The script captures up to 5 UNK examples with:
- The example index in the test file
- The true label
- The first 200 characters of the input text
- The last 300 characters of the decoded model output (the `decoded_tail`) — the most useful field for debugging parser failures

A console warning is printed if any UNK predictions exist:

```
[WARN] UNK predictions: 6 (see file for examples)
```

High UNK rates (>5%) typically indicate a prompt mismatch, an insufficiently fine-tuned model, or a `MAX_NEW_TOKENS` setting that is too restrictive.

---

## Usage

### Basic Evaluation

```bash
TASK=sentiment \
MODEL_PATH=/workspace/classification_out/sentiment_stage2/merged_bf16 \
TEST_FILE=/workspace/data/sentiment/test.jsonl \
python evaluate.py
```

### All Three Tasks

```bash
for TASK in sentiment writing news; do
    TASK=$TASK \
    MODEL_PATH=/workspace/classification_out/${TASK}_stage2/merged_bf16 \
    TEST_FILE=/workspace/data/$TASK/test.jsonl \
    python evaluate.py
done
```

### Longer Generation Window

If you observe high UNK rates, try increasing `MAX_NEW_TOKENS`:

```bash
MAX_NEW_TOKENS=15 TASK=writing python evaluate.py
```

### Evaluating the CPT Base Model (Zero-Shot Baseline)

To measure how much fine-tuning helped, evaluate the CPT model before classification fine-tuning:

```bash
MODEL_PATH=/workspace/model/SinLlama_CPT \
TASK=sentiment \
TEST_FILE=/workspace/data/sentiment/test.jsonl \
python evaluate.py
```

---

## Design Rationale

### Why greedy decoding (`do_sample=False`)?

Classification evaluation should be **deterministic and reproducible**. Sampling introduces randomness that makes results non-comparable across runs. Greedy decoding always selects the most probable token, giving a stable benchmark.

### Why decode the full sequence and split on `"Answer:"`?

`model.generate()` returns the full input + generated tokens concatenated. Splitting on `"Answer:"` and taking the last segment isolates only the generated label portion. This is more robust than slicing by input length (which could be fragile under different tokenizer behaviors).

### Why not use constrained decoding (e.g., force the model to choose from valid labels)?

Constrained decoding (e.g., via `LogitsProcessor`) would guarantee valid outputs and eliminate UNK entirely. However, it adds complexity and may mask genuine model failures — if a model is uncertain, forcing a label gives artificially inflated accuracy. The current approach exposes model failures via the UNK mechanism, making evaluation more informative.

### Why plain text output instead of JSON?

Plain text with ASCII tables is immediately human-readable in a terminal, easily shareable via email or paste, and requires no additional tooling to inspect. JSON would be more machine-readable for downstream processing but less convenient for quick result inspection.

### Why `batch_size=1` inference?

Single-example inference avoids padding complexity (different prompts have different lengths), simplifies label extraction, and keeps the code straightforward. For large test sets requiring faster evaluation, batched inference with dynamic padding and careful output decoding would be a worthwhile optimization.

---

## Limitations and Considerations

- **Single-example inference**: Processing one example at a time is slow for large test sets. Batched inference could provide 4–8× speedup at the cost of additional padding logic.
- **No batched GPU utilization**: The current loop underutilizes GPU throughput. For test sets larger than ~5,000 examples, consider implementing batched inference.
- **Parser priority assumptions**: For sentiment, `POSITIVE` is checked before `NEGATIVE` and `NEUTRAL`. If the model frequently generates all three words in its output, this ordering introduces a bias. Inspect `decoded_tail` in UNK examples to verify parser assumptions hold.
- **No confidence scores**: The script does not extract token probabilities or log-likelihoods for the predicted label. Adding probability-based scoring (e.g., the probability of the first generated token) would enable threshold-based rejection and calibration analysis.
- **UNK treated as a wrong prediction**: UNK predictions are included in `y_pred` and compared against true labels, always counting as incorrect. This is the correct behavior for accuracy metrics but means UNK rate directly impacts reported scores.
- **Hardcoded results directory**: `RESULTS_DIR` is hardcoded to `/workspace/results`. Unlike the model and data paths, it cannot be overridden via environment variable without modifying the script.
- **No multi-GPU batching**: While `device_map="auto"` supports multi-GPU model placement, the inference loop is single-threaded and does not parallelize across examples.