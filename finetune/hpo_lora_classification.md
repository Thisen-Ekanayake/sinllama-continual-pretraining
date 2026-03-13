# The Story of `hpo_lora_classification.py`
### Teaching a Sinhala Language Model to Read Between the Lines

---

## Prologue: What Are We Trying to Do?

Somewhere inside a large language model lives latent knowledge — patterns, associations, representations — that can be repurposed for new tasks without retraining the whole thing from scratch. This script is the story of exploiting that knowledge to turn **SinLlama**, a Sinhala-language causal language model, into a text classifier.

The goal is straightforward: given a piece of Sinhala text, predict a label — a *sentiment*, a *writing style*, or a *news category*. Simple in concept, expensive in practice. Large models are memory-hungry, and GPU memory is finite. This script is as much a story about solving an engineering problem as it is about fine-tuning a model.

---

## Chapter 1: The World This Script Inhabits

Before anything runs, the script sets the stage — a set of **configuration constants** that describe the environment, the task, and the operating mode.

```
MODEL_PATH  →  Where the base SinLlama checkpoint lives
DATA_DIR    →  Where the training and validation JSONL files are
OUT_DIR     →  Where trained models and checkpoints are saved
LOG_DIR     →  Where per-epoch JSON metrics are written
```

These aren't hardcoded. Every one of them can be overridden by an environment variable, making the script equally at home in a local experiment, a Docker container, or a SLURM job.

Two of these settings carry particular narrative weight:

- **`TASK`** — One of `sentiment`, `writing`, or `news`. This single value shapes how the data is formatted, what the model is asked to predict, and what labels it must choose between.
- **`HPO_MODE`** — The fork in the road. Set it to `1` and the script launches a full hyperparameter search across 20 (or more) trials. Set it to `0` and it trains exactly once with whatever defaults are given.

---

## Chapter 2: The Three Tasks

The script isn't built for one problem — it's built for three, each defined in a dictionary called `TASK_META`.

Each task entry contains two things: a **prompt function** that wraps raw Sinhala text into a question the model can answer, and a list of valid **label strings** the model is expected to generate.

| Task | What it asks | Labels |
|---|---|---|
| `sentiment` | Is this POSITIVE, NEGATIVE, or NEUTRAL? | `POSITIVE`, `NEGATIVE`, `NEUTRAL` |
| `writing` | What style is this? | `ACADEMIC`, `CREATIVE`, `NEWS`, `BLOG` |
| `news` | What category? | `0`, `1`, `2`, `3`, `4` |

The prompts are deliberately conversational — they read like questions a human would ask — because the model was trained to complete natural language sequences. This framing helps it produce the right label token at the end, rather than something arbitrary.

---

## Chapter 3: Preparing the Data

The dataset arrives as two JSONL files — one for training, one for validation. Each line contains a `text` field (the Sinhala input) and a `label` field (the target class).

Here's what happens to each example:

**Step 1 — Build the prompt.**
The text is passed through the task's prompt function, producing something like:
```
Does the following Sinhala sentence have a POSITIVE, NEGATIVE or NEUTRAL sentiment?

<Sinhala text here>

Answer: POSITIVE
```

The full sequence, including the label and an end-of-sequence token, becomes the training target.

**Step 2 — Tokenize.**
The full text is tokenized and padded to `SEQ_LEN` (default 512 tokens).

**Step 3 — Mask the prompt.**
This is the most important step. The model is a *causal* language model — it learns by predicting each token from those before it. But we don't want it to waste effort learning to reproduce the prompt. We only want it to learn to predict the *label*.

To enforce this, all token positions corresponding to the prompt are replaced with `-100` in the label tensor. The HuggingFace `Trainer` ignores any position marked `-100` when computing the cross-entropy loss. The model ends up trained on exactly one or two tokens per example — the label itself.

---

## Chapter 4: The Memory Problem (and Its Solution)

When you ask `Trainer` to evaluate a model, it runs the model on every batch in the validation set, accumulates all the outputs in memory, and then calls `compute_metrics` at the end. For a small model, this is fine. For a LLaMA-class model with a vocabulary of 32,000+ tokens, the accumulated logit tensor has shape `(N_eval, seq_len, vocab_size)`.

That's 70+ gigabytes for a moderately sized evaluation set. It caused OOM crashes.

The fix is elegant: a hook called `preprocess_logits_for_metrics`, which the `Trainer` calls *per batch* before any accumulation happens.

For each example in the batch, the function:
1. Finds the last non-masked label position (where the label token is).
2. Extracts the logit vector at the position *just before* the label (the model's prediction position).
3. Takes the `argmax` to get the predicted token ID.

The result is a tensor of shape `(batch_size,)` — a single integer per example. The entire eval set now accumulates as a vector of integers, not a mountain of floats. Memory problem solved.

`compute_metrics` then decodes these predicted token IDs alongside the true label IDs, and computes **macro F1** and **accuracy** using scikit-learn.

---

## Chapter 5: The Two Training Stages

The script supports two philosophically different training modes, controlled by the `STAGE` environment variable.

### Stage 1 — Adaptation (Embedding + LM Head Only)

In Stage 1, almost everything in the model is frozen. Only two components remain trainable: the **token embedding table** (`embed_tokens`) and the **language model head** (`lm_head`).

This is a lightweight first pass — a way to orient the model's input and output representations toward the classification vocabulary before committing to full adapter training. It's fast, it's cheap, and it lays groundwork.

### Stage 2 — LoRA Fine-Tuning

Stage 2 is where the real learning happens. The model's weights remain frozen, but **Low-Rank Adaptation (LoRA)** adapters are injected into seven attention and MLP projection layers:

```
q_proj, k_proj, v_proj, o_proj
gate_proj, up_proj, down_proj
```

Each adapter introduces two small matrices — one of rank `r`, one of dimension `hidden_size × r` — whose product approximates the weight update the model would learn if fully fine-tuned. The rank `r` is a hyperparameter: higher values give the adapter more expressive capacity but consume more memory.

One important omission: the script deliberately **does not** include `modules_to_save` in the LoRA config. The base model uses tied word embeddings — `embed_tokens` and `lm_head` share the same weight tensor. PEFT's `modules_to_save` creates independent copies of saved modules per trial, which causes memory to balloon during HPO. Removing it sidesteps the OOM.

---

## Chapter 6: Training — The Careful Choreography

The `train()` function is the heart of the script. It wires together everything that came before and hands it to HuggingFace's `Trainer`.

Several decisions here are worth understanding:

**`bf16=True, tf32=False`**
The model trains in bfloat16, which halves the memory footprint of weights and activations compared to float32. `tf32`, a precision format exclusive to NVIDIA Ampere GPUs, is explicitly disabled — it crashes on AMD ROCm hardware (the MI300X this script targets).

**`gradient_checkpointing=True`**
During a forward pass, PyTorch saves intermediate activations so the backward pass can compute gradients from them. For a large model, this is expensive — it can consume as much VRAM as the model weights themselves. Gradient checkpointing recomputes activations during the backward pass instead of storing them. This roughly halves activation memory at the cost of about 20% slower training. For a 192 GB GPU budget, it's a necessary trade.

**`optim="adamw_torch_fused"`**
Adam-style optimizers maintain two extra tensors per parameter (moment estimates). For a large model, the optimizer state alone can use 4× the model's weight memory. The fused AdamW kernel computes updates in a single GPU operation per parameter, reducing peak memory during the optimizer step.

**`per_device_eval_batch_size=min(micro_bs, 16)`**
Evaluation is more memory-intensive than training because the model computes full logits over the vocabulary for each token. The eval batch size is capped at 16 regardless of the training batch size, to keep full-logit computation from overflowing VRAM.

**`device_map={"": torch.cuda.current_device()}`**
When Ray Tune spawns worker processes, each worker is assigned a specific GPU. This device map ensures the model loads onto whatever GPU the current process owns — rather than defaulting to GPU 0 and causing all workers to pile onto the same device.

---

## Chapter 7: The Epoch Logger — Watching the Model Learn

Every epoch, a custom callback called `EpochJSONLogger` wakes up and writes a record to disk.

It captures:
- The epoch number and global step
- A UTC timestamp
- The average training loss over the epoch
- All evaluation metrics (loss, F1, accuracy)

Each record is appended to a `.jsonl` file — one line per epoch, easy to parse with any JSON tooling. At the end of training, a summary file is written with the full history and a pointer to the best epoch by F1 score.

This gives any downstream analysis — or a researcher reviewing results the next morning — a complete, structured audit trail of how the model improved (or didn't) over time.

---

## Chapter 8: The Hyperparameter Search

When `HPO_MODE=1`, the script stops being a single training run and becomes an **orchestrator** of many.

Ray Tune manages the search. The script defines a search space across seven hyperparameters:

| Hyperparameter | Range |
|---|---|
| Learning rate | Log-uniform between 5e-6 and 5e-4 |
| LoRA rank | 8, 16, 32, or 64 |
| LoRA dropout | 0.0, 0.05, or 0.1 |
| Weight decay | 0.01, 0.05, or 0.1 |
| Micro batch size | 8, 16, or 32 |
| Gradient accumulation | 1, 2, or 4 steps |
| Epochs | 2, 3, or 5 |

Two algorithms collaborate to navigate this space:

**OptunaSearch** proposes the next configuration to try, using a Bayesian model of which regions of the search space have historically produced good results.

**ASHAScheduler** (Asynchronous Successive Halving Algorithm) is the enforcer. After each epoch, it compares all running trials against each other. Trials performing in the bottom half are terminated early. This means the search budget is concentrated on promising configurations rather than spent equally on everything — including dead ends.

Each trial runs sequentially, using one full GPU. There's no concurrency — a deliberate choice to prevent multiple large models from competing for memory on the same hardware.

---

## Chapter 9: After the Search — Retraining the Winner

The HPO loop doesn't end with just a recommendation. When `RETRAIN_BEST=1` (the default), the script takes the best configuration found during the search and trains it again — this time at full budget, for as many epochs as `RETRAIN_EPOCHS` specifies (default: 5).

This retraining run is logged to Weights & Biases and produces the final model artifact.

If `STAGE == 2`, this final model is saved in two forms:
1. **Adapter weights only** — the LoRA matrices, small and portable.
2. **Merged model** — the adapters mathematically merged back into the base weights, producing a standalone checkpoint that can be loaded without PEFT.

---

## Epilogue: The Flow, End to End

```
                    ┌─────────────────────────┐
                    │   Environment Variables │
                    │  (TASK, STAGE, HPO, ...)│
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │ HPO_MODE = 1     │                  │ HPO_MODE = 0
              ▼                  │                  ▼
     ┌─────────────────┐         │         ┌─────────────────┐
     │  Ray Tune HPO   │         │         │   Single Train  │
     │  (20 trials)    │         │         │   (fixed config)│
     │  ASHA + Optuna  │         │         └────────┬────────┘
     └────────┬────────┘         │                  │
              │                  │                  │
     Best config found           │                  │
              │                  │                  │
     Retrain at full budget      │                  │
              │                  │                  │
              └──────────────────┘──────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   train()              │
                    │  · load tokenizer      │
                    │  · build dataset       │
                    │  · build model         │
                    │    (Stage 1 or 2)      │
                    │  · run Trainer         │
                    │  · log epochs to JSON  │
                    │  · save adapters       │
                    │  · merge weights       │
                    └────────────────────────┘
```

---

## Appendix: Key Files Produced

| File | Contents |
|---|---|
| `<run_name>_epochs.jsonl` | One JSON record per epoch (loss, F1, accuracy, timestamp) |
| `<run_name>_summary.json` | Full history + best epoch pointer |
| `run_config.json` | Exact hyperparameters used for this run |
| `adapters/` | LoRA adapter weights (Stage 2 only) |
| `merged_bf16/` | Full merged model in bfloat16 (Stage 2 only) |
| `hpo_<task>_stage<N>_results.json` | Best config and F1 from HPO run |

---

## Appendix: Quick-Start Cheat Sheet

```bash
# Single training run — sentiment, Stage 2
TASK=sentiment STAGE=2 python hpo_lora_classification.py

# Hyperparameter search — 30 trials
TASK=news STAGE=2 HPO=1 HPO_TRIALS=30 python hpo_lora_classification.py

# Custom paths
MODEL_PATH=/my/model DATA_DIR=/my/data OUT_DIR=/my/output \
  TASK=writing STAGE=1 python hpo_lora_classification.py

# Skip retraining after HPO
HPO=1 RETRAIN_BEST=0 python hpo_lora_classification.py
```