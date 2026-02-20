import os
from transformers import AutoTokenizer
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/model/SinLlama_merged_bf16")
TXT_PATH   = os.environ.get("TXT_PATH",   "/workspace/data/All-Text_8696658_147190824.normalized.txt")

DATA_PERC  = float(os.environ.get("DATA_PERC", "1.0"))  # e.g., 0.2 = 20%
SEQ_LEN    = int(os.environ.get("SEQ_LEN", "512"))

# ============================================================
# Load tokenizer
# ============================================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# Count lines first (to apply DATA_PERC safely)
# ============================================================

print("Counting total lines...")
with open(TXT_PATH, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

use_lines = int(total_lines * DATA_PERC)

print(f"Total lines in file : {total_lines:,}")
print(f"Using lines         : {use_lines:,} ({DATA_PERC*100:.1f}%)")

# ============================================================
# Token counting
# ============================================================

total_tokens = 0
processed_lines = 0

print("\nTokenizing...")

with open(TXT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(tqdm(f, total=use_lines)):
        if i >= use_lines:
            break

        text = line.strip()
        if not text:
            continue

        ids = tokenizer(
            text + tokenizer.eos_token,
            add_special_tokens=False
        ).input_ids

        total_tokens += len(ids)
        processed_lines += 1

# ============================================================
# Results
# ============================================================

avg_tokens = total_tokens / max(processed_lines, 1)
packed_sequences = total_tokens // SEQ_LEN

print("\n" + "="*60)
print(f"Processed lines        : {processed_lines:,}")
print(f"Total tokens           : {total_tokens:,}")
print(f"Average tokens/line    : {avg_tokens:.2f}")
print(f"Equivalent {SEQ_LEN}-token sequences: {packed_sequences:,}")
print("="*60)