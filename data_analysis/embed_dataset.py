#!/usr/bin/env python3
"""
Embed SinLlama-Dataset.txt with HelaBERT and save outputs for dataset shift analysis.

SinLlama is Dataset A (the corpus the model was already trained on) in the analysis.

Outputs (in project/outputs/):
    centroid_A.npy          — full-corpus mean CLS embedding  (768,)  float32
    embeddings_A_sample.npy — reservoir sample of 50 K embeddings  (50000, 768)  float32
    embedding_stats.json    — {"n_a": N}

Expected structure (run from project/ root):
    project/
    ├── HelaBERT_Large/
    │   ├── HelaBERT_Large/         ← config.json, model.safetensors, training_args.bin
    │   └── tokenizer/              ← unigram_32000_0.9995.model
    ├── SinLlama-Dataset.txt
    ├── embed_dataset.py
    └── outputs/

Usage:
    python embed_dataset.py
    python embed_dataset.py --batch-size 1024   # override batch size
"""

import argparse
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import sentencepiece as spm
from transformers import BertModel
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=512,
                    help="Lines per GPU batch (default: 512)")
parser.add_argument("--max-len", type=int, default=128,
                    help="Max subword tokens per line incl. CLS/SEP (default: 128)")
parser.add_argument("--reservoir-size", type=int, default=50_000,
                    help="Reservoir sample size (default: 50000)")
args = parser.parse_args()

BATCH_SIZE     = args.batch_size
MAX_LEN        = args.max_len
RESERVOIR_SIZE = args.reservoir_size
HIDDEN_DIM     = 768
CLS_ID, SEP_ID, PAD_ID = 2, 3, 0

# ── paths ──────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent
MODEL_PATH = BASE / "HelaBERT_Large" / "HelaBERT_Large"
SP_MODEL   = BASE / "HelaBERT_Large" / "tokenizer" / "unigram_32000_0.9995.model"
TEXT_FILE  = BASE / "SinLlama-Dataset.txt"
OUT_DIR    = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_CENTROID = OUT_DIR / "centroid_A.npy"
OUT_SAMPLE   = OUT_DIR / "embeddings_A_sample.npy"
OUT_STATS    = OUT_DIR / "embedding_stats.json"

# ── guard: already done ────────────────────────────────────────────────────────
if OUT_CENTROID.exists() and OUT_SAMPLE.exists():
    print("Outputs already exist — nothing to do.")
    print(f"  {OUT_CENTROID}  ({OUT_CENTROID.stat().st_size} bytes)")
    print(f"  {OUT_SAMPLE}  ({OUT_SAMPLE.stat().st_size / 1e6:.1f} MB)")
    raise SystemExit(0)

if not TEXT_FILE.exists():
    raise FileNotFoundError(
        f"{TEXT_FILE} not found.\n"
        "Run sinllama_dataset_to_text.py first."
    )

# ── device ─────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
if device == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected — embedding will be very slow on CPU.")

# ── load model + tokenizer ────────────────────────────────────────────────────
print(f"\nLoading HelaBERT from {MODEL_PATH} …")
model = BertModel.from_pretrained(str(MODEL_PATH))
model = model.to(device).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters : {n_params:,}")

sp = spm.SentencePieceProcessor()
sp.Load(str(SP_MODEL))
print(f"  Vocab size : {sp.GetPieceSize()}")

# ── config summary ─────────────────────────────────────────────────────────────
print(f"\nConfig:")
print(f"  Batch size     : {BATCH_SIZE}")
print(f"  Max length     : {MAX_LEN} tokens")
print(f"  Reservoir size : {RESERVOIR_SIZE:,}")

# ── count lines for ETA ────────────────────────────────────────────────────────
print(f"\nCounting lines in {TEXT_FILE.name} …")
t0 = time.time()
with open(TEXT_FILE, encoding="utf-8", errors="ignore") as f:
    total_lines = sum(1 for line in f if line.strip())
print(f"  {total_lines:,} lines  ({time.time()-t0:.1f}s)")

# ── tokenisation ───────────────────────────────────────────────────────────────
def tokenize_batch(lines):
    all_ids, all_masks = [], []
    for line in lines:
        ids  = [CLS_ID] + sp.EncodeAsIds(line.strip())[:MAX_LEN - 2] + [SEP_ID]
        mask = [1] * len(ids)
        all_ids.append(ids)
        all_masks.append(mask)
    max_L = max(len(x) for x in all_ids)
    for i in range(len(all_ids)):
        pad = max_L - len(all_ids[i])
        all_ids[i]   += [PAD_ID] * pad
        all_masks[i] += [0]      * pad
    return (torch.tensor(all_ids,   dtype=torch.long),
            torch.tensor(all_masks, dtype=torch.long))

# ── embedding pass ─────────────────────────────────────────────────────────────
centroid_sum = np.zeros(HIDDEN_DIM, dtype=np.float64)
n_total      = 0
reservoir    = np.zeros((RESERVOIR_SIZE, HIDDEN_DIM), dtype=np.float32)
buf          = []
t_start      = time.time()

def flush(lines):
    global centroid_sum, n_total
    ids, masks = tokenize_batch(lines)
    with torch.no_grad():
        out  = model(input_ids=ids.to(device), attention_mask=masks.to(device))
        embs = out.last_hidden_state[:, 0, :].float().cpu().numpy()
    centroid_sum += embs.sum(axis=0)
    for k, emb in enumerate(embs):          # Algorithm R reservoir sampling
        g = n_total + k
        if g < RESERVOIR_SIZE:
            reservoir[g] = emb
        else:
            j = random.randint(0, g)
            if j < RESERVOIR_SIZE:
                reservoir[j] = emb
    n_total += len(embs)

print(f"\nEmbedding …")
pbar = tqdm(
    total=total_lines, unit="line", desc="A (SinLlama)",
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    mininterval=10.0,
)

with open(TEXT_FILE, encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        buf.append(line)
        if len(buf) == BATCH_SIZE:
            flush(buf)
            pbar.update(len(buf))
            buf.clear()
    if buf:
        flush(buf)
        pbar.update(len(buf))
        buf.clear()

pbar.close()

elapsed   = time.time() - t_start
hrs, rem  = divmod(int(elapsed), 3600)
mins, sec = divmod(rem, 60)
print(f"\n✓  Embedded {n_total:,} lines in {hrs:02d}h {mins:02d}m {sec:02d}s  "
      f"| {n_total / elapsed:,.0f} lines/s")

# ── save outputs ───────────────────────────────────────────────────────────────
centroid = (centroid_sum / n_total).astype(np.float32)
sample   = reservoir[:min(n_total, RESERVOIR_SIZE)].copy()

np.save(OUT_CENTROID, centroid)
np.save(OUT_SAMPLE,   sample)
OUT_STATS.write_text(json.dumps({"n_a": n_total, "label": "A (SinLlama)"}))

print(f"\nSaved to {OUT_DIR}/")
print(f"  centroid_A.npy          : {OUT_CENTROID.stat().st_size} bytes  shape={centroid.shape}")
print(f"  embeddings_A_sample.npy : {OUT_SAMPLE.stat().st_size / 1e6:.1f} MB  shape={sample.shape}")
print(f"  embedding_stats.json    : n_a={n_total:,}")
