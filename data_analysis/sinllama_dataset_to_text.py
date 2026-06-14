#!/usr/bin/env python3
"""
Extract text from SinLlama parquet dataset to a plain text file.
One document per line (internal newlines collapsed to a space).

Expected structure (run from project/ root):
    project/
    ├── SinLlama-Dataset/   ← parquet files here
    └── SinLlama-Dataset.txt  ← written here

Usage:
    python sinllama_dataset_to_text.py
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE        = Path(__file__).parent
DATASET_DIR = BASE / "SinLlama-Dataset"
OUTPUT_FILE = BASE / "SinLlama-Dataset.txt"

parquet_files = sorted(DATASET_DIR.glob("train-*.parquet"))
if not parquet_files:
    raise FileNotFoundError(f"No parquet files found in {DATASET_DIR}")

print(f"Found {len(parquet_files)} parquet file(s):")
for pf in parquet_files:
    print(f"  {pf.name}  ({pf.stat().st_size / 1e6:.0f} MB)")

if OUTPUT_FILE.exists():
    print(f"\nOutput already exists: {OUTPUT_FILE}  ({OUTPUT_FILE.stat().st_size / 1e9:.2f} GB)")
    print("Delete it to re-extract.")
    raise SystemExit(0)

total_rows = 0
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for pf in tqdm(parquet_files, desc="Extracting parquet files"):
        df = pd.read_parquet(pf, columns=["text", "lang"])
        df = df[df["lang"] == "si"]           # keep Sinhala only
        for text in df["text"]:
            text = str(text).strip().replace("\r", "").replace("\n", " ")
            if text:
                out.write(text + "\n")
                total_rows += 1

print(f"\nDone.")
print(f"  Lines written : {total_rows:,}")
print(f"  Output file   : {OUTPUT_FILE}")
print(f"  File size     : {OUTPUT_FILE.stat().st_size / 1e9:.2f} GB")
