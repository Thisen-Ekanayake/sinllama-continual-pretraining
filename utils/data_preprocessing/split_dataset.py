#!/usr/bin/env python3
"""
Stratified JSONL Splitter
Splits a JSONL dataset into train/val/test sets while preserving label distribution.
Usage: python split_dataset.py <input.jsonl> [--train 0.8] [--val 0.1] [--test 0.1] [--seed 42]
"""

import json
import sys
import os
import random
import argparse
from collections import defaultdict, Counter
from pathlib import Path


def load_jsonl(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_distribution(records, label_name_map, title, total_original):
    counts = Counter(r["label"] for r in records)
    n = len(records)
    print(f"\n  {title} — {n} records ({n/total_original*100:.1f}% of total)")
    print(f"  {'Label':>6}  {'Name':<22}  {'Count':>6}  {'Split%':>7}  {'OrigDist%':>9}")
    print(f"  {'─'*6}  {'─'*22}  {'─'*6}  {'─'*7}  {'─'*9}")
    for label in sorted(counts):
        name = label_name_map.get(label, "?")
        count = counts[label]
        split_pct = count / n * 100
        orig_pct = count / total_original * 100  # compared to full dataset per-label
        print(f"  {label:>6}  {name:<22}  {count:>6}  {split_pct:>6.1f}%  {orig_pct:>8.1f}%")


def stratified_split(records, train_ratio, val_ratio, test_ratio, seed):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    rng = random.Random(seed)

    # Group by label
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    train, val, test = [], [], []

    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        n = len(items)

        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            print(f"  [!] Label {label} has too few samples ({n}) for a 3-way split. Adjusting.")
            n_train = 1
            n_val = max(1, (n - 1) // 2)
            n_test = n - n_train - n_val

        train += items[:n_train]
        val   += items[n_train:n_train + n_val]
        test  += items[n_train + n_val:]

    # Shuffle each split so labels aren't grouped
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Stratified JSONL train/val/test splitter")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument("--val",   type=float, default=0.1, help="Val ratio (default: 0.1)")
    parser.add_argument("--test",  type=float, default=0.1, help="Test ratio (default: 0.1)")
    parser.add_argument("--seed",  type=int,   default=42,  help="Random seed (default: 42)")
    parser.add_argument("--outdir", default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        print(f"[ERROR] Ratios must sum to 1.0 (got {args.train + args.val + args.test:.3f})")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    outdir = Path(args.outdir) if args.outdir else input_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    train_path = outdir / f"{stem}_train.jsonl"
    val_path   = outdir / f"{stem}_val.jsonl"
    test_path  = outdir / f"{stem}_test.jsonl"

    print(f"\n{'='*60}")
    print(f"  Stratified JSONL Splitter")
    print(f"{'='*60}")
    print(f"  Input  : {input_path}")
    print(f"  Split  : train={args.train:.0%}  val={args.val:.0%}  test={args.test:.0%}")
    print(f"  Seed   : {args.seed}")
    print(f"  OutDir : {outdir}")

    records = load_jsonl(input_path)
    total = len(records)
    print(f"  Loaded : {total} records")

    label_name_map = {r["label"]: r.get("label_name", str(r["label"])) for r in records}

    train, val, test = stratified_split(records, args.train, args.val, args.test, args.seed)

    write_jsonl(train, train_path)
    write_jsonl(val,   val_path)
    write_jsonl(test,  test_path)

    # --- Summary ---
    print(f"\n{'─'*60}")
    print("SPLIT SUMMARY")
    print_distribution(train, label_name_map, "TRAIN", total)
    print_distribution(val,   label_name_map, "VAL",   total)
    print_distribution(test,  label_name_map, "TEST",  total)

    print(f"\n{'─'*60}")
    print("OUTPUT FILES")
    for path, split in [(train_path, train), (val_path, val), (test_path, test)]:
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path.name:<40} {len(split):>5} records  {size_kb:>7.1f} KB")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()