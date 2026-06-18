#!/usr/bin/env python3
"""
JSONL Dataset Analyser
Analyses datasets with 'text', 'label', and 'label_name' fields.
Usage: python analyse_dataset.py <path_to_dataset.jsonl>
"""

import json
import sys
import os
from collections import Counter, defaultdict


def load_jsonl(filepath):
    records = []
    errors = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [!] Line {i}: JSON parse error — {e}")
                errors += 1
    return records, errors


def token_count(text):
    """Rough whitespace-based token count."""
    return len(text.split())


def char_count(text):
    return len(text)


def analyse(filepath):
    print(f"\n{'='*60}")
    print(f"  JSONL Dataset Analysis")
    print(f"  File: {filepath}")
    print(f"{'='*60}\n")

    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    file_size_kb = os.path.getsize(filepath) / 1024
    print(f"📁 File size: {file_size_kb:.1f} KB")

    records, errors = load_jsonl(filepath)
    total = len(records)
    print(f"📄 Total records loaded: {total}")
    if errors:
        print(f"⚠️  Parse errors: {errors}")

    if total == 0:
        print("\n[!] No records to analyse.")
        return

    # --- Field presence ---
    print(f"\n{'─'*40}")
    print("FIELD COVERAGE")
    print(f"{'─'*40}")
    fields = ["text", "label", "label_name"]
    for field in fields:
        present = sum(1 for r in records if field in r and r[field] is not None)
        pct = present / total * 100
        print(f"  {field:15s}: {present:>6} / {total}  ({pct:.1f}%)")

    # --- Label distribution ---
    print(f"\n{'─'*40}")
    print("LABEL DISTRIBUTION")
    print(f"{'─'*40}")
    label_counts = Counter()
    label_name_map = {}
    for r in records:
        label = r.get("label")
        label_name = r.get("label_name", "unknown")
        if label is not None:
            label_counts[label] += 1
            label_name_map[label] = label_name

    print(f"  Unique labels: {len(label_counts)}")
    print()
    print(f"  {'Label':>6}  {'Name':<20}  {'Count':>7}  {'%':>6}")
    print(f"  {'─'*6}  {'─'*20}  {'─'*7}  {'─'*6}")
    for label in sorted(label_counts):
        count = label_counts[label]
        name = label_name_map.get(label, "?")
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:>6}  {name:<20}  {count:>7}  {pct:>5.1f}%  {bar}")

    # --- Class balance ---
    counts = list(label_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
    print(f"\n  Most common:   {max_count} samples")
    print(f"  Least common:  {min_count} samples")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 5:
        print("  ⚠️  Significant class imbalance detected!")
    elif imbalance_ratio > 2:
        print("  ℹ️  Moderate class imbalance.")
    else:
        print("  ✅ Classes are roughly balanced.")

    # --- Text length stats ---
    print(f"\n{'─'*40}")
    print("TEXT LENGTH STATISTICS")
    print(f"{'─'*40}")
    texts = [r.get("text", "") for r in records if r.get("text")]
    chars = [char_count(t) for t in texts]
    tokens = [token_count(t) for t in texts]

    def stats(values, label):
        values.sort()
        n = len(values)
        mean = sum(values) / n
        median = values[n // 2]
        p10 = values[int(n * 0.10)]
        p90 = values[int(n * 0.90)]
        print(f"\n  {label}")
        print(f"    Min:    {min(values):>8}")
        print(f"    Max:    {max(values):>8}")
        print(f"    Mean:   {mean:>8.1f}")
        print(f"    Median: {median:>8}")
        print(f"    P10:    {p10:>8}")
        print(f"    P90:    {p90:>8}")

    stats(chars, "Character count")
    stats(tokens, "Token count (whitespace-split)")

    # --- Per-label text length ---
    print(f"\n{'─'*40}")
    print("AVG TEXT LENGTH PER LABEL")
    print(f"{'─'*40}")
    label_texts = defaultdict(list)
    for r in records:
        label = r.get("label")
        text = r.get("text", "")
        if label is not None and text:
            label_texts[label].append(token_count(text))

    print(f"  {'Label':>6}  {'Name':<20}  {'Avg Tokens':>10}  {'Avg Chars':>10}")
    print(f"  {'─'*6}  {'─'*20}  {'─'*10}  {'─'*10}")
    for label in sorted(label_texts):
        tkns = label_texts[label]
        avg_tkn = sum(tkns) / len(tkns)
        char_vals = [char_count(r.get("text","")) for r in records if r.get("label")==label and r.get("text")]
        avg_chr = sum(char_vals) / len(char_vals) if char_vals else 0
        name = label_name_map.get(label, "?")
        print(f"  {label:>6}  {name:<20}  {avg_tkn:>10.1f}  {avg_chr:>10.1f}")

    # --- Duplicate detection ---
    print(f"\n{'─'*40}")
    print("DUPLICATE DETECTION")
    print(f"{'─'*40}")
    text_counter = Counter(r.get("text", "") for r in records)
    exact_dupes = {t: c for t, c in text_counter.items() if c > 1}
    print(f"  Exact duplicate texts: {len(exact_dupes)}")
    if exact_dupes:
        total_dupe_records = sum(exact_dupes.values()) - len(exact_dupes)
        print(f"  Records that are duplicates: {total_dupe_records}")
        print("  Top 3 most duplicated:")
        for text, count in sorted(exact_dupes.items(), key=lambda x: -x[1])[:3]:
            preview = text[:60].replace("\n", " ")
            print(f"    [{count}x] \"{preview}...\"")

    # --- Empty / short text ---
    empty = sum(1 for r in records if not r.get("text", "").strip())
    short = sum(1 for r in records if 0 < token_count(r.get("text","")) < 5)
    print(f"\n  Empty texts:      {empty}")
    print(f"  Very short (<5 tokens): {short}")

    print(f"\n{'='*60}")
    print("  Analysis complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyse_dataset.py <path_to_dataset.jsonl>")
        print("\nExample:")
        print("  python analyse_dataset.py train.jsonl")
        sys.exit(1)

    analyse(sys.argv[1])