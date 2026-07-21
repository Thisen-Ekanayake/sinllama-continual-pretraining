#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build one top-level mrlbenchmarks/results.md summarizing all 4 Global-PIQA
benchmark runs (parallel/non-parallel x Sinhala/English) across all models,
by reading each benchmark's already-written <model>_metrics.json files.

Run after all 4 evaluate_piqa_*.py scripts have produced their per-model
metrics (locally, or after downloading them from the pod via run_pod_eval.sh).
"""
import os, json, argparse
from statistics import mean

import common as C

# (out-dir, display name) — Sinhala pair first, then the English pair.
BENCHMARKS = [
    ("results_parallel_sinhala", "Parallel (Sinhala)"),
    ("results_nonparallel_sinhala", "Non-parallel (Sinhala)"),
    ("results_parallel_english", "Parallel (English)"),
    ("results_nonparallel_english", "Non-parallel (English)"),
]


def load_all(root):
    """{out_dir: {model_name: metrics_dict}} for every benchmark dir that exists."""
    data = {}
    for out_dir, _ in BENCHMARKS:
        d = os.path.join(root, out_dir)
        if not os.path.isdir(d):
            continue
        models = {}
        for f in sorted(os.listdir(d)):
            if f.endswith("_metrics.json"):
                name = f[: -len("_metrics.json")]
                models[name] = json.load(open(os.path.join(d, f), encoding="utf-8"))
        if models:
            data[out_dir] = models
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="mrlbenchmarks/ directory")
    ap.add_argument("--out", default="results.md")
    args = ap.parse_args()

    data = load_all(args.root)
    present = [(d, label) for d, label in BENCHMARKS if d in data]
    if not present:
        raise SystemExit(f"no results_* directories with *_metrics.json found under {args.root}")

    all_models = sorted({m for models in data.values() for m in models})
    si_dirs = [d for d, _ in present if d.endswith("_sinhala")]
    en_dirs = [d for d, _ in present if d.endswith("_english")]

    def acc(d, name):
        return data[d].get(name, {}).get("overall", {}).get("accuracy")

    def quant_fmt(name):
        for d, _ in present:
            m = data[d].get(name)
            if m:
                q = C.QUANT_LABELS.get(m.get("quant"), m.get("quant", "?"))
                return q, m.get("format", "raw")
        return "?", "?"

    L = ["# Global-PIQA — Combined Results\n"]
    L.append("Overall accuracy across all 4 benchmark runs "
             "(parallel/non-parallel × Sinhala/English), one row per model.\n")

    header = ["Model"] + [label for _, label in present]
    if si_dirs:
        header.append("Sinhala avg")
    if en_dirs:
        header.append("English avg")
    header += ["Overall avg", "Quant", "Format"]

    rows = []
    for name in all_models:
        accs = [acc(d, name) for d, _ in present]
        row = [name] + [f"{a:.2f}%" if a is not None else "—" for a in accs]
        if si_dirs:
            si_vals = [acc(d, name) for d in si_dirs if acc(d, name) is not None]
            row.append(f"{mean(si_vals):.2f}%" if si_vals else "—")
        if en_dirs:
            en_vals = [acc(d, name) for d in en_dirs if acc(d, name) is not None]
            row.append(f"{mean(en_vals):.2f}%" if en_vals else "—")
        valid_accs = [a for a in accs if a is not None]
        row.append(f"{mean(valid_accs):.2f}%" if valid_accs else "—")
        q, fmt = quant_fmt(name)
        row += [q, fmt]
        rows.append(row)

    L.append(C.md_table(header, rows))

    for out_dir, label in present:
        L.append(f"\n## {label}\n")
        models = data[out_dir]
        sub_rows = [[name, f"{m['overall']['accuracy']:.2f}%",
                     m["overall"]["correct"], m["overall"]["total"]]
                    for name, m in sorted(models.items())]
        L.append(C.md_table(["Model", "Accuracy", "Correct", "Total"], sub_rows))
        L.append(f"\nFull detail: `{out_dir}/results.md`, `{out_dir}/<model>_results.txt`, "
                 f"`{out_dir}/<model>_predictions.csv`\n")

    out_path = os.path.join(args.root, args.out)
    open(out_path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
