#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build one top-level mrlbenchmarks/results.md summarizing all 4 Global-PIQA
benchmark runs (parallel/non-parallel x Sinhala/English) across all models,
by reading each benchmark's already-written <model>_metrics.json files.

Run after all 4 evaluate_piqa_*.py scripts have produced their per-model
metrics (locally, or after downloading them from the pod via run_pod_eval.sh).
"""
import os, csv, json, argparse
from collections import Counter
from statistics import mean

import benchmark.mrlbenchmarks.common as C

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


LABELS = {"1", "2", "3", "4", "A", "B", "C", "D", "?"}


def read_label_counts(path):
    """Recover (pred_counts, gold_counts) from a <model>_predictions.csv.

    Some result CSVs were later pretty-printed with space-padding, which
    dropped the quoting around comma-containing text fields — so column
    positions (from either end) are unreliable. Labels are recovered by
    scanning each row's tail right-to-left: fields -1/-2 are valid/correct
    (True/False), then the first plausible label token leftward of them is
    pred_label and the next one further left is gold_label. The caller MUST
    validate the recovery against metrics.json (correct/total must match).
    Returns (None, None) if any row can't be parsed."""
    pred, gold = Counter(), Counter()
    n_rows = n_correct = 0
    with open(path, encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r)                                        # header
        for row in r:
            if len(row) < 6:
                continue
            tail = [x.strip() for x in row]
            if tail[-1] not in ("True", "False") or tail[-2] not in ("True", "False"):
                return None, None
            p = g = None
            for x in reversed(tail[:-2]):              # rightmost label = pred,
                if x in LABELS:                        # next one left = gold
                    if p is None:
                        p = x
                    else:
                        g = x
                        break
            if p is None or g is None:
                return None, None
            pred[p] += 1
            gold[g] += 1
            n_rows += 1
            n_correct += int(p == g)
    return dict(pred=pred, n_rows=n_rows, n_correct=n_correct), gold


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

    # ---- answer distributions: predicted counts per model + actual (gold) ---- #
    L.append("\n## Answer distributions (predicted vs actual)\n")
    L.append("Per benchmark: how many times each model predicted each answer "
             "option, against the actual (gold) distribution. A model whose "
             "predictions pile onto one option has collapsed to position bias.\n")
    for out_dir, label in present:
        dists, golds = {}, {}
        for name in sorted(data[out_dir]):
            path = os.path.join(args.root, out_dir, f"{name}_predictions.csv")
            if not os.path.exists(path):
                continue
            pred_c, gold_c = read_label_counts(path)
            if pred_c is None:
                print(f"  WARNING: could not recover labels from {path} — skipped")
                continue
            # end-to-end check: recovered labels must reproduce the metrics
            o = data[out_dir][name]["overall"]
            if pred_c["n_correct"] != o["correct"] or pred_c["n_rows"] != o["total"]:
                print(f"  WARNING: {path} label recovery mismatches metrics.json "
                      f"({pred_c['n_correct']}/{pred_c['n_rows']} vs "
                      f"{o['correct']}/{o['total']}) — skipped")
                continue
            dists[name] = pred_c["pred"]
            golds[name] = gold_c
        if not dists:
            continue
        gold = golds[next(iter(golds))]
        if any(g != gold for g in golds.values()):     # should never differ
            print(f"  WARNING: gold distributions differ across models in {out_dir}")
        labels = sorted(set(gold) | {k for d in dists.values() for k in d})
        L.append(f"\n### {label}\n")
        rows = [[name + " (predicted)"] + [d.get(k, 0) for k in labels]
                for name, d in dists.items()]
        rows.append(["**Actual (gold)**"] + [gold.get(k, 0) for k in labels])
        L.append(C.md_table([""] + labels, rows))

    out_path = os.path.join(args.root, args.out)
    open(out_path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
