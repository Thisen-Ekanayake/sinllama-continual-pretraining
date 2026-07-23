#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine the zero-shot and few-shot results of one attention implementation
(default: sdpa — the adopted setting) into a single report: per benchmark
(task × language) and per model, zero-shot vs few-shot accuracy with the delta,
plus both regimes' answer distributions.

Reads:  results/results_{zero,few}_shot_<attn>/<benchmark>/<model>_metrics.json
        (+ each <model>_predictions.csv for the answer distributions)
Writes: results/results_combined.md
"""
import os, csv, json, argparse
from statistics import mean

import common as C
from combine_all_results import read_label_counts

BENCHMARKS = [
    ("results_parallel_sinhala", "Parallel (Sinhala)"),
    ("results_nonparallel_sinhala", "Non-parallel (Sinhala)"),
    ("results_parallel_english", "Parallel (English)"),
    ("results_nonparallel_english", "Non-parallel (English)"),
]
MODEL_ORDER = ["llama-3-8b", "SinLlama_v01", "SinLlama_cpt", "SinLlama_Bactrianx_Instruct"]


def load(root, regime, attn):
    """{bench_dir: {model: metrics_dict}} for one condition folder."""
    cond = os.path.join(root, f"results_{regime}_{attn}")
    out = {}
    for bench, _ in BENCHMARKS:
        d = os.path.join(cond, bench)
        if not os.path.isdir(d):
            continue
        models = {}
        for f in sorted(os.listdir(d)):
            if f.endswith("_metrics.json"):
                name = f[: -len("_metrics.json")]
                models[name] = json.load(open(os.path.join(d, f), encoding="utf-8"))
        if models:
            out[bench] = models
    return out


def fmt_acc(o):
    return f"{o['accuracy']:.2f}% ({o['correct']}/{o['total']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", default="results_combined.md")
    args = ap.parse_args()

    zero = load(args.root, "zero_shot", args.attn)
    few = load(args.root, "few_shot", args.attn)
    if not zero or not few:
        raise SystemExit(f"missing results_zero_shot_{args.attn} or results_few_shot_{args.attn} under {args.root}")

    L = [f"# Global-PIQA — Combined Results ({args.attn}: zero-shot vs few-shot)\n"]
    L.append(f"Adopted setting: **{args.attn} attention, bf16**. Zero-shot scores every "
             f"question (103 parallel / 100 non-parallel); few-shot prepends **8 balanced "
             f"exemplars** (equal count per answer key, shuffled) and excludes them from "
             f"the test set (95/92 test questions). Δ is **few-shot − zero-shot** in "
             f"percentage points.\n")

    # Each accuracy table is also written as a bare CSV next to the report:
    # columns model, zero_shot_accuracy, few_shot_accuracy (percent, 2 dp).
    csv_tables = {}                     # csv filename -> [(model, zero%, few%)]

    # ---- overall averages across the 4 benchmarks ---- #
    names = [n for n in MODEL_ORDER
             if all(n in zero.get(b, {}) and n in few.get(b, {}) for b, _ in BENCHMARKS)]
    L.append("## Overall accuracy (average across the 4 benchmarks)\n")
    rows = []
    for n in names:
        z = mean(zero[b][n]["overall"]["accuracy"] for b, _ in BENCHMARKS)
        f = mean(few[b][n]["overall"]["accuracy"] for b, _ in BENCHMARKS)
        fmt = few[BENCHMARKS[0][0]][n].get("format", "raw")
        rows.append([n, f"{z:.2f}%", f"{f:.2f}%", f"{f - z:+.2f} pp", fmt])
        csv_tables.setdefault("combined_overall.csv", []).append((n, z, f))
    L.append(C.md_table(["Model", "Zero-shot avg", "Few-shot avg", "Δ", "Format"], rows))

    # ---- per benchmark ---- #
    for bench, label in BENCHMARKS:
        L.append(f"\n## {label}\n")
        rows = []
        csv_name = "combined_" + bench.replace("results_", "") + ".csv"
        for n in names:
            z, f = zero[bench][n]["overall"], few[bench][n]["overall"]
            rows.append([n, fmt_acc(z), fmt_acc(f), f"{f['accuracy'] - z['accuracy']:+.2f} pp"])
            csv_tables.setdefault(csv_name, []).append((n, z["accuracy"], f["accuracy"]))
        L.append(C.md_table(["Model", "Zero-shot", "Few-shot", "Δ"], rows))
        L.append(f"\nFull detail: `results_zero_shot_{args.attn}/{bench}/` and "
                 f"`results_few_shot_{args.attn}/{bench}/`\n")

    # ---- answer distributions, both regimes ---- #
    L.append("\n## Answer distributions (predicted vs actual)\n")
    L.append("Counts of each predicted option per model and regime, against each "
             "regime's actual (gold) distribution (they differ because few-shot "
             "excludes the 8 exemplars). Predictions piling onto one option = "
             "position-bias collapse.\n")
    for bench, label in BENCHMARKS:
        dists, gold_by_regime = [], {}
        per_regime = {}                 # regime_key -> {model: pred Counter}
        for regime_key, regime_label, cond_data in [("zero_shot", "zero-shot", zero),
                                                    ("few_shot", "few-shot", few)]:
            cond_dir = f"results_{regime_key}_{args.attn}"
            for n in names:
                path = os.path.join(args.root, cond_dir, bench, f"{n}_predictions.csv")
                if not os.path.exists(path):
                    continue
                pred_c, gold_c = read_label_counts(path)
                o = cond_data[bench][n]["overall"]
                if pred_c is None or pred_c["n_correct"] != o["correct"] \
                        or pred_c["n_rows"] != o["total"]:
                    print(f"  WARNING: label recovery mismatch for {path} — skipped")
                    continue
                dists.append((f"{n} ({regime_label})", pred_c["pred"]))
                per_regime.setdefault(regime_key, {})[n] = pred_c["pred"]
                gold_by_regime[regime_key] = gold_c
        if not dists:
            continue
        labels = sorted({k for _, d in dists for k in d}
                        | {k for g in gold_by_regime.values() for k in g})
        L.append(f"\n### {label}\n")
        rows = [[name] + [d.get(k, 0) for k in labels] for name, d in dists]
        for regime_key, g in gold_by_regime.items():
            rows.append([f"**Actual gold ({regime_key.replace('_', '-')})**"]
                        + [g.get(k, 0) for k in labels])
        L.append(C.md_table([""] + labels, rows))

        # grouped-bar-chart CSVs: one per benchmark × regime; rows = answer
        # options, columns = the 4 models' predicted counts + that regime's gold
        for regime_key, models in per_regime.items():
            cpath = os.path.join(args.root,
                                 f"distribution_{bench.replace('results_', '')}_{regime_key}.csv")
            with open(cpath, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["answer"] + names + ["actual_gold"])
                for k in labels:
                    w.writerow([k] + [models[n].get(k, 0) for n in names]
                               + [gold_by_regime[regime_key].get(k, 0)])
            print(f"Wrote {cpath}")

    out_path = os.path.join(args.root, args.out)
    open(out_path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"Wrote {out_path}")

    for csv_name, table in csv_tables.items():
        cpath = os.path.join(args.root, csv_name)
        with open(cpath, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "zero_shot_accuracy", "few_shot_accuracy"])
            for n, z, f in table:
                w.writerow([n, f"{z:.2f}", f"{f:.2f}"])
        print(f"Wrote {cpath}")


if __name__ == "__main__":
    main()
