#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the English-prompt vs Sinhala-prompt evaluation results.

Both results/english_prompt/ and results/sinhala_prompt/ hold the same-schema
combined CSVs (same 4 benchmarks x 4 models x zero/few-shot, sdpa+bf16); they
differ only in the language of the instruction/prompt template. This script
lines them up and reports, per model / benchmark / regime, the accuracy under
each prompt language and the delta (Sinhala prompt - English prompt).

Reads:  results/english_prompt/combined_{overall,parallel_english,parallel_sinhala,
                                          nonparallel_english,nonparallel_sinhala}.csv
        results/english_prompt/distribution_<benchmark>_<regime>.csv
        results/sinhala_prompt/{combined,distribution}_*.csv    (same filenames)
Writes: results/prompt_comparison/comparison_overall.csv       (wide, per model)
        results/prompt_comparison/comparison_by_benchmark.csv   (tidy long)
        results/prompt_comparison/distribution_comparison.csv   (tidy long)
"""
import os, csv, argparse

# (source filename stem, benchmark label used in the tidy output)
BENCHMARKS = [
    ("combined_overall", "overall"),
    ("combined_parallel_sinhala", "parallel_sinhala"),
    ("combined_nonparallel_sinhala", "nonparallel_sinhala"),
    ("combined_parallel_english", "parallel_english"),
    ("combined_nonparallel_english", "nonparallel_english"),
]
# benchmarks that have per-option answer distributions (overall has none)
DIST_BENCHMARKS = ["parallel_sinhala", "nonparallel_sinhala",
                   "parallel_english", "nonparallel_english"]
MODEL_ORDER = ["llama-3-8b", "SinLlama_v01", "SinLlama_cpt", "SinLlama_Bactrianx_Instruct"]
# distribution columns compared per cell (models + the gold reference)
DIST_SERIES = MODEL_ORDER + ["actual_gold"]
REGIMES = [("zero_shot", "zero_shot_accuracy"), ("few_shot", "few_shot_accuracy")]


def load(folder, stem):
    """{model: {'zero_shot_accuracy': float, 'few_shot_accuracy': float}}"""
    path = os.path.join(folder, stem + ".csv")
    rows = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["model"]] = {
                "zero_shot_accuracy": float(r["zero_shot_accuracy"]),
                "few_shot_accuracy": float(r["few_shot_accuracy"]),
            }
    return rows


def load_dist(folder, benchmark, regime):
    """{answer: {series: int}} for one distribution CSV, or {} if missing."""
    path = os.path.join(folder, f"distribution_{benchmark}_{regime}.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["answer"]] = {s: int(r[s]) for s in DIST_SERIES}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--en", default="english_prompt")
    ap.add_argument("--si", default="sinhala_prompt")
    ap.add_argument("--out", default="prompt_comparison")
    args = ap.parse_args()

    en_dir = os.path.join(args.root, args.en)
    si_dir = os.path.join(args.root, args.si)
    out_dir = os.path.join(args.root, args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Load every benchmark from both prompt-language folders.
    en = {stem: load(en_dir, stem) for stem, _ in BENCHMARKS}
    si = {stem: load(si_dir, stem) for stem, _ in BENCHMARKS}

    # ---- 1) tidy long: benchmark x model x regime ---------------------------
    long_path = os.path.join(out_dir, "comparison_by_benchmark.csv")
    with open(long_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "model", "regime",
                    "english_prompt_acc", "sinhala_prompt_acc", "delta_si_minus_en"])
        for stem, label in BENCHMARKS:
            for model in MODEL_ORDER:
                for regime, col in REGIMES:
                    e = en[stem][model][col]
                    s = si[stem][model][col]
                    w.writerow([label, model, regime,
                                f"{e:.2f}", f"{s:.2f}", f"{s - e:+.2f}"])
    print("wrote", long_path)

    # ---- 2) wide overall: one row per model, both regimes -------------------
    over_path = os.path.join(out_dir, "comparison_overall.csv")
    with open(over_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model",
                    "en_zero_shot", "si_zero_shot", "delta_zero_shot",
                    "en_few_shot", "si_few_shot", "delta_few_shot"])
        for model in MODEL_ORDER:
            ez = en["combined_overall"][model]["zero_shot_accuracy"]
            sz = si["combined_overall"][model]["zero_shot_accuracy"]
            ef = en["combined_overall"][model]["few_shot_accuracy"]
            sf = si["combined_overall"][model]["few_shot_accuracy"]
            w.writerow([model,
                        f"{ez:.2f}", f"{sz:.2f}", f"{sz - ez:+.2f}",
                        f"{ef:.2f}", f"{sf:.2f}", f"{sf - ef:+.2f}"])
    print("wrote", over_path)

    # ---- 3) tidy long: predicted-answer distribution, per cell --------------
    # benchmark x regime x answer-option x series -> EN count, SI count, delta.
    # actual_gold rows should always show delta 0 (same gold set) -> sanity check.
    dist_path = os.path.join(out_dir, "distribution_comparison.csv")
    with open(dist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "regime", "answer", "series",
                    "english_prompt_count", "sinhala_prompt_count", "delta_si_minus_en"])
        for bench in DIST_BENCHMARKS:
            for regime, _ in REGIMES:
                ed = load_dist(en_dir, bench, regime)
                sd = load_dist(si_dir, bench, regime)
                answers = sorted(set(ed) | set(sd))
                for ans in answers:
                    for series in DIST_SERIES:
                        e = ed.get(ans, {}).get(series, 0)
                        s = sd.get(ans, {}).get(series, 0)
                        w.writerow([bench, regime, ans, series, e, s, f"{s - e:+d}"])
    print("wrote", dist_path)


if __name__ == "__main__":
    main()
