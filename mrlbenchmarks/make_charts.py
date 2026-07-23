#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render one chart per table in results/results_combined.md (sdpa: zero-shot vs
few-shot) into results/charts/:

  overall_accuracy.png            — grouped bars, avg accuracy across benchmarks
  acc_<benchmark>.png   (x4)      — grouped bars, zero vs few-shot per model
  dist_<benchmark>.png  (x4)      — annotated heatmap of predicted-answer counts
                                    per model x regime, vs the gold distribution

Data comes from the same source of truth as the report (metrics.json +
predictions CSVs), not from parsing the markdown.
"""
import os, argparse
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns

from combine_all_results import read_label_counts
from combine_regimes import BENCHMARKS, MODEL_ORDER, load

# ---- palette (validated categorical slots + chrome, from the dataviz system) - #
BLUE, ORANGE = "#2a78d6", "#eb6834"          # zero-shot / few-shot series
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

SHORT = {"llama-3-8b": "llama-3-8b", "SinLlama_v01": "v01",
         "SinLlama_cpt": "cpt", "SinLlama_Bactrianx_Instruct": "Bactrianx"}

sns.set_theme(style="white", rc={
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "font.family": "sans-serif",
})


def style_ax(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def grouped_bars(path, title, subtitle, names, zero_vals, few_vals, chance=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    x = np.arange(len(names))
    w = 0.32
    b1 = ax.bar(x - w / 2, zero_vals, w, color=BLUE, label="Zero-shot", zorder=3)
    b2 = ax.bar(x + w / 2, few_vals, w, color=ORANGE, label="Few-shot (8, balanced)", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            # surface-colored bbox so labels stay legible where the chance
            # line or a gridline passes behind them
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, fontweight="bold", color=INK2,
                        zorder=4, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6))
    if chance is not None:
        ax.axhline(chance, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=2)
        ax.set_xlim(-0.5, len(names) - 0.5 + 0.72)   # right margin for the label
        ax.annotate(f"chance {chance:.0f}%", (len(names) - 0.5 + 0.68, chance),
                    ha="right", va="center", fontsize=8, color=MUTED, zorder=4,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))
    ax.set_xticks(x, [SHORT[n] for n in names], fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)], fontsize=8.5)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=26, color=INK)
    ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def dist_heatmap(path, title, subtitle, row_labels, matrix, col_labels, n_block):
    """Annotated count heatmap; horizontal separators between the zero-shot
    block, few-shot block, and the gold rows. Title/subtitle sit at figure
    level so long titles don't overhang narrow (2-column) heatmaps."""
    w = max(6.0, 1.9 + 1.35 * len(col_labels))
    h = 0.52 * len(row_labels) + 1.7
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    cmap = LinearSegmentedColormap.from_list("blues", ["#f2f7fd"] + BLUE_RAMP)
    sns.heatmap(np.array(matrix), annot=True, fmt="d", cmap=cmap, cbar=False,
                linewidths=2, linecolor=SURFACE,
                xticklabels=col_labels, yticklabels=row_labels,
                annot_kws={"fontsize": 9.5}, ax=ax)
    for y in (n_block, 2 * n_block):                 # block separators
        ax.axhline(y, color=INK2, linewidth=1.4)
    ax.tick_params(length=0)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9, color=INK2)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=10, color=INK2)
    top_pad = 1.0 / h                                 # ~1 inch for title block
    fig.subplots_adjust(top=1 - top_pad, bottom=0.45 / h, left=0.30, right=0.97)
    fig.text(0.02, 1 - 0.32 / h, title, fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.02, 1 - 0.60 / h, subtitle, fontsize=9, color=MUTED)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out-dir", default="charts")
    args = ap.parse_args()

    zero = load(args.root, "zero_shot", args.attn)
    few = load(args.root, "few_shot", args.attn)
    out = os.path.join(args.root, args.out_dir)
    os.makedirs(out, exist_ok=True)
    names = [n for n in MODEL_ORDER
             if all(n in zero.get(b, {}) and n in few.get(b, {}) for b, _ in BENCHMARKS)]

    # 1) overall averages
    z_avg = [mean(zero[b][n]["overall"]["accuracy"] for b, _ in BENCHMARKS) for n in names]
    f_avg = [mean(few[b][n]["overall"]["accuracy"] for b, _ in BENCHMARKS) for n in names]
    grouped_bars(os.path.join(out, "overall_accuracy.png"),
                 "Global-PIQA — overall accuracy",
                 f"Average across the 4 benchmarks · {args.attn} · bf16",
                 names, z_avg, f_avg)

    # 2) per benchmark
    for bench, label in BENCHMARKS:
        z = [zero[bench][n]["overall"]["accuracy"] for n in names]
        f = [few[bench][n]["overall"]["accuracy"] for n in names]
        chance = 50.0 if "nonparallel" in bench else 25.0    # 2-way vs 4-way MCQ
        n_z = zero[bench][names[0]]["overall"]["total"]
        n_f = few[bench][names[0]]["overall"]["total"]
        grouped_bars(os.path.join(out, f"acc_{bench.replace('results_', '')}.png"),
                     f"{label} — accuracy",
                     f"Zero-shot n={n_z} · few-shot n={n_f} (8 exemplars held out) · {args.attn} · bf16",
                     names, z, f, chance=chance)

    # 3) answer-distribution heatmaps
    for bench, label in BENCHMARKS:
        rows, labels_set = [], set()
        for regime, cond in [("zero-shot", zero), ("few-shot", few)]:
            cond_dir = f"results_{'zero_shot' if regime == 'zero-shot' else 'few_shot'}_{args.attn}"
            for n in names:
                pred_c, gold_c = read_label_counts(
                    os.path.join(args.root, cond_dir, bench, f"{n}_predictions.csv"))
                o = cond[bench][n]["overall"]
                assert pred_c and pred_c["n_correct"] == o["correct"] \
                       and pred_c["n_rows"] == o["total"], f"label recovery mismatch: {bench}/{n}"
                rows.append((f"{SHORT[n]} ({regime})", pred_c["pred"]))
                labels_set |= set(pred_c["pred"]) | set(gold_c)
                if n == names[-1]:
                    rows_gold = (f"gold ({regime})", gold_c)
            rows.append(rows_gold)
        # reorder: zero block, few block, then the two gold rows
        model_rows = [r for r in rows if not r[0].startswith("gold")]
        gold_rows = [r for r in rows if r[0].startswith("gold")]
        ordered = model_rows + gold_rows
        cols = sorted(labels_set)
        matrix = [[d.get(c, 0) for c in cols] for _, d in ordered]
        dist_heatmap(os.path.join(out, f"dist_{bench.replace('results_', '')}.png"),
                     f"{label} — answer distribution",
                     "Predicted-option counts per model & regime vs actual (gold)",
                     [r[0] for r in ordered], matrix, cols, n_block=len(names))


if __name__ == "__main__":
    main()
