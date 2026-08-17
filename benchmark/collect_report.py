"""Collect every benchmark result into one comparison table.

Pulls from two snapshots that use different directory layouts and different
spellings of the same model, normalises them, and emits Markdown.

    python benchmark/collect_report.py                  # print tables
    python benchmark/collect_report.py -o docs/x.md     # write a report

Sources (relative to benchmark/results/):
  prior_2026-08-01/   the five earlier models, all tasks
  uc_instruct_2026-08-17/  today's run; *_raw dirs are raw-prompt scoring,
                           the plain dirs are chat-template scoring
"""
from __future__ import annotations

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PRIOR = os.path.join(RESULTS, "prior_2026-08-01")
NEW = os.path.join(RESULTS, "uc_instruct_2026-08-17")

# Display name -> the spellings each snapshot uses. The same checkpoint is
# called SinLlama_cpt_merged by the MMLU/PIQA runs and SinLlama_cpt by the
# downstream runs, and Bactrianx is spelled three different ways; without this
# map the two snapshots simply do not join.
MODELS = [
    ("Llama-3-8B",        ["llama-3-8b"]),
    ("SinLlama_v01",      ["SinLlama_v01"]),
    ("SinLlama_cpt",      ["SinLlama_cpt_merged", "SinLlama_cpt"]),
    ("SinLlama_v02",      ["SinLlama_v02"]),
    ("Bactrianx-Instruct", ["SinLlama_Backtrianx_instruct", "SinLlama_Bactrianx_Instruct"]),
    ("uc_instruct_cleaned", ["SinLlama_uc_instruct_cleaned"]),
]

MC_TASKS = [
    ("MMLU-Si",   "mmlu/SinhalaMMLU_results_final",             "mmlu/SinhalaMMLU_results_final_raw"),
    ("MMLU-En",   "mmlu/EnglishMMLU_results_final",             "mmlu/EnglishMMLU_results_final_raw"),
    ("PIQA-Si-p", "mrlbenchmarks/results_parallel_sinhala",     "mrlbenchmarks/results_parallel_sinhala_raw"),
    ("PIQA-En-p", "mrlbenchmarks/results_parallel_english",     "mrlbenchmarks/results_parallel_english_raw"),
    ("PIQA-Si-n", "mrlbenchmarks/results_nonparallel_sinhala",  "mrlbenchmarks/results_nonparallel_sinhala_raw"),
    ("PIQA-En-n", "mrlbenchmarks/results_nonparallel_english",  "mrlbenchmarks/results_nonparallel_english_raw"),
]

DOWN_TASKS = [("News", "news_lora", "results_news.txt"),
              ("Sentiment", "sentiment_lora", "results_sentiment.txt"),
              ("Writing", "writing_lora", "results_writing.txt")]


def mc_score(root, subdir, aliases):
    """(accuracy, n, format) for the first alias that has a metrics file."""
    for a in aliases:
        p = os.path.join(root, subdir, f"{a}_metrics.json")
        if os.path.exists(p):
            j = json.load(open(p))
            return j["overall"]["accuracy"], j["overall"]["total"], j.get("format", "raw")
    return None


def parse_downstream(path):
    """Pull the [TEST RESULTS] block out of a results_<task>.txt."""
    if not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    m = re.search(r"accuracy\s*:\s*([\d.]+)", txt)
    if m:
        out["acc"] = float(m.group(1)) * 100
    for key in ("precision", "recall", "f1"):
        m = re.search(rf"{key}\s*\([^)]*\)\s*:\s*([\d.]+)\s*/\s*([\d.]+)", txt)
        if m:
            out[f"{key}_micro"] = float(m.group(1)) * 100
            out[f"{key}_macro"] = float(m.group(2)) * 100
    return out or None


def down_score(disp, aliases, task_dir, fname):
    for root in (PRIOR, NEW):
        for a in aliases:
            for cand in (os.path.join(root, "downstream", "runs", task_dir, a, fname),
                         os.path.join(root, "downstream", task_dir, a, fname),
                         os.path.join(root, task_dir, a, fname)):
                r = parse_downstream(cand)
                if r:
                    return r
    return None


def fmt(v, w=7):
    return f"{v:{w}.2f}" if isinstance(v, (int, float)) else f"{'--':>{w}}"


def build():
    lines = []
    # ---- multiple choice -------------------------------------------------
    mc = {}
    for disp, aliases in MODELS:
        row = {}
        for task, sub, sub_raw in MC_TASKS:
            # today's models: prefer the raw arm so every column is raw-scored
            r = mc_score(NEW, sub_raw, aliases) or mc_score(PRIOR, sub, aliases)
            row[task] = r
        mc[disp] = row

    hdr = "| model | " + " | ".join(t for t, _, _ in MC_TASKS) + " |"
    lines.append(hdr)
    lines.append("|" + "---|" * (len(MC_TASKS) + 1))
    for disp, _ in MODELS:
        cells = []
        for task, _, _ in MC_TASKS:
            r = mc[disp][task]
            cells.append(f"{r[0]:.2f}" if r else "--")
        lines.append(f"| {disp} | " + " | ".join(cells) + " |")
    lines.append("")

    ns = {t: (mc["SinLlama_v02"][t][1] if mc["SinLlama_v02"][t] else "?") for t, _, _ in MC_TASKS}
    lines.append("Items per task: " + ", ".join(f"{t} n={ns[t]}" for t, _, _ in MC_TASKS))
    lines.append("")

    # ---- downstream ------------------------------------------------------
    down = {d: {t: down_score(d, a, td, fn) for t, td, fn in DOWN_TASKS}
            for d, a in MODELS}

    lines.append("| model | " + " | ".join(
        f"{t} acc | {t} F1" for t, _, _ in DOWN_TASKS) + " |")
    lines.append("|" + "---|" * (len(DOWN_TASKS) * 2 + 1))
    for disp, _ in MODELS:
        cells = []
        for task, _, _ in DOWN_TASKS:
            r = down[disp][task]
            cells.append(f"{r['acc']:.2f}" if r and "acc" in r else "--")
            cells.append(f"{r['f1_macro']:.2f}" if r and "f1_macro" in r else "--")
        lines.append(f"| {disp} | " + " | ".join(cells) + " |")

    return "\n".join(lines), mc, down


# Global-PIQA runs 92-95 items per variant, so one answer moves the score 1.1pp
# and re-running the same model has been observed to move it 4.21pp. Marking a
# "winner" in those columns would dress noise up as a result, so they are
# excluded from the bolding and flagged in the header instead.
NOISY = {"PIQA-Si-p", "PIQA-En-p", "PIQA-Si-n", "PIQA-En-n"}


def best_marked(rows, higher_is_better=True):
    """Bold the winning cell of each column. rows: {model: {col: value|None}}.

    A column is skipped entirely when the top score is tied — an arbitrary
    tie-break rendered in bold reads as a finding that is not there.
    """
    cols = {c for r in rows.values() for c in r}
    best = {}
    for c in cols:
        if c in NOISY:
            continue
        vals = [(m, r[c]) for m, r in rows.items() if r.get(c) is not None]
        if not vals:
            continue
        pick = (max if higher_is_better else min)(vals, key=lambda t: t[1])
        if sum(1 for _, v in vals if v == pick[1]) == 1:
            best[c] = pick[0]
    return best


def table(rows, cols, best, digits=2):
    out = ["| model | " + " | ".join(cols) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for m, r in rows.items():
        cells = []
        for c in cols:
            v = r.get(c)
            if v is None:
                cells.append("--")
            elif best.get(c) == m:
                cells.append(f"**{v:.{digits}f}**")
            else:
                cells.append(f"{v:.{digits}f}")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def report():
    _, mc, down = build()
    mc_rows = {d: {t: (mc[d][t][0] if mc[d][t] else None) for t, _, _ in MC_TASKS}
               for d, _ in MODELS}
    ns = {t: next((mc[d][t][1] for d, _ in MODELS if mc[d][t]), "?") for t, _, _ in MC_TASKS}

    down_acc = {d: {t: (down[d][t] or {}).get("acc") for t, _, _ in DOWN_TASKS}
                for d, _ in MODELS}
    down_f1 = {d: {t: (down[d][t] or {}).get("f1_macro") for t, _, _ in DOWN_TASKS}
               for d, _ in MODELS}

    # deltas against the finetuned base model
    base = MODELS[0][0]
    down_delta = {d: {t: (None if down_acc[d][t] is None or down_acc[base][t] is None
                          else down_acc[d][t] - down_acc[base][t])
                      for t, _, _ in DOWN_TASKS} for d, _ in MODELS}

    cols_mc = [t + (" †" if t in NOISY else "") for t, _, _ in MC_TASKS]
    mc_rows_lbl = {m: {c: r[c.removesuffix(" †")] for c in cols_mc} for m, r in mc_rows.items()}
    best_mc = {c: v for c, v in
               ((c, best_marked(mc_rows).get(c.removesuffix(" †"))) for c in cols_mc) if v}

    L = []
    L.append("## Zero-shot / few-shot knowledge and reasoning\n")
    L.append("Accuracy %, all raw-prompt scored, higher is better. "
             "**Bold** marks an outright column winner.\n")
    L.append(table(mc_rows_lbl, cols_mc, best_mc))
    L.append("")
    L.append("Items: " + ", ".join(f"`{t}` n={ns[t]}" for t, _, _ in MC_TASKS))
    L.append("")
    L.append("† Global-PIQA. At n=92–95 one item is 1.1pp and the same model re-run "
             "has moved 4.21pp, so these four columns are below the noise floor: no "
             "winner is marked and no gap in them should be read as a result.\n")

    L.append("## Downstream, after per-model LoRA finetuning\n")
    L.append("Test accuracy %, with the change against finetuned "
             f"{base} in brackets.\n")
    acc_lbl = {m: {t: down_acc[m][t] for t, _, _ in DOWN_TASKS} for m, _ in MODELS}
    rows_out = ["| model | " + " | ".join(t for t, _, _ in DOWN_TASKS) + " |",
                "|" + "---|" * (len(DOWN_TASKS) + 1)]
    bm = best_marked(acc_lbl)
    for m, _ in MODELS:
        cells = []
        for t, _, _ in DOWN_TASKS:
            v, dv = down_acc[m][t], down_delta[m][t]
            if v is None:
                cells.append("--")
                continue
            s = f"**{v:.2f}**" if bm.get(t) == m else f"{v:.2f}"
            if dv is not None and m != base:
                s += f" ({dv:+.2f})"
            cells.append(s)
        rows_out.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("\n".join(rows_out))
    L.append("\nMacro-F1 %.\n")
    L.append(table(down_f1, [t for t, _, _ in DOWN_TASKS], best_marked(down_f1)))
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out")
    ap.add_argument("--full", action="store_true", help="formatted report tables")
    a = ap.parse_args()
    text = report() if a.full else build()[0]
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)
        print(f"wrote {a.out}")
    else:
        print(text)
