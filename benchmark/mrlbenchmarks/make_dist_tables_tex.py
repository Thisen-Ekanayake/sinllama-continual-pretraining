#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render the 8 predicted-answer distribution tables (task x content-language x
regime) as a compilable LaTeX file, from the prompt comparison CSV.

Each table lists, per model, the predicted-answer count vector under the
English prompt vs the Sinhala prompt, plus the per-option delta (Sinhala -
English). The regime's gold vector goes in the caption.

Reads:  results/prompt_comparison/distribution_comparison.csv
Writes: results/prompt_comparison/distribution_tables.tex
"""
import os, csv
from collections import defaultdict, OrderedDict

SRC = "results/prompt_comparison/distribution_comparison.csv"
OUT = "results/prompt_comparison/distribution_tables.tex"

MODELS = ["llama-3-8b", "SinLlama_v01", "SinLlama_cpt", "SinLlama_Bactrianx_Instruct"]
BENCH = OrderedDict([
    ("parallel_sinhala",    ("Parallel",     "Sinhala")),
    ("nonparallel_sinhala", ("Non-parallel", "Sinhala")),
    ("parallel_english",    ("Parallel",     "English")),
    ("nonparallel_english", ("Non-parallel", "English")),
])
REG = [("zero_shot", "Zero-shot"), ("few_shot", "Few-shot")]


def tex_escape(s):
    return s.replace("_", r"\_")


def load():
    raw = list(csv.reader(open(SRC, encoding="utf-8-sig")))
    hdr = [h.strip() for h in raw[0]]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    answers = defaultdict(list)
    for row in raw[1:]:
        if not row:
            continue
        d = {k: v.strip() for k, v in zip(hdr, row)}
        b, reg, ans, s = d["benchmark"], d["regime"], d["answer"], d["series"]
        data[b][reg][s][ans] = (int(d["english_prompt_count"]),
                                int(d["sinhala_prompt_count"]))
        if ans not in answers[b]:
            answers[b].append(ans)
    return data, answers


def main():
    data, answers = load()
    L = []
    ap = L.append
    ap(r"\documentclass{article}")
    ap(r"\usepackage[margin=1in]{geometry}")
    ap(r"\usepackage{booktabs}")
    ap(r"\usepackage{float}")  # provides [H] to pin tables in source order
    ap(r"\begin{document}")
    ap("")
    ap(r"\section*{Predicted-answer distribution: English prompt vs.\ Sinhala prompt}")
    ap(r"Each cell is the vector of predicted-answer counts across the option order "
       r"shown in the caption. $\Delta =$ Sinhala prompt $-$ English prompt.")
    ap("")

    for b, (task, lang) in BENCH.items():
        opts = answers[b]
        order = "/".join(opts)
        for reg, reg_lbl in REG:
            gold = data[b][reg]["actual_gold"]
            goldv = "/".join(str(gold[o][0]) for o in opts)
            ap(r"\begin{table}[H]")
            ap(r"\centering")
            ap(r"\begin{tabular}{llll}")
            ap(r"\toprule")
            ap(r"Model & EN prompt & SI prompt & $\Delta$ (SI$-$EN) \\")
            ap(r"\midrule")
            for m in MODELS:
                d = data[b][reg][m]
                env = "/".join(str(d[o][0]) for o in opts)
                siv = "/".join(str(d[o][1]) for o in opts)
                dlt = "/".join(f"{d[o][1] - d[o][0]:+d}" for o in opts)
                ap(f"{tex_escape(m)} & \\texttt{{{env}}} & \\texttt{{{siv}}} "
                   f"& \\texttt{{{dlt}}} \\\\")
            ap(r"\bottomrule")
            ap(r"\end{tabular}")
            cap = (f"{task} $\\cdot$ {lang} content, {reg_lbl.lower()}: "
                   f"predicted-answer distribution under the English vs.\\ Sinhala "
                   f"prompt (option order {order}; gold $=$ {goldv}).")
            lbl = f"tab:dist_{b}_{reg}"
            ap(f"\\caption{{{cap}}}")
            ap(f"\\label{{{lbl}}}")
            ap(r"\end{table}")
            ap("")

    ap(r"\end{document}")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", OUT, f"({len(BENCH) * len(REG)} tables)")


if __name__ == "__main__":
    main()
