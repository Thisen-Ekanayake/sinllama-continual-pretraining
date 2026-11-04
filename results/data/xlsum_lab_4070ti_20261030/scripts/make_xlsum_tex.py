"""Build results/latex/XLSum_Summarization.tex from the score JSONs in
results/data/xlsum_lab_4070ti_20261030/ (no numbers typed by hand).
Usage: python make_xlsum_tex.py   (then pdflatex in results/latex)
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1]
OUT = D.parents[1] / "latex" / "XLSum_Summarization.tex"
MODELS = ["llama-3-8b", "SinLlama_v01", "SinLlama_wiki_uc_gen_mix", "SinLlama_textbook_mix"]
NAME = {"llama-3-8b": "llama-3-8b (base)", "SinLlama_v01": "SinLlama\\_v01",
        "SinLlama_wiki_uc_gen_mix": "wiki\\_uc\\_gen\\_mix", "SinLlama_textbook_mix": "textbook\\_mix"}
SHORT = {"llama-3-8b": "llama-3-8b", "SinLlama_v01": "v01",
         "SinLlama_wiki_uc_gen_mix": "wiki\\_uc\\_gen\\_mix", "SinLlama_textbook_mix": "textbook\\_mix"}
# XL-Sum paper (fine-tuned multilingual mT5-base, full test set), /ml/xl-sum/README.md
MT5 = {"sinhala": (27.29, 13.38, 23.47), "english": (37.60, 15.15, 29.88)}
BS = {"sinhala": ("bertscore_helabert.json", "L9"), "english": ("bertscore_roberta_large.json", "L17")}
STEPS = [("SinLlama_v01", "llama-3-8b"), ("SinLlama_wiki_uc_gen_mix", "SinLlama_v01"),
         ("SinLlama_textbook_mix", "SinLlama_wiki_uc_gen_mix"), ("SinLlama_textbook_mix", "llama-3-8b")]
RK = ["rouge1", "rouge2", "rougeL"]


def load(lang):
    rouge = json.load(open(D / lang / "rouge.json"))
    bs_file, layer = BS[lang]
    bs = json.load(open(D / lang / bs_file))
    r = {t["model"]: t for t in rouge["scores"]}
    b = {t["model"]: t for t in bs["scores"]}
    rp = {(p["a"], p["b"]): p for p in rouge["paired_bootstrap"]}
    bp = {(p["a"], p["b"]): p[f"{layer}_F1"] for p in bs["paired_bootstrap"]}
    return r, b, rp, bp, layer


def main_table(lang, caption, label):
    r, b, _, _, layer = load(lang)
    cols = [[r[m][k] for m in MODELS] for k in RK] + [[b[m][f"{layer}_{k}"] for m in MODELS] for k in ("P", "R", "F1")]
    best = [max(c) for c in cols]
    rows = []
    for i, m in enumerate(MODELS):
        cells = [f"\\textbf{{{c[i]:.2f}}}" if c[i] == bst else f"{c[i]:.2f}" for c, bst in zip(cols, best)]
        rows.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
    mt5 = " & ".join(f"\\textit{{{v:.2f}}}" for v in MT5[lang])
    return f"""\\begin{{table}}[H]
\\centering\\small
\\begin{{tabular}}{{lcccccc}}
\\toprule
& \\multicolumn{{3}}{{c}}{{ROUGE (F1)}} & \\multicolumn{{3}}{{c}}{{BERTScore}} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}
Model & R-1 & R-2 & R-L & P & R & F1 \\\\
\\midrule
{chr(10).join(rows)}
\\midrule
\\textit{{XL-Sum mT5 (paper)}} & {mt5} & -- & -- & -- \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table}}
"""


def fmt(d, p):
    ps = "$\\le10^{-4}$" if p <= 1e-4 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
    sign = "$+$" if d >= 0 else "$-$"
    return f"{sign}{abs(d):.2f} ({ps})"


def steps_table():
    blocks = []
    for lang, title in (("sinhala", "Sinhala"), ("english", "English")):
        _, _, rp, bp, _ = load(lang)
        rows = [f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        for a, bm in STEPS:
            cells = [fmt(rp[(a, bm)][k]["delta"], rp[(a, bm)][k]["p"]) for k in RK]
            cells.append(fmt(bp[(a, bm)]["delta"], bp[(a, bm)]["p"]))
            rows.append(f"{SHORT[bm]} $\\rightarrow$ {SHORT[a]} & " + " & ".join(cells) + " \\\\")
        blocks.append("\n".join(rows))
    body = "\n\\midrule\n".join(blocks)
    return f"""\\begin{{table}}[H]
\\centering\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Step & R-1 & R-2 & R-L & BERTScore F1 \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{XL-Sum, change between models: $\\Delta$ points (paired bootstrap $p$, 10{{,}}000 resamples).}}
\\label{{tab:xlsum_steps}}
\\end{{table}}
"""


si_table = main_table("sinhala",
            "XL-Sum v2.0 Sinhala test ($n$=500). Each model QLoRA-fine-tuned on the 3{,}249 training articles "
            "(best checkpoint by val loss), beam 4. ROUGE: XL-Sum multilingual scorer (no Sinhala stemmer). "
            "BERTScore: HelaBERT\\_Large layer 9, unrescaled. mT5 row from the XL-Sum paper. Best per column in bold.",
            "tab:xlsum_si")
en_table = main_table("english",
            "XL-Sum v2.0 English, fixed random sample (seed 42): 3{,}249 train / 500 val / 500 test, same protocol as Sinhala. "
            "ROUGE with the scorer's English stemmer. BERTScore: RoBERTa-large layer 17, unrescaled. "
            "mT5 row from the XL-Sum paper (full 11{,}535-article test set). Best per column in bold.",
            "tab:xlsum_en")

tex = f"""\\documentclass[10pt,a4paper]{{article}}
\\usepackage[margin=1.7cm]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{float}}
\\usepackage{{caption}}
\\usepackage[hidelinks]{{hyperref}}
\\captionsetup{{font=footnotesize,labelfont=bf,skip=3pt}}
\\setlength{{\\parindent}}{{0pt}}
\\setlength{{\\parskip}}{{4pt}}
\\begin{{document}}
\\sloppy

\\begin{{center}}
{{\\LARGE\\bfseries XL-Sum: Sinhala and English summarization}}\\\\[3pt]
{{\\small llama-3-8b, SinLlama\\_v01, SinLlama\\_wiki\\_uc\\_gen\\_mix, SinLlama\\_textbook\\_mix ---
2026-10-31, RTX 4070 Ti SUPER}}
\\end{{center}}

{si_table}
{en_table}
{steps_table()}
\\end{{document}}
"""
OUT.write_text(tex)
print("wrote", OUT)
