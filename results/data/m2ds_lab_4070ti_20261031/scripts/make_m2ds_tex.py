"""Build results/latex/M2DS_Summarization.tex from the score JSONs in
results/data/m2ds_lab_4070ti_20261031/ (no numbers typed by hand; layout as
XLSum_Summarization.tex). Models without predictions yet are left out.
Usage: python make_m2ds_tex.py   (then pdflatex in results/latex)
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1]
OUT = D.parents[1] / "latex" / "M2DS_Summarization.tex"
MODELS = ["llama-3-8b", "SinLlama_v01", "SinLlama_wiki_uc_gen_mix"]   # textbook_mix dropped by the user
BASELINES = ["LEAD-3", "Lead-1/article"]
NAME = {"llama-3-8b": "llama-3-8b (base)", "SinLlama_v01": "SinLlama\\_v01",
        "SinLlama_wiki_uc_gen_mix": "wiki\\_uc\\_gen\\_mix", "SinLlama_textbook_mix": "textbook\\_mix",
        "LEAD-3": "\\textit{LEAD-3}", "Lead-1/article": "\\textit{Lead-1 per article}"}
SHORT = {"llama-3-8b": "llama-3-8b", "SinLlama_v01": "v01",
         "SinLlama_wiki_uc_gen_mix": "wiki\\_uc\\_gen\\_mix", "SinLlama_textbook_mix": "textbook\\_mix"}
# M2DS paper (Hewapathirana et al. 2024), Table 3 (fine-tuned) / Table 4
PAPER = {"sinhala": [("Llama 2 7B (paper)", (20.2, 6.5, 17.3))],
         "english": [("Llama 2 7B (paper)", (20.8, 13.5, 19.2)),
                     ("PRIMERA fine-tuned (paper)", (28.7, 12.3, 17.1))]}
BS = {"sinhala": ("bertscore_helabert.json", "L9"), "english": ("bertscore_roberta_large.json", "L17")}
STEPS = [("SinLlama_v01", "llama-3-8b"), ("SinLlama_wiki_uc_gen_mix", "SinLlama_v01"),
         ("SinLlama_wiki_uc_gen_mix", "llama-3-8b")]
RK = ["rouge1", "rouge2", "rougeL"]


def load(lang):
    rouge = json.load(open(D / lang / "rouge.json"))
    bs_file, layer = BS[lang]
    bsp = D / lang / bs_file
    bs = json.load(open(bsp)) if bsp.exists() else {"scores": [], "paired_bootstrap": []}
    r = {t["model"]: t for t in rouge["scores"]}
    b = {t["model"]: t for t in bs["scores"]}
    rp = {(p["a"], p["b"]): p for p in rouge["paired_bootstrap"]}
    bp = {(p["a"], p["b"]): p[f"{layer}_F1"] for p in bs["paired_bootstrap"]}
    return r, b, rp, bp, layer


def main_table(lang, caption, label):
    r, b, _, _, layer = load(lang)
    models = [m for m in MODELS if m in r]
    cols = [[r[m][k] for m in models] for k in RK] + \
           [[b[m][f"{layer}_{k}"] if m in b else None for m in models] for k in ("P", "R", "F1")]
    best = [max((v for v in c if v is not None), default=None) for c in cols]
    cell = lambda v, bst: "--" if v is None else f"\\textbf{{{v:.2f}}}" if v == bst else f"{v:.2f}"
    rows = [f"{NAME[m]} & " + " & ".join(cell(c[i], bst) for c, bst in zip(cols, best)) + " \\\\"
            for i, m in enumerate(models)]
    base = [f"{NAME[m]} & " + " & ".join(f"\\textit{{{r[m][k]:.2f}}}" for k in RK) + " & -- & -- & -- \\\\"
            for m in BASELINES]
    off = []
    ro, bo = D / lang / "rouge_official.json", D / lang / BS[lang][0].replace(".json", "_official.json")
    if ro.exists():
        r_o = {t["model"]: t for t in json.load(open(ro))["scores"]}.get("SinLlama_v01")
        b_o = {t["model"]: t for t in json.load(open(bo))["scores"]}.get("SinLlama_v01") if bo.exists() else None
        if r_o:
            bcells = [f"\\textit{{{b_o[f'{layer}_{k}']:.2f}}}" if b_o else "--" for k in ("P", "R", "F1")]
            off = ["\\textit{SinLlama\\_v01, official train (leaky)} & "
                   + " & ".join(f"\\textit{{{r_o[k]:.2f}}}" for k in RK) + " & " + " & ".join(bcells) + " \\\\"]
    paper = [f"\\textit{{{n}}} & " + " & ".join(f"\\textit{{{v:.1f}}}" for v in vals) + " & -- & -- & -- \\\\"
             for n, vals in PAPER[lang]]
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
{chr(10).join(base)}
\\midrule
{chr(10).join(off + paper)}
\\bottomrule
\\end{{tabular}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table}}
"""


def fmt(d, p):
    ps = "$\\le10^{-4}$" if p <= 1e-4 else f"{p:.4f}" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
    sign = "$+$" if d >= 0 else "$-$"
    return f"{sign}{abs(d):.2f} ({ps})"


def steps_table():
    blocks = []
    for lang, title in (("sinhala", "Sinhala"), ("english", "English")):
        _, _, rp, bp, _ = load(lang)
        rows = [f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        for a, bm in STEPS:
            if (a, bm) not in rp:
                continue
            cells = [fmt(rp[(a, bm)][k]["delta"], rp[(a, bm)][k]["p"]) for k in RK]
            cells.append(fmt(bp[(a, bm)]["delta"], bp[(a, bm)]["p"]) if (a, bm) in bp else "--")
            rows.append(f"{SHORT[bm]} $\\rightarrow$ {SHORT[a]} & " + " & ".join(cells) + " \\\\")
        if len(rows) > 1:
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
\\caption{{M2DS, change between models: $\\Delta$ points (paired bootstrap $p$, 10{{,}}000 resamples).}}
\\label{{tab:m2ds_steps}}
\\end{{table}}
"""


def budget(lang, m):
    f = D / lang / "train_states" / f"{m}.max_source.txt"
    return int(f.read_text()) if f.exists() else None


def run_notes(lang):
    """Best checkpoint per model and clusters decoded with fewer than 4 beams (OOM fallback)."""
    best, fallback = [], []
    for m in MODELS:
        st = D / lang / "train_states" / f"{m}.json"
        if st.exists():
            ck = json.load(open(st)).get("best_checkpoint") or ""
            best.append(f"{SHORT[m]} {ck.rsplit('-', 1)[-1]}")
        pf = D / lang / "preds" / f"{m}.jsonl"
        if pf.exists():
            k = sum(json.loads(l).get("beams", 4) < 4 for l in open(pf))
            if k:
                fallback.append(f"{SHORT[m]} {k}")
    s = f"Best checkpoint step: {', '.join(best)}." if best else ""
    if fallback:
        s += f" Clusters decoded with fewer beams after GPU OOM: {', '.join(fallback)}."
    return s


fmt_n = lambda v: f"{v:,}".replace(",", "{,}") if v else "?"
LS = json.load(open(D / "leakage_stats.json"))
leak = lambda lang: LS[lang]["official"]["test_articles_in_train_pct"]
sin_b = sorted({budget("sinhala", m) for m in MODELS[1:] if budget("sinhala", m)})
ll_b = budget("sinhala", "llama-3-8b")
en_b = sorted({budget("english", m) for m in MODELS if budget("english", m)})
R_SI = {t["model"]: t for t in json.load(open(D / "sinhala" / "rouge.json"))["scores"]}
cov = lambda m: round(100 * R_SI[m]["source_coverage"]) if m in R_SI and R_SI[m].get("source_coverage") else "?"
si_table = main_table("sinhala",
            "M2DS Sinhala test (275 clusters, 2--10 articles each). The official splits leak: "
            f"{leak('sinhala')}\\% of test articles also occur in training clusters (whose summaries make up the test "
            f"references). Each model is therefore QLoRA-fine-tuned on the {fmt_n(LS['sinhala']['train_dedup'])} of "
            f"{fmt_n(LS['sinhala']['train_official'])} training clusters that share no article or summary sentence with "
            "validation or test (best checkpoint by val loss, $\\le$3 epochs, early stopping), beam 4. Articles are cut "
            "to a shared source budget, split fairly so no article is dropped: SinLlama "
            f"{'/'.join(fmt_n(v) for v in sin_b) or '?'} tokens; llama-3-8b {fmt_n(ll_b)} of its own tokens "
            "(its tokenizer needs $\\sim$8.5$\\times$ more tokens for Sinhala, so its training sequences, capped at 6{,}144 tokens to fit "
            "16 GB, leave room for only this much text next to its $\\sim$1.3k-token references): on average llama-3-8b "
            f"sees {cov('llama-3-8b')}\\% of a cluster's article text, the SinLlama models {cov('SinLlama_v01')}\\%. "
            "ROUGE: XL-Sum multilingual scorer (whitespace tokens, no stemmer). BERTScore: HelaBERT\\_Large layer 9, "
            "unrescaled. Italic rows: extractive baselines on the same test set; SinLlama\\_v01 fine-tuned on the full "
            "official training split (same recipe), showing the inflation from leakage; the M2DS paper's fine-tuned "
            f"Llama 2 (official splits). Best model per column in bold. {run_notes('sinhala')}",
            "tab:m2ds_si")
en_table = main_table("english",
            f"M2DS English test (all 1{{,}}687 clusters). Same protocol, source budget {'/'.join(fmt_n(v) for v in en_b) or '?'} "
            f"tokens; trained on the {fmt_n(LS['english']['train_dedup'])} of {fmt_n(LS['english']['train_official'])} "
            f"training clusters left after removing overlap with validation/test (officially {leak('english')}\\% of test "
            "articles occur in training), capped at 450 steps (7{,}200 clusters), val loss on a fixed "
            "300-cluster sample. 812 of the 1{,}687 references end with a scraped BBC promo fragment (``\\dots{} the best of our "
            "journalism in one place''), which models learn to reproduce; it is removed from references and predictions before "
            "scoring. ROUGE with the scorer's English stemmer. BERTScore: RoBERTa-large layer 17, unrescaled. "
            f"Paper rows from the M2DS paper (Tables 3--4). Best model per column in bold. {run_notes('english')}",
            "tab:m2ds_en")

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
{{\\LARGE\\bfseries M2DS: Sinhala and English multi-document summarization}}\\\\[3pt]
{{\\small llama-3-8b, SinLlama\\_v01, SinLlama\\_wiki\\_uc\\_gen\\_mix ---
2026-11-04, RTX 4070 Ti SUPER}}
\\end{{center}}

{si_table}
{en_table}
{steps_table()}
\\end{{document}}
"""
OUT.write_text(tex)
print("wrote", OUT)
