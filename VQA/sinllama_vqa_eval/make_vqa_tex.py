#!/usr/bin/env python3
"""Write results/latex/VQA_Benchmark.tex from the VQA eval outputs and the exact prompt strings.

The local TeX has no Unicode engine (no XeTeX / luaotfload), so the one Sinhala prompt is
rendered with pango-view (HarfBuzz shaping) to results/latex/VQA_Benchmark_si_prompt.pdf and
included as a vector graphic; everything else is plain pdflatex.

  python make_vqa_tex.py && cd ../../results/latex && pdflatex VQA_Benchmark.tex
"""
import json, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "VQA"))
import judge                                   # noqa: E402
from gen_sinllama import PROMPTS               # noqa: E402
from generate_vqa import DESCRIBE_PROMPT, CAPTIONS_BLOCK, QA_PROMPT  # noqa: E402

SI = re.compile(r"[\u0D80-\u0DFF\u200D]+(?:[ \u200D]*[\u0D80-\u0DFF\u200D]+)*")


def esc(s):
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("{", r"\{"), ("}", r"\}"), ("$", r"\$"), ("&", r"\&"), ("#", r"\#"),
                 ("_", r"\_"), ("%", r"\%"), ("^", r"\^{}"), ("~", r"\~{}")]:
        s = s.replace(a, b)
    return s.replace(r"\textbackslash\{\}", r"\textbackslash{}")


def prompt_block(text):
    """Monospace block, line breaks kept."""
    assert not SI.search(text), "Sinhala text needs sinhala_block()"
    out = []
    for para in text.split("\n\n"):
        lines = [esc(l) for l in para.split("\n")]
        out.append("\\newline ".join(l if l.strip() else r"\mbox{}" for l in lines))
    return ("\\begin{promptbox}\n" + "\\par\\smallskip\n".join(out) + "\n\\end{promptbox}\n")


def sinhala_block(text, pdf_path):
    """Render Sinhala text with pango-view (proper conjunct shaping) to a vector PDF to include."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
    subprocess.run(["pango-view", "-q", "--font=Noto Sans Mono, Noto Sans Sinhala 8", "--dpi=72",
                    "--width=500", "--wrap=word", "--margin=0", f"--output={pdf_path}", f.name], check=True)
    os.unlink(f.name)
    return ("\\begin{promptbox}\n\\includegraphics{" + os.path.basename(pdf_path) + "}\n\\end{promptbox}\n")


def pval(p):
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def fine_tune_rows(keys):
    """Test-set table (fine-tuned vs the 0-/5-shot runs on the same images) + training table."""
    sf = json.load(open(os.path.join(HERE, "ft/summary_ft.json")))
    test = set(json.load(open(os.path.join(HERE, "ft/test_ids.json"))))
    rows, tr = [], []
    for arm, lang in [("en", "English"), ("si", "Sinhala")]:
        a = sf[arm]
        g5 = [r for r in json.load(open(os.path.join(HERE, f"5shot/sinllama_{arm}_generated_5shot.json")))["records"]
              if r["image_id"] in test]
        tok = {"zero_shot": a["generation_zero_shot"]["mean_generated_tokens"],
               "five_shot": sum(r["generated_tokens"] for r in g5) / len(g5),
               "finetuned": a["generation_ft"]["mean_generated_tokens"]}
        for key, name, cmp in [("zero_shot", "0-shot", None), ("five_shot", "5-shot", None),
                               ("finetuned", "Fine-tuned", "ft_vs_zero_shot")]:
            c = a[key]
            sc = f"\\textbf{{{c['score_100']:.2f}}}" if key == "finetuned" else f"{c['score_100']:.2f}"
            d = (f"{a[cmp]['diff_100']:+.2f} ({pval(a[cmp]['wilcoxon_p'])})" if cmp else "--")
            rows.append(f"{lang} & {name} & {c['n']} & {sc} & " + " & ".join(f"{c[k]:.2f}" for k in keys)
                        + f" & {tok[key]:.0f} & {d} \\\\")
        if arm == "en":
            rows.append(r"\midrule")
        t = json.load(open(os.path.join(HERE, f"ft/{arm}/train_state.json")))
        first = next(x["eval_loss"] for x in t["log_history"] if "eval_loss" in x)
        best = int(t["best_checkpoint"].rsplit("-", 1)[1])
        tr.append(f"{lang} & {t['n_train']} / {t['n_val']} & {best} ({best / t['global_step'] * t['epochs_done']:.1f}) "
                  f"& {t['global_step']} & {first:.3f} $\\to$ {t['best_eval_loss']:.3f} & {t['train_minutes']:.0f} \\\\")
    return "\n".join(rows), "\n".join(tr)


def main():
    s5 = json.load(open(os.path.join(HERE, "5shot/summary_5shot.json")))
    toks = {}
    for arm in ["en", "si"]:
        ids = {r["image_id"] for r in json.load(open(os.path.join(HERE, f"5shot/judged_{arm}_5shot.json")))["records"]}
        g0 = [r for r in json.load(open(os.path.join(HERE, f"sinllama_{arm}_generated.json")))["records"]
              if r["image_id"] in ids]
        toks[arm, 0] = sum(r["generated_tokens"] for r in g0) / len(g0)
        toks[arm, 5] = s5[arm]["mean_generated_tokens_5shot"]
    keys = [c["key"] for c in judge.CRITERIA]
    rows = []
    for arm, lang in [("en", "English"), ("si", "Sinhala")]:
        z, f = s5[arm]["zero_shot_same_items"], s5[arm]["five_shot"]
        for shots, sc, crit in [(0, z["score_100"], z), (5, f["score_100"], f["criteria_100"])]:
            rows.append(f"{lang} & {shots} & \\textbf{{{sc:.2f}}} & "
                        + " & ".join(f"{crit[k]:.2f}" for k in keys) + f" & {toks[arm, shots]:.0f} \\\\")
        if arm == "en":
            rows.append(r"\midrule")

    ft_rows, tr_rows = fine_tune_rows(keys)

    crit_rows = "\n".join(f"{esc(c['name'])} & {c['weight']:.2f} & {esc(c['description'])} \\\\"
                          for c in judge.CRITERIA)
    crit_list = "\n".join(f"{i}. {c['key']} ({c['name']}): {c['description']}"
                          for i, c in enumerate(judge.CRITERIA, 1))
    judge_prompt = judge.PROMPT.format(
        description="{description}", question="{question}", reference="{reference answer}",
        candidate="{candidate answer}", criteria=crit_list, scale=judge.SCALE,
        mt_note="[Sinhala arm only] " + judge.MT_NOTE + "\n")
    shot_ids = s5["shot_image_ids"]

    sinllama_prompt = "<|begin_of_text|>### User:\n" + PROMPTS["en"] + "\n\n### Assistant:"
    sinllama_si = PROMPTS["si"]
    describe = DESCRIBE_PROMPT + CAPTIONS_BLOCK.format(captions="- {caption 1}\n- ...\n- {caption 5}")
    qa = QA_PROMPT.format(description="{description}",
                          captions_block="Human-written reference captions of the image (reliable but brief):\n"
                                         "- {caption 1}\n- ...\n- {caption 5}\n\n")

    tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.7cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{float}
\usepackage{caption}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=footnotesize,labelfont=bf,skip=3pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\newenvironment{promptbox}{\par\smallskip\noindent\rule{\linewidth}{0.4pt}\par\raggedright\ttfamily\small}{\par\noindent\rule{\linewidth}{0.4pt}\par\medskip}
\begin{document}
\sloppy

\begin{center}
{\LARGE\bfseries VQA: Sinhala and English visual question answering}\\[3pt]
{\small SinLlama\_wiki\_uc\_gen\_mix, judged by Qwen3.5-9B --- 2026-11-06, RTX 4070 Ti SUPER}
\end{center}

\begin{table}[H]
\centering\small
\begin{tabular}{lccccccc}
\toprule
Language & Shots & Score & Q.\ underst. & Correctness & Completeness & Coherence & Ans.\ tokens \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{SinLlama\_wiki\_uc\_gen\_mix on the full VQA dataset (5{,}000 samples; both rows scored on the 4{,}995 that are not 5-shot exemplars), score /100, Qwen3.5-9B judge, 0-shot and 5-shot.}
\label{tab:results}
\end{table}

\begin{table}[H]
\centering\small
\begin{tabularx}{\linewidth}{lcX}
\toprule
Criterion & Weight & Definition given to the judge \\
\midrule
""" + crit_rows + r"""
\bottomrule
\end{tabularx}
\caption{Judging criteria. """ + esc(judge.SCALE) + r""" Empty answers score 0 without a judge call.}
\label{tab:criteria}
\end{table}

\section*{Fine-tuning (QLoRA)}
One LoRA adapter per language on SinLlama\_wiki\_uc\_gen\_mix, 80/10/10 split by image (seed 42, same images for
both languages): 4{,}000 / 500 / 500, minus 6 training rows whose reference answers repeat themselves.
Hyperparameters as the textbook SFT: r 8, $\alpha$ 16, dropout 0.05, all 7 projections, lr $2\cdot10^{-4}$ cosine,
warmup 3\%, batch 64, max 3 epochs, loss on the answer only; base in 4-bit NF4; validation every 10 steps,
early stopping (patience 3), best checkpoint kept. At test time the adapter is merged into the bf16 model and
decoded as in the 0-shot run (greedy, repetition penalty 1.1, max 256 new tokens).

\begin{table}[H]
\centering\small
\begin{tabular}{lccccc}
\toprule
Language & Train / val & Best step (epoch) & Stopped at & Val loss (step 10 $\to$ best) & Minutes \\
\midrule
""" + tr_rows + r"""
\bottomrule
\end{tabular}
\caption{Training.}
\label{tab:ft-train}
\end{table}

\begin{table}[H]
\centering\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{llcccccccc}
\toprule
Language & Model & $n$ & Score & Q.\ underst. & Correctness & Completeness & Coherence & Ans.\ tokens & $\Delta$ vs 0-shot ($p$) \\
\midrule
""" + ft_rows + r"""
\bottomrule
\end{tabular}
\caption{Test set (500 images). 0-shot and 5-shot rows are the runs of Table~\ref{tab:results} restricted to the
test images (5-shot $n=499$: one exemplar is a test image). $p$: paired Wilcoxon signed-rank on per-answer scores.}
\label{tab:ft}
\end{table}

\clearpage
\section*{Judge prompt}
One call per answer, output constrained to JSON (\texttt{rationale} + the four integer scores).
""" + prompt_block(judge_prompt) + r"""
\section*{SinLlama prompt}
Chat template of the UltraChat SFT; generation stops at \texttt{<|end\_of\_text|>}. In the 5-shot setting
the five exemplars precede the test item as complete User/Assistant turns.
""" + prompt_block(sinllama_prompt) + r"""
Sinhala user turn:
""" + sinhala_block(sinllama_si, os.path.join(ROOT, "results/latex/VQA_Benchmark_si_prompt.pdf")) + r"""
\section*{Dataset generation prompts}
Qwen3.5-9B Q4\_K\_M (vision, llama.cpp), two calls per COCO val2017 image (5{,}000 images), each also given the
image's five human COCO captions. Sampling: temperature 0.7, top-p 0.8, top-k 20 (description: presence
penalty 0.5, repeat penalty 1.05); the question/answer call is JSON-schema constrained. The Sinhala set is the
English set translated with NLLB-200 3.3B.

\textbf{1. Description} (image attached):
""" + prompt_block(describe) + r"""
\textbf{2. Question and answer} (image attached):
""" + prompt_block(qa) + r"""
\end{document}
"""
    out = os.path.join(ROOT, "results/latex/VQA_Benchmark.tex")
    open(out, "w", encoding="utf-8").write(tex)
    print("wrote", out)


if __name__ == "__main__":
    main()
