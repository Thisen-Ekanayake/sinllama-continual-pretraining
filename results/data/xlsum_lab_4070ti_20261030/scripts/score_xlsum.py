"""Score XL-Sum predictions with the XL-Sum repo's multilingual ROUGE, as in
/ml/xl-sum/seq2seq/utils.py:calculate_rouge (use_stemmer=True, rouge_lang,
BootstrapAggregator mid F1 x100). For Sinhala no stemmer/segmenter exists, so
the scorer falls back to unicode punctuation stripping + whitespace tokens.
Also: 95% bootstrap CI, mean prediction length, and paired bootstrap between
consecutive models listed in PAIRS.
Usage: PYTHONPATH=pkgs python score_xlsum.py preds_dir out_prefix [sinhala|english]
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from rouge_score import rouge_scorer, scoring

MODELS = ["llama-3-8b", "SinLlama_v01", "SinLlama_wiki_uc_gen_mix", "SinLlama_textbook_mix"]
PAIRS = [("SinLlama_v01", "llama-3-8b"), ("SinLlama_wiki_uc_gen_mix", "SinLlama_v01"),
         ("SinLlama_textbook_mix", "SinLlama_wiki_uc_gen_mix"), ("SinLlama_textbook_mix", "llama-3-8b")]
KEYS = ["rouge1", "rouge2", "rougeL"]

np.random.seed(1234)
preds_dir, out = Path(sys.argv[1]), sys.argv[2]
lang = sys.argv[3] if len(sys.argv) > 3 else "sinhala"     # english -> Snowball stemmer, as XL-Sum
scorer = rouge_scorer.RougeScorer(KEYS, lang=lang, use_stemmer=True)
per_ex, table = {}, []
for m in MODELS:
    f = preds_dir / f"{m}.jsonl"
    if not f.exists():
        continue
    rows = sorted((json.loads(l) for l in open(f)), key=lambda r: r["id"])
    agg = scoring.BootstrapAggregator()
    s = {k: [] for k in KEYS}
    for r in rows:
        sc = scorer.score(r["reference"], r["prediction"])
        agg.add_scores(sc)
        for k in KEYS:
            s[k].append(sc[k].fmeasure)
    res = agg.aggregate()
    per_ex[m] = {k: np.array(v) for k, v in s.items()}
    table.append({
        "model": m, "n": len(rows),
        **{k: round(res[k].mid.fmeasure * 100, 2) for k in KEYS},
        **{k + "_ci": [round(res[k].low.fmeasure * 100, 2), round(res[k].high.fmeasure * 100, 2)] for k in KEYS},
        "pred_words": round(float(np.mean([len(r["prediction"].split()) for r in rows])), 1),
        "ref_words": round(float(np.mean([len(r["reference"].split()) for r in rows])), 1),
        "empty_preds": sum(not r["prediction"] for r in rows),
        "hit_token_cap": sum(r.get("hit_cap", False) for r in rows),
        
    })

print(f"{'model':28s} {'n':>4s} {'R-1':>6s} {'R-2':>6s} {'R-L':>6s}  pred/ref words")
for t in table:
    print(f"{t['model']:28s} {t['n']:4d} {t['rouge1']:6.2f} {t['rouge2']:6.2f} {t['rougeL']:6.2f}  "
          f"{t['pred_words']}/{t['ref_words']}  (empty {t['empty_preds']}, hit 84-token cap {t['hit_token_cap']})")

pairs = []
for a, b in PAIRS:
    if a not in per_ex or b not in per_ex:
        continue
    row = {"a": a, "b": b}
    for k in KEYS:
        d = per_ex[a][k] - per_ex[b][k]
        boot = np.array([d[np.random.randint(0, len(d), len(d))].mean() for _ in range(10000)])
        p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
        row[k] = {"delta": round(d.mean() * 100, 2), "p": round(float(max(p, 1e-4)), 4)}
    pairs.append(row)
    print(f"{a} - {b}: " + ", ".join(f"{k} {row[k]['delta']:+.2f} (p={row[k]['p']})" for k in KEYS))

json.dump({"scores": table, "paired_bootstrap": pairs}, open(out + ".json", "w"), indent=1)
