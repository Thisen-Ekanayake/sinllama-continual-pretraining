"""Score M2DS predictions with the XL-Sum repo's multilingual ROUGE (same scorer
as the XL-Sum report: use_stemmer=True with rouge_lang, BootstrapAggregator mid
F1 x100; Sinhala has no stemmer, so unicode punctuation stripping + whitespace
tokens, as in the M2DS paper's space-based Sinhala tokenisation).
Also scores two extractive baselines built from the test clusters:
  LEAD-3      first three sentences of the concatenated source (the paper's LEAD-3)
  Lead-1/art  first sentence of every article in the cluster (each M2DS reference
              is the concatenation of the articles' own BBC summaries)
and paired bootstrap (10,000 resamples) between the models in PAIRS.
References and predictions are cleaned of the BBC promo fragment that the
M2DS scrape appended to many English summaries ("...around the world and the
unique human stories behind current events, we've got the best of our
journalism in one place."; 812 of 1,687 English test references): models learn
to emit it and it would otherwise count as a 20-word free match.
Usage: PYTHONPATH=pkgs python score_m2ds.py preds_dir test.json out_prefix sinhala|english
"""
import json, re, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from rouge_score import rouge_scorer, scoring

MODELS = ["llama-3-8b", "SinLlama_v01", "SinLlama_wiki_uc_gen_mix"]   # textbook_mix dropped by the user
PAIRS = [("SinLlama_v01", "llama-3-8b"), ("SinLlama_wiki_uc_gen_mix", "SinLlama_v01"),
         ("SinLlama_wiki_uc_gen_mix", "llama-3-8b")]
KEYS = ["rouge1", "rouge2", "rougeL"]
SENT = re.compile(r"(?<=[.!?])\s+")
PROMO = re.compile(r"[^.!?]*best of our journalism in one place[.!?]?")
clean = lambda t: " ".join(PROMO.sub(" ", t).split())

np.random.seed(1234)
preds_dir, test_file, out, lang = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
scorer = rouge_scorer.RougeScorer(KEYS, lang=lang, use_stemmer=True)
test = [json.loads(l) for l in open(test_file)]


def sents(text):
    return [s for s in SENT.split(" ".join(text.split())) if s]


def baselines():
    lead3, lead1 = [], []
    for i, r in enumerate(test):
        arts = [a for a in r["document"].split("|||||") if a.strip()]
        lead3.append({"id": i, "prediction": " ".join(sents(" ".join(arts))[:3]), "reference": r["summary"].strip()})
        lead1.append({"id": i, "prediction": " ".join(s[0] for s in map(sents, arts) if s), "reference": r["summary"].strip()})
    return {"LEAD-3": lead3, "Lead-1/article": lead1}


systems = dict(baselines())
for m in MODELS:
    f = preds_dir / f"{m}.jsonl"
    if f.exists():
        systems[m] = sorted((json.loads(l) for l in open(f)), key=lambda r: r["id"])

per_ex, table = {}, []
for m, rows in systems.items():
    assert len(rows) == len(test) and all(r["id"] == i for i, r in enumerate(rows)), m
    assert all(r["reference"] == t["summary"].strip() for r, t in zip(rows, test)), m
    agg = scoring.BootstrapAggregator()
    s = {k: [] for k in KEYS}
    for r in rows:
        sc = scorer.score(clean(r["reference"]), clean(r["prediction"]))
        agg.add_scores(sc)
        for k in KEYS:
            s[k].append(sc[k].fmeasure)
    res = agg.aggregate()
    per_ex[m] = {k: np.array(v) for k, v in s.items()}
    table.append({
        "model": m, "n": len(rows),
        **{k: round(res[k].mid.fmeasure * 100, 2) for k in KEYS},
        **{k + "_ci": [round(res[k].low.fmeasure * 100, 2), round(res[k].high.fmeasure * 100, 2)] for k in KEYS},
        **{k + "_P": round(res[k].mid.precision * 100, 2) for k in KEYS},
        **{k + "_R": round(res[k].mid.recall * 100, 2) for k in KEYS},
        "pred_words": round(float(np.mean([len(clean(r["prediction"]).split()) for r in rows])), 1),
        "ref_words": round(float(np.mean([len(clean(r["reference"]).split()) for r in rows])), 1),
        "promo_in_pred": sum(bool(PROMO.search(r["prediction"])) for r in rows),
        "empty_preds": sum(not r["prediction"] for r in rows),
        "hit_token_cap": sum(r.get("hit_cap", False) for r in rows),
        "source_coverage": round(float(np.mean([r["source_tokens_used"] / r["source_tokens_total"] for r in rows])), 3)
        if "source_tokens_used" in rows[0] else None,
    })

print(f"{'system':28s} {'n':>5s} {'R-1':>6s} {'R-2':>6s} {'R-L':>6s}  pred/ref words")
for t in table:
    print(f"{t['model']:28s} {t['n']:5d} {t['rouge1']:6.2f} {t['rouge2']:6.2f} {t['rougeL']:6.2f}  "
          f"{t['pred_words']}/{t['ref_words']}  (empty {t['empty_preds']}, hit cap {t['hit_token_cap']}, "
          f"source coverage {t['source_coverage']})")

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
