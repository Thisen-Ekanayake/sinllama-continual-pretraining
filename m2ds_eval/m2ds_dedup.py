"""Leakage-free M2DS training splits.

M2DS clusters are built from BBC "related links", so one article (and its BBC
summary, which becomes part of the cluster reference) appears in many clusters
across the official train/validation/test splits. This keeps validation and
test unchanged and drops every training cluster that shares an article with
validation or test (articles matched on their first 200 characters after
whitespace normalisation) or a reference-summary sentence of >= 5 words with
them (the same BBC summary is sometimes attached to re-worded articles).
Boilerplate sentences found in >= 20 training clusters (e.g. the BBC promo
line scraped into many English summaries) are not counted as overlap.
Writes <out>/<lang>/{train,validation,test}.json and <out>/leakage_stats.json
(overlap before/after).
Usage: python3 m2ds_dedup.py ~/m2ds ~/m2ds_dedup
"""
import collections, json, os, re, shutil, sys

src, out = sys.argv[1], sys.argv[2]
SENT = re.compile(r"(?<=[.!?])\s+")
norm = lambda s: " ".join(s.split())
key = lambda a: norm(a)[:200]
arts = lambda r: [key(a) for a in r["document"].split("|||||") if a.strip()]
sents = lambda t: [norm(x) for x in SENT.split(norm(t)) if len(x.split()) >= 5]


def overlap(train, test, boiler=frozenset()):
    tr_art = {a for r in train for a in arts(r)}
    tr_sum = {s for r in train for s in sents(r["summary"])}
    te_art = [a for r in test for a in arts(r)]
    te_sum = [s for r in test for s in sents(r["summary"]) if s not in boiler]
    return {
        "test_articles_in_train_pct": round(100 * sum(a in tr_art for a in te_art) / len(te_art), 1),
        "test_ref_sentences_in_train_summaries_pct_excl_boilerplate": round(100 * sum(s in tr_sum for s in te_sum) / len(te_sum), 1),
        "test_clusters_sharing_no_article": sum(not any(a in tr_art for a in arts(r)) for r in test),
    }


stats = {}
for lang in ("sinhala", "english"):
    rd = lambda split: [json.loads(l) for l in open(f"{src}/{lang}/{split}.json")]
    train, val, test = rd("train"), rd("validation"), rd("test")
    freq = collections.Counter(s for r in train for s in set(sents(r["summary"])))
    boiler = {s for s, c in freq.items() if c >= 20}
    held = {a for r in val + test for a in arts(r)}
    held_s = {s for r in val + test for s in sents(r["summary"])} - boiler
    kept = [r for r in train if not any(a in held for a in arts(r))
            and not any(s in held_s for s in sents(r["summary"]))]
    os.makedirs(f"{out}/{lang}", exist_ok=True)
    with open(f"{out}/{lang}/train.json", "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for split in ("validation", "test"):
        shutil.copy(f"{src}/{lang}/{split}.json", f"{out}/{lang}/{split}.json")
    stats[lang] = {"train_official": len(train), "train_dedup": len(kept), "boilerplate_sentences": len(boiler), "validation": len(val), "test": len(test),
                   "official": overlap(train, test, boiler), "dedup": overlap(kept, test, boiler),
                   "val_official": overlap(train, val, boiler), "val_dedup": overlap(kept, val, boiler)}
    print(lang, json.dumps(stats[lang]))
json.dump(stats, open(f"{out}/leakage_stats.json", "w"), indent=1)
