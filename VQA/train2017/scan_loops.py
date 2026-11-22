#!/usr/bin/env python3
"""Scan finished chunks (chunks/en_XX.json + si_XX.json) for repetition loops and print random samples.

A loop is a unit of 1-10 words repeated at least 3 times back to back ("A B C A B C A B C"), or an n-gram of
>= 6 words that occurs 3+ times in one field. Shorter n-grams that recur (e.g. "is coloured and") are normal
phrasing in long descriptions and are not counted. repair_nllb.py's detector only looked at 1-3 word units,
so longer loops can pass it.

Usage (lab box, ~/coco_train):  python3 scan_loops.py [--samples 2] [--out loops.jsonl]
"""
import argparse
import collections
import glob
import json
import random

FIELDS = ("description", "question", "answer")


def loop_info(t):
    w = t.split()
    for n in range(1, 11):
        for i in range(len(w) - 3 * n + 1):
            if w[i:i + n] == w[i + n:i + 2 * n] == w[i + 2 * n:i + 3 * n]:
                return f"{n}-word unit x3+: " + " ".join(w[i:i + n])
    g = collections.Counter(tuple(w[i:i + 6]) for i in range(len(w) - 5))
    if g:
        k, c = g.most_common(1)[0]
        if c >= 3:
            return f"6-gram x{c}: " + " ".join(k)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="chunks")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--out", default=None, help="write every hit to this jsonl")
    args = ap.parse_args()

    rng = random.Random(7)
    tot = collections.Counter()
    out = open(args.out, "w", encoding="utf-8") if args.out else None
    for path in sorted(glob.glob(f"{args.dir}/si_[0-9][0-9].json")):
        X = path[-7:-5]
        en = json.load(open(f"{args.dir}/en_{X}.json", encoding="utf-8"))
        si = json.load(open(path, encoding="utf-8"))
        hits = []
        for e, s in zip(en, si):
            for k in FIELDS:
                for lang, rec in (("en", e), ("si", s)):
                    li = loop_info(rec[k])
                    if li:
                        tot[(lang, k)] += 1
                        hits.append({"chunk": X, "lang": lang, "image_id": e["image_id"], "field": k,
                                     "why": li, "text": rec[k], "en": e[k]})
        if out:
            for h in hits:
                out.write(json.dumps(h, ensure_ascii=False) + "\n")
        n_en = sum(h["lang"] == "en" for h in hits)
        print(f"=== chunk {X}: {len(si)} records; loop hits: en {n_en}, si {len(hits) - n_en}")
        for h in hits[:3]:
            print(f"   HIT {h['lang']} id={h['image_id']} {h['field']} [{h['why'][:70]}]")
            print(f"       {h['text'][:230]}")
        for i in rng.sample(range(len(si)), args.samples):
            print(f"   sample id={si[i]['image_id']}")
            for k in FIELDS:
                print(f"     {k[:4]} EN: {en[i][k][:150]}")
                print(f"     {k[:4]} SI: {si[i][k][:150]}")
    print("totals (lang, field):", dict(tot))


if __name__ == "__main__":
    main()
