#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean `data/mmlu-sinhala/auxiliary_train` into an SFT-ready set and report the
token-length distribution under the SinLlama tokenizer.

Only auxiliary_train is touched. dev/validation/test are Sinhala translations of
the exact English MMLU items that mmlu/evaluate_english_mmlu.py scores, so
training on them would contaminate that eval cross-lingually — they are read
here purely to build the leak blacklist.

Prompt format matches mmlu/evaluate_sinhala_mmlu.py exactly (same template, same
clean(), same "N. option" rendering, same trailing "පිළිතුර: "), so the token the
model is trained to emit is the token the evaluator scores. auxiliary_train has
no `subject`, so [SUBJECT] takes the evaluator's own empty-subject fallback.

Output: JSONL with the cleaned item plus the rendered prompt/completion, so it
can be re-templated later without re-running the filters.
"""
import argparse
import json
import os
import re
import random
import collections

# --------------------------------------------------------------------------- #
# Prompt rendering — kept byte-identical to mmlu/evaluate_sinhala_mmlu.py
# --------------------------------------------------------------------------- #
TEMPLATE = (
    "මෙය [SUBJECT] විෂයයට අදාළ බහුවරණ ප්‍රශ්නයකි. පහත ප්‍රශ්නයට [NUMS] යන "
    "පිළිතුරු වලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන හෝ පිළිතුර තෝරන්න.\n"
    "ප්‍රශ්නය: [QUESTION]\n"
    "[OPTIONS]\n"
    "පිළිතුර:"
)
SUBJECT_FALLBACK = "විෂය"           # evaluator's `clean(subject) or "විෂය"`


def clean(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def render(question, choices):
    n = len(choices)
    nums = ", ".join(str(i) for i in range(1, n + 1))
    opts = "\n".join(f"{i+1}. {clean(c)}" for i, c in enumerate(choices))
    block = (TEMPLATE
             .replace("[SUBJECT]", SUBJECT_FALLBACK)
             .replace("[NUMS]", nums)
             .replace("[QUESTION]", clean(question))
             .replace("[OPTIONS]", opts))
    return block + " "                       # next token = the answer digit


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
# Options whose meaning depends on their position ("all of the above", "A and
# B", ...). These items are kept but never shuffled.
ORDER_DEPENDENT = ("ඉහත සියල්ල", "ඉහත සඳහන් සියල්ල", "ඉහත කිසිවක්",
                   "කිසිවක් නොවේ", "A සහ B", "I සහ II")


def order_dependent(choices):
    return any(p in str(c) for c in choices for p in ORDER_DEPENDENT)


def degenerate(text, min_rep=3, max_unique=0.6):
    """True if the text loops: some word 5-gram occurs >= min_rep times AND
    fewer than `max_unique` of all its 5-grams are distinct. Both conditions
    must hold — a single repeated 5-gram alone false-positives on ordinary RACE
    passages that simply reuse a topic phrase.

    Deliberately not a regex backreference (`(.{15,}?)\\1{2,}`): that backtracks
    catastrophically on the multi-KB passages in this corpus."""
    w = text.split()
    if len(w) < 20:
        return False
    grams = [tuple(w[i:i+5]) for i in range(len(w) - 4)]
    counts = collections.Counter(grams)
    return (counts.most_common(1)[0][1] >= min_rep
            and len(counts) / len(grams) < max_unique)


def norm_q(s):
    return re.sub(r"\s+", "", str(s))


def load_records(path):
    """auxiliary_train wraps every record as {"train": {...}}; eval splits don't."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [r["train"] if isinstance(r, dict) and "train" in r else r for r in data]


def build_leak_set(root):
    leaks = set()
    import glob
    for split in ("dev", "validation", "test"):
        for f in sorted(glob.glob(os.path.join(root, "*", f"{split}-*.json"))):
            for r in load_records(f):
                leaks.add(norm_q(r.get("question", "")))
    return leaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/mmlu-sinhala")
    ap.add_argument("--out", default="data/mmlu_sft/aux_train_clean.jsonl")
    ap.add_argument("--tokenizer", default="models/SinLlama_v01")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-shuffle-options", action="store_true",
                    help="keep the original option order (leaves the position "
                         "prior of the raw data intact)")
    ap.add_argument("--max-seq-lengths", default="512,768,1024,1280,1536",
                    help="candidate max_seq_length values to report coverage for")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = os.path.join(args.data_root, "auxiliary_train",
                       "train-00000-of-00001.sinhala.json")

    print(f"loading {src} ...", flush=True)
    recs = load_records(src)
    n_in = len(recs)
    print(f"  {n_in} raw records")

    print("building eval-leak blacklist from dev/validation/test ...", flush=True)
    leaks = build_leak_set(args.data_root)
    print(f"  {len(leaks)} distinct eval questions")

    drop = collections.Counter()
    seen = set()
    kept = []
    n_shuffled = 0

    for r in recs:
        q = r.get("question")
        ch = r.get("choices")
        a = r.get("answer")

        # 1. structural validity
        if not isinstance(ch, list) or len(ch) != 4 or not isinstance(a, int) \
                or not (0 <= a < len(ch)):
            drop["malformed"] += 1
            continue
        if not clean(q) or any(not clean(c) for c in ch):
            drop["empty_field"] += 1
            continue

        # 2. leakage into the eval splits
        key = norm_q(q)
        if key in leaks:
            drop["eval_leak"] += 1
            continue

        # 3. duplicate question (keep first occurrence)
        if key in seen:
            drop["duplicate"] += 1
            continue
        seen.add(key)

        # 4. options the translator collapsed into each other
        opts = [clean(c) for c in ch]
        if len(set(opts)) < 4:
            drop["collapsed_choices"] += 1
            continue

        # 5. MT degeneration
        if degenerate(clean(q)):
            drop["degenerate_text"] += 1
            continue

        # 6. de-bias option position
        gold = opts[a]
        if not args.no_shuffle_options and not order_dependent(opts):
            rng.shuffle(opts)
            n_shuffled += 1
        a = opts.index(gold)

        kept.append({"question": clean(q), "choices": opts, "answer": a})

    print("\n--- filter report ---")
    print(f"  input                     {n_in:>7}")
    for k in ("malformed", "empty_field", "eval_leak", "duplicate",
              "collapsed_choices", "degenerate_text"):
        print(f"  - {k:<24}{drop[k]:>7}")
    print(f"  = kept                    {len(kept):>7}"
          f"   ({100*len(kept)/n_in:.2f}% of input)")
    print(f"  options shuffled          {n_shuffled:>7}"
          f"   ({len(kept)-n_shuffled} left in original order)")

    post = collections.Counter(r["answer"] for r in kept)
    print("  answer index distribution: "
          + "  ".join(f"{i}={100*post[i]/len(kept):.1f}%" for i in sorted(post)))

    # ----------------------------------------------------------------------- #
    # Render + tokenize
    # ----------------------------------------------------------------------- #
    from transformers import AutoTokenizer
    print(f"\nloading tokenizer {args.tokenizer} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    prompts = [render(r["question"], r["choices"]) for r in kept]
    targets = [str(r["answer"] + 1) for r in kept]

    print(f"tokenizing {len(prompts)} prompts ...", flush=True)
    plens, tlens = [], []
    B = 1000
    for i in range(0, len(prompts), B):
        for ids in tok(prompts[i:i+B], add_special_tokens=False)["input_ids"]:
            plens.append(len(ids))
        for ids in tok(targets[i:i+B], add_special_tokens=False)["input_ids"]:
            tlens.append(len(ids))
        if (i // B) % 20 == 0:
            print(f"  {min(i+B, len(prompts))}/{len(prompts)}", flush=True)

    # +1 for BOS, +1 for EOS after the answer digit
    total = [p + t + 2 for p, t in zip(plens, tlens)]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r, p, c, n in zip(kept, prompts, targets, total):
            f.write(json.dumps({**r, "prompt": p, "completion": c,
                                "n_tokens": n}, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")

    # ----------------------------------------------------------------------- #
    # Distribution report
    # ----------------------------------------------------------------------- #
    s = sorted(total)
    n = len(s)

    def pct(p):
        return s[min(int(n * p / 100), n - 1)]

    print("\n--- token length distribution (BOS + prompt + answer + EOS) ---")
    print(f"  count   {n}")
    print(f"  mean    {sum(s)/n:.1f}")
    print(f"  min     {s[0]}")
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9):
        print(f"  p{p:<6} {pct(p)}")
    print(f"  max     {s[-1]}")

    print("\n  histogram")
    edges = [0, 64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 10**9]
    import bisect
    for lo, hi in zip(edges, edges[1:]):
        c = bisect.bisect_left(s, hi) - bisect.bisect_left(s, lo)
        label = f"{lo:>4}-{hi:<4}" if hi < 10**9 else f"{lo:>4}+    "
        bar = "#" * round(60 * c / n)
        print(f"   {label} {c:>6} {100*c/n:>5.1f}% {bar}")

    sup = sum(t + 1 for t in tlens)          # answer digit + EOS
    print(f"\n  total tokens        {sum(s):,}  ({sum(s)/1e6:.2f}M)")
    print(f"  supervised tokens   {sup:,}  ({100*sup/sum(s):.2f}% of the corpus)")
    print(f"    ^ only the answer digit + EOS carries loss if the prompt is masked")

    print("\n  coverage / padding by max_seq_length")
    print(f"   {'msl':>6} {'items kept':>11} {'%':>7} {'padded tokens':>15} {'waste':>7}")
    for msl in [int(x) for x in args.max_seq_lengths.split(",")]:
        fit = bisect.bisect_right(s, msl)
        padded = msl * n
        waste = 100 * (1 - sum(min(x, msl) for x in s) / padded)
        print(f"   {msl:>6} {fit:>11} {100*fit/n:>6.2f}% {padded:>15,} {waste:>6.1f}%")


if __name__ == "__main__":
    main()
