#!/usr/bin/env python3
"""
Compare the raw / calibrated / cyclic-permuted MMLU arms.

The question this answers: the SinhalaMMLU gap between SinLlama_v02 and
SinLlama_uc_instruct_cleaned is -3.07pp raw, and highly significant. Is that
lost knowledge, or does uc_instruct_cleaned simply prefer a different option
slot? Removing position bias two independent ways settles it:

  * if the gap largely survives both debiasing arms -> real knowledge loss,
    and a stronger claim than the raw number could support;
  * if it collapses toward zero -> the chat SFT is much cheaper than reported
    and the raw benchmark was measuring answer-slot preference.

Also prints, per model and arm, the predicted-option distribution and its total
variation distance from the gold distribution, so you can see the bias actually
being removed rather than taking it on faith.

Usage:
  python benchmark/mmlu/compare_debias.py --root benchmark/mmlu
"""
import argparse
import glob
import json
import math
import os
from collections import Counter

ARMS = ("permuted", "calibrated")          # raw comes free from permuted's rot-0


def load_preds(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("valid"):
                    out.append(r)
    return out


def mcnemar(a_correct, b_correct):
    """Paired test on item-aligned correctness. Returns (b01, b10, p)."""
    b01 = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    b10 = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    chi2 = max(0.0, (abs(b01 - b10) - 1) ** 2 / n)
    return b01, b10, math.erfc(math.sqrt(chi2 / 2.0))


def tvd(pred_counts, gold_counts):
    sp, sg = sum(pred_counts), sum(gold_counts)
    if not sp or not sg:
        return float("nan")
    return 50.0 * sum(abs(p / sp - g / sg)
                      for p, g in zip(pred_counts, gold_counts))


def dist(rows, key, base, nmax):
    c = Counter(r[key] for r in rows)
    return [c.get(o, 0) for o in range(base, base + nmax)]


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def collect(root, tag):
    """arm -> model -> list of prediction rows."""
    found = {}
    for arm in ARMS:
        d = os.path.join(root, f"{tag}_results_{arm}")
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*_predictions.jsonl"))):
            model = os.path.basename(p).replace("_predictions.jsonl", "")
            found.setdefault(arm, {})[model] = load_preds(p)
    return found


def report(root, tag, base):
    found = collect(root, tag)
    if not found:
        print(f"\n(no {tag} arms found under {root})")
        return

    # the permuted dump carries pred_raw == the standard arm, same run
    raw = {}
    if "permuted" in found:
        for model, rows in found["permuted"].items():
            raw[model] = [dict(r, pred=r.get("pred_raw"),
                               correct=(r.get("pred_raw") == r["gold"]))
                          for r in rows]
    arms = {}
    if raw:
        arms["raw"] = raw
    for a in ARMS:
        if a in found:
            arms[a] = found[a]

    models = sorted({m for a in arms.values() for m in a})
    print(f"\n{'='*78}\n{tag}\n{'='*78}")

    print(f"\n{'arm':<12}" + "".join(f"{m.replace('SinLlama_',''):>26s}"
                                     for m in models))
    for arm, per_model in arms.items():
        row = f"{arm:<12}"
        for m in models:
            rows = per_model.get(m)
            if not rows:
                row += f"{'-':>26s}"
                continue
            c = sum(1 for r in rows if r["correct"])
            row += f"{pct(c, len(rows)):>19.2f}% {c:>5d}"
        print(row)

    # --- the headline: does the between-model gap survive debiasing? --------- #
    if len(models) >= 2:
        print(f"\n  between-model gap (later model minus earlier), by arm:")
        for i in range(len(models) - 1):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                print(f"    {b} - {a}:")
                for arm, per_model in arms.items():
                    ra, rb = per_model.get(a), per_model.get(b)
                    if not ra or not rb or len(ra) != len(rb):
                        continue
                    ca = [r["correct"] for r in ra]
                    cb = [r["correct"] for r in rb]
                    d = pct(sum(cb), len(cb)) - pct(sum(ca), len(ca))
                    b01, b10, p = mcnemar(ca, cb)
                    sig = "" if p >= 0.05 else ("  ***" if p < 0.001 else "  *")
                    print(f"      {arm:<12} {d:+7.2f} pp   "
                          f"(McNemar p={p:.4g}, {b01} vs {b10}){sig}")

    # --- did the debiasing actually remove bias? ---------------------------- #
    print(f"\n  predicted-option distribution and distance from gold:")
    for m in models:
        anyrows = next((a[m] for a in arms.values() if m in a), None)
        if not anyrows:
            continue
        nmax = max(r["n_choices"] for r in anyrows)
        gold = dist(anyrows, "gold", base, nmax)
        print(f"    {m}   gold = {'/'.join(map(str, gold))}")
        for arm, per_model in arms.items():
            rows = per_model.get(m)
            if not rows:
                continue
            pd = dist(rows, "pred", base, nmax)
            print(f"      {arm:<12} {'/'.join(map(str, pd)):<32s} "
                  f"TVD {tvd(pd, gold):5.1f} pp")

    # --- do the two debiasing methods agree? -------------------------------- #
    if "calibrated" in arms and "permuted" in arms:
        print(f"\n  agreement between the two debiasing methods:")
        for m in models:
            ca, pe = arms["calibrated"].get(m), arms["permuted"].get(m)
            if not ca or not pe or len(ca) != len(pe):
                continue
            same = sum(1 for x, y in zip(ca, pe) if x["pred"] == y["pred"])
            d = pct(sum(1 for r in pe if r["correct"]), len(pe)) - \
                pct(sum(1 for r in ca if r["correct"]), len(ca))
            print(f"    {m:<34} {pct(same, len(ca)):5.1f}% identical "
                  f"predictions, accuracy differs by {d:+.2f} pp")
        print("    (high agreement => the positional prior is content-independent,"
              "\n     so the cheap calibration is trustworthy; low agreement =>"
              "\n     trust only the permuted arm)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="benchmark/mmlu")
    args = ap.parse_args()
    report(args.root, "SinhalaMMLU", base=1)
    report(args.root, "EnglishMMLU", base=0)


if __name__ == "__main__":
    main()
