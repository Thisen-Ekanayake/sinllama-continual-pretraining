#!/usr/bin/env python3
"""
MMLU predicted-answer distribution for the six models in
results_latest/MMLU_Benchmark.pdf.

No model is re-run. The prediction dumps that produced the PDF are located by
fingerprinting: a candidate *_predictions.jsonl is accepted only if its
valid-row count AND its correct count reproduce the PDF exactly, and then its
per-domain / per-difficulty accuracies are re-derived and checked against the
PDF as well. Only after all of that passes is the pred distribution reported.

Rows with valid=false are excluded throughout (that is what the evaluator's own
metrics do: 6882 dumped -> 6878 scored on Sinhala).
"""
import json, os, sys, argparse
from collections import Counter, OrderedDict

MODELS = ["llama-3-8b", "SinLlama_v01", "SinLlama_cpt",
          "SinLlama_Bactrianx_Instruct", "SinLlama_v02",
          "SinLlama_uc_instruct_cleaned"]

# (valid_total, correct) as printed in results_latest/MMLU_Benchmark.pdf
PDF = {
 "sinhala": {"total": 6878, "correct": {
    "llama-3-8b": 2226, "SinLlama_v01": 2587, "SinLlama_cpt": 2452,
    "SinLlama_Bactrianx_Instruct": 1729, "SinLlama_v02": 2882,
    "SinLlama_uc_instruct_cleaned": 2684}},
 "english": {"total": 14042, "correct": {
    "llama-3-8b": 9197, "SinLlama_v01": 6706, "SinLlama_cpt": 5635,
    "SinLlama_Bactrianx_Instruct": 4609, "SinLlama_v02": 7187,
    "SinLlama_uc_instruct_cleaned": 7070}},
}

# Per-domain accuracies from the PDF, used as an independent second check.
PDF_DOMAIN = {
 "sinhala": {
  "business_studies": {"llama-3-8b":32.61,"SinLlama_v01":42.76,"SinLlama_cpt":38.23,"SinLlama_Bactrianx_Instruct":26.78,"SinLlama_v02":43.20,"SinLlama_uc_instruct_cleaned":39.74},
  "humanities":       {"llama-3-8b":32.36,"SinLlama_v01":36.28,"SinLlama_cpt":35.44,"SinLlama_Bactrianx_Instruct":24.72,"SinLlama_v02":40.35,"SinLlama_uc_instruct_cleaned":36.96},
  "language":         {"llama-3-8b":29.90,"SinLlama_v01":37.37,"SinLlama_cpt":32.22,"SinLlama_Bactrianx_Instruct":22.94,"SinLlama_v02":40.46,"SinLlama_uc_instruct_cleaned":32.73},
  "other":            {"llama-3-8b":28.90,"SinLlama_v01":35.40,"SinLlama_cpt":31.36,"SinLlama_Bactrianx_Instruct":23.67,"SinLlama_v02":38.56,"SinLlama_uc_instruct_cleaned":38.76},
  "social_science":   {"llama-3-8b":37.90,"SinLlama_v01":45.75,"SinLlama_cpt":44.33,"SinLlama_Bactrianx_Instruct":28.92,"SinLlama_v02":53.59,"SinLlama_uc_instruct_cleaned":48.58},
  "stem":             {"llama-3-8b":29.97,"SinLlama_v01":30.78,"SinLlama_cpt":29.15,"SinLlama_Bactrianx_Instruct":23.45,"SinLlama_v02":35.67,"SinLlama_uc_instruct_cleaned":37.62},
 },
 "english": {
  "stem":             {"llama-3-8b":55.53,"SinLlama_v01":41.62,"SinLlama_cpt":35.32,"SinLlama_Bactrianx_Instruct":30.02,"SinLlama_v02":45.13,"SinLlama_uc_instruct_cleaned":41.78},
  "humanities":       {"llama-3-8b":60.06,"SinLlama_v01":43.32,"SinLlama_cpt":36.47,"SinLlama_Bactrianx_Instruct":30.39,"SinLlama_v02":45.95,"SinLlama_uc_instruct_cleaned":47.52},
  "social_sciences":  {"llama-3-8b":76.02,"SinLlama_v01":55.18,"SinLlama_cpt":46.02,"SinLlama_Bactrianx_Instruct":35.03,"SinLlama_v02":60.06,"SinLlama_uc_instruct_cleaned":58.21},
  "other":            {"llama-3-8b":72.67,"SinLlama_v01":52.87,"SinLlama_cpt":44.32,"SinLlama_Bactrianx_Instruct":36.86,"SinLlama_v02":55.98,"SinLlama_uc_instruct_cleaned":54.97},
 },
}

PDF_DIFF_SI = {
  "easy":   {"llama-3-8b":38.54,"SinLlama_v01":45.46,"SinLlama_cpt":44.22,"SinLlama_Bactrianx_Instruct":27.14,"SinLlama_v02":50.70,"SinLlama_uc_instruct_cleaned":47.68},
  "medium": {"llama-3-8b":34.36,"SinLlama_v01":40.91,"SinLlama_cpt":38.83,"SinLlama_Bactrianx_Instruct":28.49,"SinLlama_v02":44.83,"SinLlama_uc_instruct_cleaned":41.39},
  "hard":   {"llama-3-8b":25.86,"SinLlama_v01":28.59,"SinLlama_cpt":26.22,"SinLlama_Bactrianx_Instruct":20.36,"SinLlama_v02":32.55,"SinLlama_uc_instruct_cleaned":30.34},
}


def load_valid(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("valid"):
                    rows.append(r)
    return rows


def find_sources(roots):
    """Fingerprint every prediction dump under `roots`; return lang->model->path."""
    cands = []
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            if "hellaswag" in dirpath:
                continue
            for fn in files:
                if fn.endswith("_predictions.jsonl"):
                    cands.append(os.path.join(dirpath, fn))

    found = {"sinhala": {}, "english": {}}
    for p in sorted(set(cands)):
        try:
            rows = load_valid(p)
        except Exception:
            continue
        n = len(rows)
        lang = "sinhala" if n == PDF["sinhala"]["total"] else (
               "english" if n == PDF["english"]["total"] else None)
        if lang is None:
            continue
        corr = sum(1 for r in rows if r["correct"])
        for m, want in PDF[lang]["correct"].items():
            if corr == want and m not in found[lang]:
                found[lang][m] = p
    return found


def pct(a, b):
    return 100.0 * a / b if b else 0.0


# The two evaluators dump different schemas: Sinhala uses "category" with
# 1-indexed pred/gold; English uses "domain" with 0-indexed pred/gold
# (pred 0 <-> pred_letter "A"). Normalise on read.
DOMAIN_KEY = {"sinhala": "category", "english": "domain"}
BASE = {"sinhala": 1, "english": 0}


def verify(lang, model, rows):
    """Re-derive the PDF's domain/difficulty cells; return list of failures."""
    fails = []
    dkey = DOMAIN_KEY[lang]
    by_dom = {}
    for r in rows:
        by_dom.setdefault(r[dkey], []).append(r)
    for dom, want in PDF_DOMAIN[lang].items():
        got = [r for r in by_dom.get(dom, [])]
        if not got:
            fails.append(f"{dom}: no rows")
            continue
        acc = pct(sum(1 for r in got if r["correct"]), len(got))
        if abs(acc - want[model]) > 0.015:
            fails.append(f"{dom}: got {acc:.2f} want {want[model]:.2f}")
    if lang == "sinhala":
        by_d = {}
        for r in rows:
            by_d.setdefault(r["difficulty"], []).append(r)
        for d, want in PDF_DIFF_SI.items():
            got = by_d.get(d, [])
            if not got:
                fails.append(f"difficulty {d}: no rows")
                continue
            acc = pct(sum(1 for r in got if r["correct"]), len(got))
            if abs(acc - want[model]) > 0.015:
                fails.append(f"difficulty {d}: got {acc:.2f} want {want[model]:.2f}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["benchmark/results", "results"])
    ap.add_argument("--out", default="mmlu_answer_distribution")
    args = ap.parse_args()

    src = find_sources(args.roots)

    missing = [(l, m) for l in ("sinhala", "english") for m in MODELS
               if m not in src[l]]
    if missing:
        print("FATAL: no PDF-matching prediction dump for:", file=sys.stderr)
        for l, m in missing:
            print(f"  {l}/{m}", file=sys.stderr)
        sys.exit(1)

    report = OrderedDict()
    print("=" * 78)
    print("PROVENANCE  (accepted only on exact valid-count + correct-count match)")
    print("=" * 78)
    for lang in ("sinhala", "english"):
        print(f"\n[{lang}]")
        for m in MODELS:
            print(f"  {m:32s} {src[lang][m]}")

    print("\n" + "=" * 78)
    print("VERIFICATION  (per-domain / per-difficulty cells re-derived vs PDF)")
    print("=" * 78)
    all_ok = True
    for lang in ("sinhala", "english"):
        for m in MODELS:
            rows = load_valid(src[lang][m])
            f = verify(lang, m, rows)
            if f:
                all_ok = False
                print(f"  MISMATCH {lang}/{m}:")
                for x in f:
                    print(f"      {x}")
            else:
                print(f"  ok  {lang:8s} {m}")
    if not all_ok:
        print("\nFATAL: verification failed; not emitting distributions.",
              file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 78)
    print("PREDICTED-ANSWER DISTRIBUTION")
    print("=" * 78)

    for lang, labels in (("sinhala", None), ("english", ["A", "B", "C", "D"])):
        rows0 = load_valid(src[lang][MODELS[0]])
        base = BASE[lang]
        nmax = max(r["n_choices"] for r in rows0)
        opts = list(range(base, base + nmax))
        lab = labels if labels else [str(o) for o in opts]

        comp = Counter(r["n_choices"] for r in rows0)
        print(f"\n### {lang.upper()}  (n={len(rows0)}; "
              f"n_choices composition: "
              f"{', '.join(f'{k}-way x{v}' for k, v in sorted(comp.items()))})")

        gold = Counter(r["gold"] for r in rows0)
        gold_vec = [gold.get(o, 0) for o in opts]
        print(f"  {'gold':34s} " + "/".join(str(x) for x in gold_vec))

        report.setdefault(lang, OrderedDict())
        report[lang]["_labels"] = lab
        report[lang]["_gold"] = gold_vec
        report[lang]["_n"] = len(rows0)

        for m in MODELS:
            rows = load_valid(src[lang][m])
            c = Counter(r["pred"] for r in rows)
            vec = [c.get(o, 0) for o in opts]
            top = max(range(len(vec)), key=lambda i: vec[i])
            print(f"  {m:34s} " + "/".join(str(x) for x in vec) +
                  f"    (mode {lab[top]} = {pct(vec[top], len(rows)):.1f}%)")
            report[lang][m] = {
                "path": src[lang][m],
                "n": len(rows),
                "correct": sum(1 for r in rows if r["correct"]),
                "dist": vec,
                "dist_pct": [round(pct(v, len(rows)), 2) for v in vec],
            }

        # Sinhala mixes 4/5/6-way items, so option 5 and 6 are not offered on
        # every item. Repeat the table on the 4-way subset for a clean read.
        if lang == "sinhala":
            sub = [r for r in rows0 if r["n_choices"] == 4]
            print(f"\n  -- 4-way subset only (n={len(sub)}) --")
            g4 = Counter(r["gold"] for r in sub)
            print(f"  {'gold':34s} " + "/".join(str(g4.get(o, 0)) for o in range(1, 5)))
            report[lang]["_gold_4way"] = [g4.get(o, 0) for o in range(1, 5)]
            report[lang]["_n_4way"] = len(sub)
            for m in MODELS:
                rows = [r for r in load_valid(src[lang][m]) if r["n_choices"] == 4]
                c = Counter(r["pred"] for r in rows)
                vec = [c.get(o, 0) for o in range(1, 5)]
                print(f"  {m:34s} " + "/".join(str(x) for x in vec))
                report[lang][m]["dist_4way"] = vec

    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}.json")


if __name__ == "__main__":
    main()
