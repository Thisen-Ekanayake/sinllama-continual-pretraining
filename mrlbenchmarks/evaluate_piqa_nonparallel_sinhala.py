#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate llama-3-8b / SinLlama_v01 / SinLlama_cpt / SinLlama_Bactrianx_Instruct
on the Sinhala half of global-piqa-nonparallel (mrlbenchmarks/global-piqa-nonparallel).

Method (see common.py docstring): zero-shot, 2-way MCQ, scored by highest
next-token probability among digits "1"/"2" after the Sinhala "පිළිතුර:" cue.
base/CPT/llama scored raw; the instruct model (--alpaca-models) is re-wrapped
in the Alpaca template it was SFT'd on. Models are loaded in bf16 by default
(--quant bf16/4bit/8bit).
"""
import os, json, time, argparse
import pandas as pd
import torch

import common as C


def build_records(rows):
    records = []
    for i, row in enumerate(rows):
        choices = [row["solution0"], row["solution1"]]
        gold = int(row["label"])
        valid = 0 <= gold < len(choices)
        cscore = row.get("approx_cultural_score")
        records.append(dict(
            idx=i, example_id=row.get("example_id", ""),
            cultural_score=f"score={cscore}",
            n_choices=len(choices), gold=gold, valid=valid,
            question=C.clean(row["prompt"]), choices=[C.clean(c) for c in choices],
            prompt=C.render_si(row["prompt"], choices, answer=None)))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="global-piqa-nonparallel/0.jsonl")
    ap.add_argument("--models", nargs="+",
                    default=["../models/llama-3-8b", "../models/SinLlama_v01",
                             "../models/SinLlama_cpt", "../models/SinLlama_Bactrianx_Instruct"])
    ap.add_argument("--alpaca-models", nargs="*", default=["SinLlama_Bactrianx_Instruct"],
                    help="model-name substrings to score in the Alpaca template")
    ap.add_argument("--out-dir", default="results_nonparallel_sinhala")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--quant", choices=["bf16", "4bit", "8bit"], default="bf16")
    ap.add_argument("--limit", type=int, default=0, help="cap #questions (0=all), for a smoke test")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--bucket", default="", help="GCS bucket/prefix (optional upload)")
    ap.add_argument("--combine-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_json(args.data, lines=True)
    rows = df.to_dict(orient="records")
    if args.limit:
        rows = rows[:args.limit]
    records0 = build_records(rows)
    n_valid = sum(r["valid"] for r in records0)
    print(f"Loaded {len(records0)} questions ({n_valid} valid) from {args.data}")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    GROUP_KEYS = {"by_cultural_score": lambda r: r["cultural_score"]}
    SECTIONS = [("Accuracy by cultural-specificity score", "by_cultural_score", None)]
    TITLE = "Global-PIQA (non-parallel) — Sinhala"

    if args.combine_only:
        all_metrics = {}
        for path in args.models:
            name = os.path.basename(path.rstrip("/"))
            mpath = os.path.join(args.out_dir, f"{name}_metrics.json")
            if os.path.exists(mpath):
                all_metrics[name] = json.load(open(mpath, encoding="utf-8"))
            else:
                print(f"  (combine) missing {mpath} — skipping")
        if not all_metrics:
            raise SystemExit("combine-only: no *_metrics.json found in --out-dir")
        meta = dict(bench="Global-PIQA non-parallel (Sinhala)", gpu=gpu, total=n_valid, quant=args.quant)
        rpath = os.path.join(args.out_dir, "results.md")
        C.write_results_md(rpath, TITLE, all_metrics, meta, [("By cultural score", "by_cultural_score")])
        if args.bucket:
            C.gcs_cp([rpath], args.bucket.rstrip("/") + "/")
        print(f"Wrote combined {rpath}")
        return

    all_metrics = {}
    for path in args.models:
        name = os.path.basename(path.rstrip("/"))
        use_alpaca = any(s in name for s in args.alpaca_models)
        fmt = "alpaca" if use_alpaca else "raw"
        mpath = os.path.join(args.out_dir, f"{name}_metrics.json")
        if args.skip_existing and os.path.exists(mpath):
            m = json.load(open(mpath, encoding="utf-8"))
            m["format"] = fmt
            m["quant"] = args.quant
            print(f"\n=== {name}: reusing cached metrics ({m['overall']['accuracy']:.2f}%) ===")
        else:
            print(f"\n=== Evaluating {name}  (prompt format: {fmt}) ===")
            t0 = time.time()
            model, tok = C.load_model(path, args.quant)
            records = [dict(r) for r in records0]
            if use_alpaca:
                for r in records:
                    r["prompt"] = C.to_alpaca(r["prompt"], C.block_to_alpaca_si)
            C.evaluate_digits(model, tok, records, args.batch_size, args.max_len)
            m = C.compute_metrics(records, GROUP_KEYS)
            m["format"] = fmt
            m["quant"] = args.quant
            print(f"  {name}: {m['overall']['accuracy']:.2f}%  "
                  f"({m['overall']['correct']}/{m['overall']['total']}) in {time.time()-t0:.0f}s")
            json.dump(m, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            C.write_predictions_csv(os.path.join(args.out_dir, f"{name}_predictions.csv"),
                                    records, extra_cols=[("cultural_score", lambda r: r["cultural_score"])])
            del model, tok
            C.free_model()

        all_metrics[name] = m
        C.write_txt(os.path.join(args.out_dir, f"{name}_results.txt"), TITLE, name, m, SECTIONS)
        C.write_md(os.path.join(args.out_dir, f"{name}_results.md"), TITLE, name, m, SECTIONS)
        if args.bucket:
            C.gcs_cp(C.per_model_files(args.out_dir, name), args.bucket.rstrip("/") + f"/{name}/")

    meta = dict(bench="Global-PIQA non-parallel (Sinhala)", gpu=gpu, total=n_valid)
    rpath = os.path.join(args.out_dir, "results.md")
    C.write_results_md(rpath, TITLE, all_metrics, meta, [("By cultural score", "by_cultural_score")])
    if args.bucket and len(args.models) > 1:
        C.gcs_cp([rpath], args.bucket.rstrip("/") + "/")
    print(f"\nWrote results to {args.out_dir}/  (results.md + per-model files)")


if __name__ == "__main__":
    main()
