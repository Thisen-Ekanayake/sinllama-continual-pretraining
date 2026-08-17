#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate llama-3-8b / SinLlama_v01 / SinLlama_cpt / SinLlama_Bactrianx_Instruct
on the Sinhala half of global-piqa-parallel (mrlbenchmarks/global-piqa-parallel).

Method (see common.py docstring for full detail): few-shot (default 8, balanced
across the 4 answer keys and held out from the test set — pass --kshot 0 for a
zero-shot run), 4-way MCQ, scored by highest next-token probability among digits
"1".."4" after the answer cue. base/CPT/llama scored raw; the instruct model
(--alpaca-models) is re-wrapped in the Alpaca template it was SFT'd on. Models
are loaded in bf16 by default (--quant bf16/4bit/8bit).
"""
import os, json, time, argparse
import pandas as pd
import torch

import benchmark.mrlbenchmarks.common as C

N_OPTS = 4


def build_records(rows, kshot, seed):
    golds = [int(r["label"]) for r in rows]
    per_key = max(1, kshot // N_OPTS)
    shot_idx = C.pick_fewshot(golds, per_key, seed=seed) if kshot else []
    shot_set = set(shot_idx)

    shot_blocks = []
    for i in shot_idx:
        row = rows[i]
        choices = [row["solution0"], row["solution1"], row["solution2"], row["solution3"]]
        shot_blocks.append(C.render_si(row["prompt"], choices, answer=int(row["label"]) + 1))
    prefix = "\n\n".join(shot_blocks)

    records = []
    for i, row in enumerate(rows):
        if i in shot_set:
            continue                                    # exemplars are never scored
        choices = [row["solution0"], row["solution1"], row["solution2"], row["solution3"]]
        gold = int(row["label"])
        valid = 0 <= gold < len(choices)
        test_block = C.render_si(row["prompt"], choices, answer=None)
        prompt = (prefix + "\n\n" + test_block) if prefix else test_block
        records.append(dict(
            idx=i, example_id=row.get("example_id", ""),
            category=row.get("categories") or "uncategorized",
            n_choices=len(choices), gold=gold, valid=valid,
            question=C.clean(row["prompt"]), choices=[C.clean(c) for c in choices],
            prompt=prompt))
    return records, shot_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="global-piqa-parallel/0.jsonl")
    ap.add_argument("--models", nargs="+",
                    default=["../models/llama-3-8b", "../models/SinLlama_v01",
                             "../models/SinLlama_cpt", "../models/SinLlama_Bactrianx_Instruct"])
    ap.add_argument("--alpaca-models", nargs="*", default=["SinLlama_Bactrianx_Instruct"],
                    help="model-name substrings to score in the Alpaca template")
    ap.add_argument("--chat-models", nargs="*", default=[],
                    help="model-name substrings to score in the UltraChat "
                         "### User/### Assistant template")
    ap.add_argument("--out-dir", default="results_parallel_sinhala")
    ap.add_argument("--kshot", type=int, default=8, help="# balanced few-shot exemplars, held out (0 = zero-shot)")
    ap.add_argument("--seed", type=int, default=0, help="exemplar selection seed")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--quant", choices=["bf16", "4bit", "8bit"], default="bf16")
    ap.add_argument("--limit", type=int, default=0, help="cap #test questions (0=all), for a smoke test")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--bucket", default="", help="GCS bucket/prefix (optional upload)")
    ap.add_argument("--combine-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_json(args.data, lines=True)
    rows = df.to_dict(orient="records")
    records0, shot_idx = build_records(rows, args.kshot, args.seed)
    if args.limit:
        records0 = records0[:args.limit]
    n_valid = sum(r["valid"] for r in records0)
    print(f"Loaded {len(records0)} test questions ({n_valid} valid) from {args.data}; "
          f"{len(shot_idx)} exemplars held out: {shot_idx}")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    GROUP_KEYS = {"by_category": lambda r: r["category"]}
    SECTIONS = [("Accuracy by category", "by_category", None)]
    TITLE = "Global-PIQA (parallel) — Sinhala"

    def make_meta():
        return dict(bench="Global-PIQA parallel (Sinhala)", gpu=gpu, total=n_valid,
                    quant=args.quant, kshot=args.kshot, n_shots=len(shot_idx))

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
        rpath = os.path.join(args.out_dir, "results.md")
        C.write_results_md(rpath, TITLE, all_metrics, make_meta(), [("By category", "by_category")])
        if args.bucket:
            C.gcs_cp([rpath], args.bucket.rstrip("/") + "/")
        print(f"Wrote combined {rpath}")
        return

    all_metrics = {}
    for path in args.models:
        name = os.path.basename(path.rstrip("/"))
        use_alpaca = any(s in name for s in args.alpaca_models)
        use_chat = (not use_alpaca) and any(s in name for s in args.chat_models)
        fmt = "alpaca" if use_alpaca else ("chat" if use_chat else "raw")
        mpath = os.path.join(args.out_dir, f"{name}_metrics.json")
        if args.skip_existing and os.path.exists(mpath):
            m = json.load(open(mpath, encoding="utf-8"))
            m["format"] = fmt
            m["quant"] = args.quant
            m["kshot"] = args.kshot
            print(f"\n=== {name}: reusing cached metrics ({m['overall']['accuracy']:.2f}%) ===")
        else:
            print(f"\n=== Evaluating {name}  (prompt format: {fmt}, {args.kshot}-shot) ===")
            t0 = time.time()
            model, tok = C.load_model(path, args.quant)
            records = [dict(r) for r in records0]
            if use_alpaca:
                for r in records:
                    r["prompt"] = C.to_alpaca(r["prompt"], C.block_to_alpaca_si)
            elif use_chat:
                for r in records:
                    r["prompt"] = C.to_chat(r["prompt"], C.block_to_chat_si)
            C.evaluate_digits(model, tok, records, args.batch_size, args.max_len)
            m = C.compute_metrics(records, GROUP_KEYS)
            m["format"] = fmt
            m["quant"] = args.quant
            m["kshot"] = args.kshot
            m["fewshot_indices"] = shot_idx
            print(f"  {name}: {m['overall']['accuracy']:.2f}%  "
                  f"({m['overall']['correct']}/{m['overall']['total']}) in {time.time()-t0:.0f}s")
            json.dump(m, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            C.write_predictions_csv(os.path.join(args.out_dir, f"{name}_predictions.csv"),
                                    records, extra_cols=[("category", lambda r: r["category"])])
            del model, tok
            C.free_model()

        all_metrics[name] = m
        C.write_txt(os.path.join(args.out_dir, f"{name}_results.txt"), TITLE, name, m, SECTIONS)
        C.write_md(os.path.join(args.out_dir, f"{name}_results.md"), TITLE, name, m, SECTIONS)
        if args.bucket:
            C.gcs_cp(C.per_model_files(args.out_dir, name), args.bucket.rstrip("/") + f"/{name}/")

    rpath = os.path.join(args.out_dir, "results.md")
    C.write_results_md(rpath, TITLE, all_metrics, make_meta(), [("By category", "by_category")])
    if args.bucket and len(args.models) > 1:
        C.gcs_cp([rpath], args.bucket.rstrip("/") + "/")
    print(f"\nWrote results to {args.out_dir}/  (results.md + per-model files)")


if __name__ == "__main__":
    main()
