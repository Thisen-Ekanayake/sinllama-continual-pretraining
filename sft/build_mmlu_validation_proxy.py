#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render `data/mmlu-sinhala/*/validation-*.json` into an internal monitoring
proxy for the MMLU-Sinhala SFT run.

Native SinhalaMMLU has no validation split separate from its TEST set, so it
can't be used for mid-training checkpoint selection without biasing the final
benchmark number. This translated-dataset validation split fills that role
purely as an internal signal — it is never touched for final reporting, and
`moral_scenarios` (degenerate in the Sinhala MT, see sft/build_mmlu_sft.py) is
excluded.

No leak-check against anything (there is nothing to check leakage against)
and no option-shuffle (this is a monitoring signal, not a reported score).
Prompts render through [SUBJECT] = the same "විෂය" fallback build_mmlu_sft.py
uses for auxiliary_train — the model only ever sees that fallback subject
during training, so the monitoring signal must match that, not the real
per-item subject.
"""
import argparse
import glob
import json
import os
import collections

from build_mmlu_sft import render, clean, load_records, degenerate

EXCLUDED_SUBJECTS = {"moral_scenarios"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/mmlu-sinhala")
    ap.add_argument("--out", default="data/mmlu_sft/aux_val_proxy.jsonl")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_root, "*", "validation-*.json")))
    if not files:
        raise SystemExit(f"no validation-*.json found under {args.data_root}/*/")

    drop = collections.Counter()
    kept = []

    for f in files:
        subject = os.path.basename(os.path.dirname(f))
        if subject in EXCLUDED_SUBJECTS:
            drop["excluded_subject"] += len(load_records(f))
            continue
        for r in load_records(f):
            q = r.get("question")
            ch = r.get("choices")
            a = r.get("answer")

            if not isinstance(ch, list) or len(ch) != 4 or not isinstance(a, int) \
                    or not (0 <= a < len(ch)):
                drop["malformed"] += 1
                continue
            if not clean(q) or any(not clean(c) for c in ch):
                drop["empty_field"] += 1
                continue

            opts = [clean(c) for c in ch]
            if len(set(opts)) < 4:
                drop["collapsed_choices"] += 1
                continue

            if degenerate(clean(q)):
                drop["degenerate_text"] += 1
                continue

            kept.append({"question": clean(q), "choices": opts, "answer": a,
                        "subject": subject})

    prompts = [render(r["question"], r["choices"]) for r in kept]
    targets = [str(r["answer"] + 1) for r in kept]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r, p, c in zip(kept, prompts, targets):
            f.write(json.dumps({**r, "prompt": p, "completion": c},
                                ensure_ascii=False) + "\n")

    print("--- filter report ---")
    for k in ("excluded_subject", "malformed", "empty_field",
              "collapsed_choices", "degenerate_text"):
        print(f"  - {k:<20}{drop[k]:>6}")
    print(f"  = kept              {len(kept):>6}")
    print(f"\nwrote {args.out}  ({os.path.getsize(args.out)/1e3:.1f} KB)")


if __name__ == "__main__":
    main()
