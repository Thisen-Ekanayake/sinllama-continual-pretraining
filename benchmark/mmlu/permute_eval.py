#!/usr/bin/env python3
"""
Cyclic-permutation MMLU scoring — the position-bias-free arm.

WHY
---
MMLU is scored by reading P(next token = "A"/"B"/... or "1"/"2"/...) and taking
the argmax. That number confounds two things: whether the model knows the
answer, and which *slot* it likes to answer in. On SinhalaMMLU the gold answer
is near-uniform across options (22/24/25/22%), but SinLlama_v02 answers option 3
30% of the time and SinLlama_uc_instruct_cleaned answers option 1 29% of the
time. Breaking accuracy down by which option happens to be correct shows the
consequence: uc_instruct_cleaned beats v02 by +9.4pp on items whose answer is
option 1 and loses on every other option. That is the signature of a shifted
prior, not of lost knowledge.

WHAT THIS DOES
--------------
Each question is scored n times, once per cyclic rotation of its option list,
so every answer *text* appears in every *position* exactly once. The scores for
a given answer text are averaged across its n positions, and the argmax is taken
over answer texts rather than over slots. Any preference the model has for a
position is applied equally to all answers and cancels out by construction.

This is the rigorous complement to `--calibrate` (contextual calibration), which
estimates the prior once from a content-free prompt and subtracts it. Calibration
costs one extra forward pass; permutation costs n times the inference but makes
no assumption that the prior is content-independent. Run both: agreement between
them is strong evidence the debiasing is sound.

Rotation r maps the option originally at index i to position (i + r) % n, so the
score of original option i under rotation r is read from slot (i + r) % n.

OUTPUT
------
Writes the same `<model>_metrics.json` / `<model>_predictions.jsonl` shape as the
normal evaluators, so every existing downstream tool (collect_report.py, the
answer-distribution scripts, the .tex builders) works unchanged. Predictions
additionally carry `pred_raw` — the argmax of the *un*-rotated pass, i.e. exactly
what the standard evaluator would have produced — so the two arms are paired
item-by-item and a McNemar test is available without a second run.

USAGE
-----
  python permute_eval.py --lang sinhala \
      --models models/SinLlama_v02 models/SinLlama_uc_instruct_cleaned \
      --data-root benchmark/mmlu/SinhalaMMLU \
      --out-dir benchmark/mmlu/SinhalaMMLU_results_permuted

  python permute_eval.py --lang english \
      --models models/SinLlama_v02 models/SinLlama_uc_instruct_cleaned \
      --data-root benchmark/mmlu/english_mmlu \
      --out-dir benchmark/mmlu/EnglishMMLU_results_permuted
"""
import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate_sinhala_mmlu as SI
import evaluate_english_mmlu as EN


# ----------------------------------------------------------------------------- #
# Language adapter — the two evaluators differ in option labels (digits vs
# letters), answer indexing (1-based vs 0-based) and prompt rendering, so each
# language supplies the four hooks the permutation loop needs.
# ----------------------------------------------------------------------------- #
class SinhalaAdapter:
    name = "sinhala"
    mod = SI
    base = 1                      # options are 1..n, gold is 1-based

    def __init__(self, args):
        self.template = SI.load_template(args.prompt_file)

    def build(self, args):
        records, missing = SI.build_examples(args.data_root, self.template,
                                             args.kshot)
        if missing:
            print(f"WARNING: no few-shot match for: {sorted(missing)}")
        if args.limit:
            # SI.build_examples has no limit param; main() caps per TEST file
            # after the fact, so mirror that exactly
            seen, kept = defaultdict(int), []
            for r in records:
                if seen[r["file"]] < args.limit:
                    kept.append(r)
                    seen[r["file"]] += 1
            records = kept
        return records

    def option_ids(self, tok):
        # render() already emits the space, so these are bare digit tokens
        return check_distinct([tok.encode(str(d), add_special_tokens=False)[0]
                               for d in range(1, 10)], "answer digits 1-9")

    def render_test(self, rec, choices):
        return SI.render(self.template, rec["subject_original"], rec["question"],
                         choices, answer=None)

    def metrics(self, records):
        return SI.compute_metrics(records)

    def pred_fields(self):
        return ("file", "difficulty", "category", "subject_display",
                "n_choices", "gold", "valid", "pred", "correct")


class EnglishAdapter:
    name = "english"
    mod = EN
    base = 0                      # options are A..D, gold is 0-based

    def __init__(self, args):
        self.canonical = args.canonical

    def build(self, args):
        return EN.build_examples(args.data_root, args.kshot, args.limit,
                                 args.canonical)

    def option_ids(self, tok):
        # the leading space is part of the letter token here
        return check_distinct([tok.encode(" " + L, add_special_tokens=False)[0]
                               for L in EN.LETTERS], "option letters A-D")

    def render_test(self, rec, choices):
        if self.canonical:
            return EN.render_canonical(rec["question"], choices, None)
        return EN.render(rec["subject_pretty"], rec["question"], choices, None)

    def metrics(self, records):
        return EN.compute_metrics(records)

    def pred_fields(self):
        return ("subject", "domain", "n_choices", "gold", "gold_letter",
                "valid", "pred", "pred_letter", "correct")


def check_distinct(ids, what):
    """Option scoring reads the FIRST token of each option label's encoding.
    That is only valid when the labels are single tokens; with a SentencePiece
    tokenizer they all share a leading-space token and collapse to one id,
    which would silently make every option score identical."""
    if len(set(ids)) != len(ids):
        raise SystemExit(
            f"FATAL: {what} do not map to distinct single tokens with this "
            f"tokenizer (got {ids}). Option-probability scoring is invalid here.")
    return ids


def rotate(choices, r):
    """Rotation r puts the option originally at index i into position (i+r)%n."""
    n = len(choices)
    out = [None] * n
    for i, c in enumerate(choices):
        out[(i + r) % n] = c
    return out


def build_jobs(adapter, records, wrap, rotations):
    """Flatten (record, rotation) into a list of prompts to score."""
    jobs = []
    for idx, r in enumerate(records):
        if not r.get("valid"):
            continue
        n = r["n_choices"]
        nrot = n if rotations <= 0 else min(rotations, n)
        for rot in range(nrot):
            if rot == 0:
                prompt = r["prompt"]          # identical to the standard arm
            else:
                block = adapter.render_test(r, rotate(r["choices"], rot))
                prompt = ((r["shot_prefix"] + "\n\n" + block)
                          if r["shot_prefix"] else block)
                prompt = wrap(prompt)
            jobs.append(dict(idx=idx, rot=rot, n=n, prompt=prompt))
    return jobs


@torch.no_grad()
def _score_jobs(model, tok, chunk, option_ids, max_len, out):
    """Score a chunk of jobs; accumulate per-original-option log-probs in `out`.

    Mirrors the OOM back-off in the standard evaluators: split the batch, then
    shrink max_len, before giving up on a single item."""
    enc = res = None
    try:
        enc = tok([j["prompt"] for j in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(model.device)
        try:
            res = model(**enc, logits_to_keep=1)
        except TypeError:
            res = model(**enc)
        logits = res.logits[:, -1, :].float().cpu()
        for b, j in enumerate(chunk):
            n, rot = j["n"], j["rot"]
            cand = option_ids[:n]
            lp = torch.log_softmax(logits[b, cand], dim=-1)
            acc = out[j["idx"]]
            for i in range(n):                 # original option i -> slot (i+rot)%n
                acc["sum"][i] += float(lp[(i + rot) % n])
            acc["k"] += 1
            if rot == 0:                       # the un-rotated pass == standard arm
                acc["raw"] = int(torch.argmax(lp).item())
    except torch.cuda.OutOfMemoryError:
        enc = res = None
        torch.cuda.empty_cache()
        if len(chunk) == 1:
            if max_len <= 768:
                out[chunk[0]["idx"]]["failed"] = True
            else:
                _score_jobs(model, tok, chunk, option_ids, 768, out)
        else:
            mid = len(chunk) // 2
            _score_jobs(model, tok, chunk[:mid], option_ids, max_len, out)
            _score_jobs(model, tok, chunk[mid:], option_ids, max_len, out)


def make_batches(jobs, tok, max_bs, max_len, budget):
    """Group jobs into batches under a B*S^2 memory budget.

    EAGER attention materialises a [B, heads, S, S] score tensor, so peak memory
    grows with batch size times the square of the batch's longest sequence -- not
    linearly in tokens. Batching purely by count is therefore unsafe once the
    batch happens to contain long prompts: at B=16, S=4096, 32 heads, bf16 that
    single tensor is ~17 GB, which aborts the HIP runtime outright rather than
    raising a catchable torch OOM (observed: SIGABRT, no Python traceback, so the
    recursive split in _score_jobs never runs).

    Sorting longest-first for padding efficiency makes this *worse*, because it
    puts all the longest prompts in the same batch. So: sort by length, then cut
    batches whenever B*S^2 would exceed the budget. Short prompts still batch
    densely up to max_bs; only the long tail is throttled."""
    lens = []
    B = 512
    for i in range(0, len(jobs), B):
        enc = tok([j["prompt"] for j in jobs[i:i + B]], add_special_tokens=True)
        lens.extend(min(len(x), max_len) for x in enc["input_ids"])
    order = sorted(range(len(jobs)), key=lambda i: -lens[i])

    batches, cur, cur_max = [], [], 0
    for i in order:
        nmax = max(cur_max, lens[i])
        if cur and ((len(cur) + 1) * nmax * nmax > budget or len(cur) >= max_bs):
            batches.append(cur)
            cur, cur_max = [jobs[i]], lens[i]
        else:
            cur.append(jobs[i])
            cur_max = nmax
    if cur:
        batches.append(cur)
    longest = max(lens) if lens else 0
    print(f"  {len(batches)} batches (max seq {longest}, "
          f"largest batch {max(len(b) for b in batches) if batches else 0})")
    return batches


def run_model(adapter, path, records0, args):
    name = os.path.basename(path.rstrip("/"))
    use_alpaca = any(s in name for s in args.alpaca_models)
    use_chat = (not use_alpaca) and any(s in name for s in args.chat_models)
    fmt = "alpaca" if use_alpaca else ("chat" if use_chat else "raw")
    wrap = (adapter.mod.to_alpaca if use_alpaca else
            adapter.mod.to_chat if use_chat else (lambda s: s))

    print(f"\n=== {name}  (format: {fmt}, cyclic permutation) ===")
    t0 = time.time()
    records = [dict(r) for r in records0]
    if use_alpaca or use_chat:
        for r in records:
            r["prompt"] = wrap(r["prompt"])

    jobs = build_jobs(adapter, records, wrap, args.rotations)
    print(f"  {len(jobs)} forward passes "
          f"({sum(1 for r in records if r.get('valid'))} items)")

    model, tok = adapter.mod.load_model(path)
    option_ids = adapter.option_ids(tok)
    out = defaultdict(lambda: {"sum": [0.0] * 8, "k": 0, "raw": None,
                               "failed": False})

    batches = make_batches(jobs, tok, args.batch_size, args.max_len,
                           args.attn_budget)
    for b in tqdm(batches, desc="  permute"):
        _score_jobs(model, tok, b, option_ids, args.max_len, out)

    n_failed = 0
    for idx, r in enumerate(records):
        if not r.get("valid"):
            continue
        acc = out[idx]
        n = r["n_choices"]
        if acc["failed"] or acc["k"] == 0:
            r["pred"] = -1
            r["pred_raw"] = -1
            r["correct"] = False
            n_failed += 1
            continue
        means = [acc["sum"][i] / acc["k"] for i in range(n)]
        r["pred"] = int(max(range(n), key=lambda i: means[i])) + adapter.base
        r["pred_raw"] = (acc["raw"] + adapter.base
                         if acc["raw"] is not None else -1)
        r["correct"] = (r["pred"] == r["gold"])
        if adapter.base == 0:
            r["pred_letter"] = EN.LETTERS[r["pred"]] if r["pred"] >= 0 else "?"
    if n_failed:
        print(f"  WARNING: {n_failed} items failed to score (OOM)")

    m = adapter.metrics(records)
    m["format"] = fmt
    m["calibrated"] = False
    m["permuted"] = True
    m["rotations"] = ("full" if args.rotations <= 0 else args.rotations)

    # paired raw-vs-permuted summary, free because rot==0 is the standard arm
    ev = [r for r in records if r.get("valid") and r["pred"] != -1]
    raw_corr = sum(1 for r in ev if r["pred_raw"] == r["gold"])
    b01 = sum(1 for r in ev if r["pred_raw"] == r["gold"] and not r["correct"])
    b10 = sum(1 for r in ev if r["pred_raw"] != r["gold"] and r["correct"])
    m["raw_arm"] = dict(correct=raw_corr, total=len(ev),
                        accuracy=100.0 * raw_corr / len(ev) if ev else 0.0)
    m["paired_vs_raw"] = dict(only_raw_correct=b01, only_permuted_correct=b10)

    print(f"  raw (rot 0)      : {m['raw_arm']['accuracy']:.2f}%  "
          f"({raw_corr}/{len(ev)})")
    print(f"  cyclic-permuted  : {m['overall']['accuracy']:.2f}%  "
          f"({m['overall']['correct']}/{m['overall']['total']})   "
          f"[{b01} lost / {b10} gained]  in {time.time()-t0:.0f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(m, open(os.path.join(args.out_dir, f"{name}_metrics.json"), "w",
                      encoding="utf-8"), ensure_ascii=False, indent=2)
    fields = adapter.pred_fields()
    with open(os.path.join(args.out_dir, f"{name}_predictions.jsonl"), "w",
              encoding="utf-8") as fh:
        for r in records:
            row = {k: r.get(k) for k in fields}
            row["pred_raw"] = r.get("pred_raw")
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=("sinhala", "english"))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt-file", default="benchmark/prompts/mmlu_sinhala.txt",
                    help="sinhala only")
    ap.add_argument("--canonical", action="store_true", help="english only")
    ap.add_argument("--kshot", type=int, default=None,
                    help="default: 3 for sinhala, 5 for english")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attn-budget", type=float, default=4.0e7,
                    help="max batch_size * seq_len^2 per batch. Eager attention "
                         "allocates [B, heads, S, S], so this -- not token count "
                         "-- is what bounds peak memory. Default allows B=16 at "
                         "S=1580, B=4 at S=3162, B=2 at S=4096.")
    ap.add_argument("--rotations", type=int, default=0,
                    help="0 = full cyclic (n rotations per item, the default "
                         "and the only fully unbiased setting); a smaller "
                         "number trades rigour for inference cost")
    ap.add_argument("--alpaca-models", nargs="*", default=[])
    ap.add_argument("--chat-models", nargs="*", default=[])
    args = ap.parse_args()

    if args.kshot is None:
        args.kshot = 3 if args.lang == "sinhala" else 5

    adapter = (SinhalaAdapter(args) if args.lang == "sinhala"
               else EnglishAdapter(args))
    records0 = adapter.build(args)
    n_valid = sum(1 for r in records0 if r.get("valid"))
    print(f"Loaded {len(records0)} questions ({n_valid} valid).")

    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.models:
        run_model(adapter, path, records0, args)
    print(f"\nWrote results to {args.out_dir}/")


if __name__ == "__main__":
    main()
