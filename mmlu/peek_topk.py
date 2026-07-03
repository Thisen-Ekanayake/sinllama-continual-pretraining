#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Print the top-k next-token predictions for each model at the SinhalaMMLU
answer position (immediately after "පිළිතුර: ").

Why this exists
---------------
evaluate_sinhala_mmlu.py scores an answer by argmax over the *digit* tokens
1..N of the next-token logits. If a model's highest-probability tokens at that
position are NOT digits — e.g. a newline, <|eot_id|>, "The", or a chat header —
it will score near-random no matter how capable it is. That is the classic
base-vs-instruct evaluation artifact and the first thing to rule out when a
more-capable model (instruct) scores *worse* than the base model.

This script shows, per model, exactly which tokens win at that position, plus a
vocab/tokenizer sanity line and the digit tokenization (leading-space check).

Model paths mirror mmlu/run_eval.sh. Run from the mmlu/ directory:

    python peek_topk.py                          # all 4 models, 3 sample prompts
    python peek_topk.py --n-prompts 5 --topk 20
    python peek_topk.py --models /path/to/SinLlama_Backtrianx_instruct
"""
import argparse, os, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_sinhala_mmlu import load_template, build_examples

try:
    from tqdm import tqdm
except Exception:                                    # pragma: no cover
    def tqdm(x, **k): return x


def load(path):
    """Load exactly like the eval does (bf16, left padding/truncation)."""
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa").eval()
    return model, tok


@torch.no_grad()
def topk_next(model, tok, prompt, k, max_len):
    enc = tok(prompt, return_tensors="pt", truncation=True,
              max_length=max_len).to(model.device)
    logits = model(**enc).logits[0, -1].float()          # next-token logits
    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(logits, k)
    return [(int(i), tok.decode([int(i)]), float(v), float(probs[int(i)]))
            for v, i in zip(vals, idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir",
                    default="/sinllama-continual-pretraining")
    ap.add_argument("--data-root", default=None,
                    help="SinhalaMMLU dir (default: <project-dir>/SinhalaMMLU)")
    ap.add_argument("--prompt-file", default="prompt.txt")
    ap.add_argument("--models", nargs="+", default=None,
                    help="override the default 4-model list")
    ap.add_argument("--kshot", type=int, default=3)
    ap.add_argument("--n-prompts", type=int, default=3,
                    help="how many sample test questions to inspect")
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--max-len", type=int, default=3500)
    args = ap.parse_args()

    proj = args.project_dir
    data_root = args.data_root or os.path.join(proj, "SinhalaMMLU")
    models = args.models or [
        os.path.join(proj, "SinLlama_v01"),
        os.path.join(proj, "SinLlama_cpt_merged"),
        os.path.join(proj, "SinLlama_Backtrianx_instruct"),
        os.path.join(proj, "llama-3-8b"),
    ]

    # build real 3-shot prompts the same way the eval does
    template = load_template(args.prompt_file)
    records, _ = build_examples(data_root, template, args.kshot)
    valid = [r for r in records if r["valid"]][:args.n_prompts]
    if not valid:
        raise SystemExit(f"No valid prompts found under {data_root}")
    print(f"Inspecting {len(valid)} prompt(s); showing top-{args.topk} next "
          f"tokens at the answer position.\n")

    for mpath in tqdm(models, desc="Models", unit="model"):
        name = os.path.basename(mpath.rstrip("/"))
        print("=" * 80)
        print(f"MODEL: {name}   ({mpath})")
        if not os.path.exists(mpath):
            print("  !! path does not exist — skipping")
            continue
        model, tok = load(mpath)
        print(f"  vocab_size={model.config.vocab_size}  len(tok)={len(tok)}  "
              f"name_or_path={getattr(model.config, '_name_or_path', None)}")
        # digit tokenization: does the model want a bare '3' or a leading ' 3'?
        for d in ("1", "2", "3", "4"):
            print(f"  digit {d!r} -> {tok.encode(d, add_special_tokens=False)}"
                  f"    ' {d}' -> {tok.encode(' ' + d, add_special_tokens=False)}")

        for qi, r in enumerate(valid):
            n, gold = r["n_choices"], r["gold"]
            digits = {str(x) for x in range(1, n + 1)}
            rows = topk_next(model, tok, r["prompt"], args.topk, args.max_len)
            print(f"\n  --- prompt {qi} | subject={r['subject_display']} "
                  f"| n_choices={n} | gold={gold} ---")
            print(f"      prompt tail: ...{r['prompt'][-50:]!r}")
            for rank, (tid, dec, lg, pr) in enumerate(rows):
                mark = "  <-- answer digit" if dec.strip() in digits else ""
                print(f"      {rank:2d}. logit={lg:8.2f}  p={pr:6.3f}  "
                      f"id={tid:<7} {dec!r}{mark}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
