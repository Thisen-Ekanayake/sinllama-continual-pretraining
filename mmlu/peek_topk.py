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
    python peek_topk.py --models /path/to/SinLlama_Backtrianx_instruct \
                        --format alpaca          # test the instruct SFT format
"""
import argparse, os, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_sinhala_mmlu import load_template, build_examples

try:
    from tqdm import tqdm
except Exception:                                    # pragma: no cover
    def tqdm(x, **k): return x

# repo root = parent of the directory holding this script (<repo>/mmlu/peek_topk.py)
DEFAULT_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The Alpaca instruction template SinLlama_Backtrianx_instruct was SFT'd on
# (see 158_P_Finetunning_BacterianX.ipynb). The instruct model expects THIS
# scaffolding, not a chat template, so --format alpaca re-wraps each MMLU block
# in it to test whether the "raw prompt" was making the instruct model unsure.
ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}")


def block_to_alpaca(block):
    """Re-wrap one rendered MMLU block into the Alpaca template.

    A block looks like:
        <task instruction line>
        ප්‍රශ්නය: <question>
        <options>
        පිළිතුර: <answer or trailing space>
    -> instruction = task line, input = question+options, response = the
    "පිළිතුර: ..." tail (trailing space preserved for the test block)."""
    before, cue, after = block.rpartition("පිළිතුර:")   # cue only at answer slot
    instr, _, qbody = before.partition("\nප්‍රශ්නය:")
    instruction = instr.strip()
    input_text = ("ප්‍රශ්නය:" + qbody).strip()
    response = cue + after                               # "පිළිතුර: " or "පිළිතුර: 3"
    return ALPACA_PROMPT.format(instruction, input_text, response)


def to_alpaca(raw_prompt):
    """Re-wrap every block of a (few-shot) prompt. Blocks are joined by blank
    lines in build_examples; clean() strips internal newlines so the split is
    safe. The trailing space of the final (test) block is preserved."""
    return "\n\n".join(block_to_alpaca(b) for b in raw_prompt.split("\n\n"))


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
    ap.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    ap.add_argument("--data-root", default=None,
                    help="SinhalaMMLU dir (default: <project-dir>/SinhalaMMLU)")
    ap.add_argument("--prompt-file", default="prompt.txt")
    ap.add_argument("--models", nargs="+", default=None,
                    help="override the default 4-model list")
    ap.add_argument("--kshot", type=int, default=3)
    ap.add_argument("--n-prompts", type=int, default=3,
                    help="how many sample test questions to inspect")
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"],
                    default="all", help="restrict sample to one difficulty")
    ap.add_argument("--n-choices", type=int, default=0,
                    help="only inspect questions with this many options "
                         "(e.g. 5 for A/L; 0 = any)")
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--max-len", type=int, default=3500)
    ap.add_argument("--format", choices=["raw", "alpaca"], default="raw",
                    help="raw = plain MMLU prompt; alpaca = wrap in the "
                         "### Instruction/Input/Response template the instruct "
                         "model was fine-tuned on")
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
    valid = [r for r in records if r["valid"]]
    if not valid:
        raise SystemExit(
            f"No questions parsed from {data_root} "
            f"(expected {data_root}/TEST/<easy|medium|hard>/*.json). "
            f"Pass --data-root or --project-dir if the dataset is elsewhere.")
    if args.difficulty != "all":
        valid = [r for r in valid if r["difficulty"] == args.difficulty]
    if args.n_choices:
        valid = [r for r in valid if r["n_choices"] == args.n_choices]
    valid = valid[:args.n_prompts]
    if not valid:
        raise SystemExit(f"No questions match difficulty={args.difficulty} "
                         f"n_choices={args.n_choices or 'any'}.")
    max_choices = max(r["n_choices"] for r in valid)
    print(f"Inspecting {len(valid)} prompt(s) "
          f"(difficulty={args.difficulty}, n_choices={args.n_choices or 'any'}, "
          f"format={args.format}); "
          f"showing top-{args.topk} next tokens at the answer position.\n")

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
        for d in [str(x) for x in range(1, max_choices + 1)]:
            print(f"  digit {d!r} -> {tok.encode(d, add_special_tokens=False)}"
                  f"    ' {d}' -> {tok.encode(' ' + d, add_special_tokens=False)}")

        for qi, r in enumerate(valid):
            n, gold = r["n_choices"], r["gold"]
            digits = {str(x) for x in range(1, n + 1)}
            prompt = to_alpaca(r["prompt"]) if args.format == "alpaca" else r["prompt"]
            rows = topk_next(model, tok, prompt, args.topk, args.max_len)
            print(f"\n  --- prompt {qi} | subject={r['subject_display']} "
                  f"| n_choices={n} | gold={gold} ---")
            print(f"      prompt tail: ...{prompt[-50:]!r}")
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
