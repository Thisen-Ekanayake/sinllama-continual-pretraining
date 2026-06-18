#!/usr/bin/env python3
"""
Quick inference test for SinLlama (merged bf16) with 4-bit quantization.

SinLlama = Meta-Llama-3-8B + Sinhala vocab extension (vocab 139,336),
continual-pretrained on a 10.7M-sentence Sinhala corpus. This is a BASE
completion model (not instruct-tuned) — so prompts are text to *continue*,
not chat turns.

4-bit (nf4) quantization brings the 16 GB bf16 checkpoint down to ~5 GB so it
fits on an 8 GB GPU.

Usage:
    python inference_sinllama.py                          # run built-in Sinhala prompts
    python inference_sinllama.py --prompt "ශ්‍රී ලංකාව"     # custom prompt
    python inference_sinllama.py --max-new-tokens 200 --temperature 0.7
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_DIR = "/ml/SinLlama_CPT/SinLlama_merged_bf16"

# Built-in Sinhala completion prompts (base model → continue the text)
DEFAULT_PROMPTS = [
    "ශ්‍රී ලංකාවේ අගනුවර වන්නේ ",
    "කෘතිම බුද්ධිය යනු ",
    "මම අද උදෑසන නැගිට ",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default=None,
                   help="Custom prompt (overrides the built-in set)")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--greedy", action="store_true",
                   help="Disable sampling (deterministic)")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected — 4-bit inference needs a GPU.")
    print(f"GPU  : {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # ── 4-bit quantization config (nf4 + double quant, bf16 compute) ──────────
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\nLoading tokenizer from {MODEL_DIR} …")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"  vocab size : {tok.vocab_size:,}  (+{len(tok) - tok.vocab_size} added)")

    print("Loading model in 4-bit (nf4) … this takes a minute on first load.")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory allocated: {mem:.2f} GB")

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        repetition_penalty=1.1,
    )

    print("\n" + "=" * 70)
    for i, prompt in enumerate(prompts, 1):
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        n_in = inputs["input_ids"].shape[1]
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        dt = time.time() - t0
        new_tokens = out[0][n_in:]
        completion = tok.decode(new_tokens, skip_special_tokens=True)
        tok_per_s = len(new_tokens) / dt if dt > 0 else 0

        print(f"\n[{i}] PROMPT : {prompt!r}")
        print(f"    OUTPUT : {prompt}{completion}")
        print(f"    ({len(new_tokens)} new tokens in {dt:.1f}s = {tok_per_s:.1f} tok/s)")
        print("-" * 70)


if __name__ == "__main__":
    main()
