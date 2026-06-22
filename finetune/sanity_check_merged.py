"""
Quick health check for the merged SinLlama model BEFORE launching a long finetune.

Confirms two things:
  1. The model generates coherent Sinhala (the vocab-extension merge wasn't corrupted).
  2. The answer-token cross-entropy for a writing-style prompt is sane, not ~ln(V).

Usage:
    python finetune/sanity_check_merged.py --model_name_or_path /dev/shm/SinLlama_cpt_merged
"""
import argparse
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model_name_or_path", default="/dev/shm/SinLlama_cpt_merged")
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path, torch_dtype=torch.bfloat16,
    device_map={"": 0}, attn_implementation="sdpa")
model.eval()

V = model.config.vocab_size
print(f"vocab_size={V}  uniform-random CE = ln(V) = {math.log(V):.2f}")

# --- 1) Free-form Sinhala generation: should be fluent, not gibberish/repetition.
prompt = "ශ්‍රී ලංකාවේ අගනුවර වන්නේ"  # "The capital of Sri Lanka is"
ids = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=40, do_sample=False)
print("\n[GEN] " + tok.decode(out[0], skip_special_tokens=True))

# --- 2) Answer-token loss on a tiny writing-style style prompt.
prefix = ("You are an NLP assistant... Choose one category: ACADEMIC, CREATIVE, "
          "NEWS, or BLOG.\nComment: ")
example = prefix + "මෙම පර්යේෂණය සිදු කරන ලද්දේ විද්‍යාත්මක ක්‍රමවේදයක් අනුවය." + "\nAnswer:"
answer = " ACADEMIC"
pids = tok(example, add_special_tokens=True)["input_ids"]
aids = tok(answer, add_special_tokens=False)["input_ids"]
input_ids = torch.tensor([pids + aids], device=model.device)
labels = torch.tensor([[-100] * len(pids) + aids], device=model.device)
with torch.no_grad():
    loss = model(input_ids=input_ids, labels=labels).loss.item()
print(f"\n[LOSS] answer-token CE for one example = {loss:.2f}  "
      f"(>{math.log(V):.1f} means worse-than-random -> merge likely broken)")
