"""Generation acceptance test for the merged instruct model.

The pass/fail this whole template decision rested on: the model must emit exactly
one assistant turn and stop at <|end_of_text|>, rather than rolling on into a
fabricated "### User:" turn. Also probes multi-turn, English retention, and the
teacher-forced EOS probability that was the bactrianx failure mode.

    python sft/test_generate.py [model_dir] [--max-new N]
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ap = argparse.ArgumentParser()
_ap.add_argument("model", nargs="?", default="models/SinLlama_uc_instruct")
_ap.add_argument("--max-new", type=int, default=256)
_ap.add_argument("--eos", type=int, default=128001)
_ap.add_argument("--eval-file", default="UltraChat_Sinhala/test_sft.parquet")
_args = _ap.parse_args()

MODEL = _args.model
MAX_NEW = _args.max_new
EOS = _args.eos

SINHALA = [
    "ශ්‍රී ලංකාවේ අගනුවර කුමක්ද?",
    "ප්‍රභාසංශ්ලේෂණය යනු කුමක්දැයි සරලව පැහැදිලි කරන්න.",
    "මුහුද ගැන කෙටි කවියක් ලියන්න.",
    "පයිතන් භාෂාවෙන් ලැයිස්තුවක් අනුපිළිවෙළට සකසන ආකාරය කේතය සමඟ පෙන්වන්න.",
    "හොඳ නින්දක් ලබා ගැනීමට උපදෙස් තුනක් දෙන්න.",
    "ජල චක්‍රය පියවරෙන් පියවර විස්තර කරන්න.",
]
ENGLISH = [
    "What is the capital of Sri Lanka?",
    "Write a Python function that reverses a string.",
    "Explain gradient descent in two sentences.",
]
MULTITURN = [
    {"role": "user", "content": "ශ්‍රී ලංකාවේ ප්‍රධාන අපනයන භෝග මොනවාද?"},
    {"role": "assistant", "content": "ශ්‍රී ලංකාවේ ප්‍රධාන අපනයන භෝග වන්නේ තේ, රබර් සහ පොල් ය."},
    {"role": "user", "content": "ඒවායින් වැඩිම විදේශ විනිමය ලැබෙන්නේ කුමකින්ද? හේතුව කෙටියෙන් කියන්න."},
]

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
)
model.eval()
print(f"loaded {MODEL}  dtype={model.dtype}  eos={tok.eos_token_id} pad={tok.pad_token_id}\n")


def chat(messages, do_sample=False):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=MAX_NEW,
            do_sample=do_sample,
            temperature=0.7 if do_sample else None,
            top_p=0.9 if do_sample else None,
            eos_token_id=EOS,
            pad_token_id=tok.pad_token_id,
        )
    new = out[0, ids["input_ids"].shape[1]:]
    stopped = bool(len(new) and new[-1].item() == EOS)
    return tok.decode(new, skip_special_tokens=True), len(new), stopped, time.time() - t0


def report(title, prompts, do_sample=False):
    print("=" * 78)
    print(f"{title}   (greedy)" if not do_sample else f"{title}   (sampled T=0.7)")
    print("=" * 78)
    rows = []
    for p in prompts:
        msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
        txt, n, stopped, dt = chat(msgs, do_sample)
        leak = [m for m in ("### User:", "### Assistant:", "### System:") if m in txt]
        rows.append((stopped, n, bool(leak)))
        shown = msgs[-1]["content"]
        print(f"\n--- {shown[:70]}")
        print(f"    [{n} tok, {dt:.1f}s, "
              f"{'STOPPED at EOS' if stopped else 'HIT ' + str(MAX_NEW) + ' TOKEN CAP'}"
              f"{', LEAKED ' + ','.join(leak) if leak else ''}]")
        print("    " + txt.strip().replace("\n", "\n    ")[:1200])
    ok = sum(r[0] for r in rows)
    print(f"\n>>> stopped cleanly {ok}/{len(rows)} | "
          f"marker leakage {sum(r[2] for r in rows)}/{len(rows)} | "
          f"mean {sum(r[1] for r in rows) / len(rows):.0f} new tokens\n")


report("SINHALA, single turn", SINHALA)
report("ENGLISH, single turn", ENGLISH)
report("SINHALA, multi-turn follow-up", [MULTITURN])
report("SINHALA, sampled (repeat of prompt 1 and 3)", [SINHALA[0], SINHALA[2]], do_sample=True)

# --- teacher-forced EOS health (the bactrianx diagnostic) ------------------
print("=" * 78)
print("TEACHER-FORCED EOS PROBABILITY at real assistant-turn ends")
print("=" * 78)
import pandas as pd

df = pd.read_parquet(_args.eval_file).head(20)
probs, ranks = [], []
for msgs in df["messages"]:
    msgs = [{"role": m["role"], "content": m["content"]} for m in msgs]
    text = tok.apply_chat_template(msgs, tokenize=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    if ids["input_ids"].shape[1] > 2048:
        continue
    with torch.no_grad():
        logits = model(**ids).logits[0].float()
    pos = (ids["input_ids"][0] == EOS).nonzero().flatten()
    for p in pos:
        dist = logits[p - 1].softmax(-1)
        probs.append(dist[EOS].item())
        ranks.append((dist > dist[EOS]).sum().item() + 1)
probs_sorted = sorted(probs)
print(f"n positions: {len(probs)}")
print(f"P(EOS)  mean {sum(probs) / len(probs):.4f}   "
      f"median {probs_sorted[len(probs) // 2]:.4f}   min {min(probs):.2e}")
print(f"rank    median {sorted(ranks)[len(ranks) // 2]}   worst {max(ranks)}   "
      f"top-1 at {100 * sum(r == 1 for r in ranks) / len(ranks):.0f}% of turn ends")
print("(bactrianx, for contrast: P(EOS)=8e-8, rank 16,825 of 139,336)")
