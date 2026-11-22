#!/usr/bin/env python3
"""Belebele (Sinhala sin_Sinh, English eng_Latn) reading comprehension, scored
like the SinhalaMMLU runs: 3-shot, next-token digit logits over options 1-4,
raw (rotation 0) and full cyclic permutation (all 4 rotations, mean log-prob
per original option). Same 16 GB loader as Appendix B/C (permute_16gb +
embed_tokens on CPU), safe OOM retry.

Arms: "passage" (the benchmark) and "nopassage" (same prompt without the
passage: how much is answerable without reading).
Exemplars: one question from each of 3 fixed passages; every question on those
3 passages is excluded from scoring (no passage overlap with the shots).

usage (repo root): ~/tts_env/bin/python belebele_eval.py --model models/X --out DIR
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/tb_eval"))
import permute_16gb  # noqa: E402

B = os.path.expanduser("~/tb_eval/belebele")
TEMPLATES = {
    "sin_Sinh": dict(
        head="පහත ඡේදය කියවා, ප්‍රශ්නයට 1, 2, 3, 4 යන පිළිතුරු වලින් නිවැරදි පිළිතුර තෝරන්න.",
        head_np="පහත ප්‍රශ්නයට 1, 2, 3, 4 යන පිළිතුරු වලින් නිවැරදි පිළිතුර තෝරන්න.",
        passage="ඡේදය", question="ප්‍රශ්නය", answer="පිළිතුර"),
    "eng_Latn": dict(
        head="Read the passage and choose the correct answer to the question from 1, 2, 3, 4.",
        head_np="Choose the correct answer to the question from 1, 2, 3, 4.",
        passage="Passage", question="Question", answer="Answer"),
}


def render(T, r, opts, with_passage, answer=None):
    lines = [T["head"] if with_passage else T["head_np"]]
    if with_passage:
        lines.append(f"{T['passage']}: {r['flores_passage'].strip()}")
    lines.append(f"{T['question']}: {r['question'].strip()}")
    lines += [f"{i+1}. {o.strip()}" for i, o in enumerate(opts)]
    s = "\n".join(lines) + f"\n{T['answer']}:"
    return s + (f" {answer}" if answer is not None else " ")


def options(r):
    return [r[f"mc_answer{i}"] for i in range(1, 5)]


def pick_shots(rows):
    """3 shortest passages (by chars) with distinct gold answers -> 1 question each."""
    by_link = {}
    for r in rows:
        by_link.setdefault(r["link"], []).append(r)
    shots, used = [], set()
    for link in sorted(by_link, key=lambda l: (len(by_link[l][0]["flores_passage"]), l)):
        for r in by_link[link]:
            if r["correct_answer_num"] not in used:
                shots.append(r)
                used.add(r["correct_answer_num"])
                break
        if len(shots) == 3:
            break
    return shots


def load(path, attn="eager"):
    name = os.path.basename(path.rstrip("/"))
    if os.path.exists(f"{path}/model.safetensors.index.json"):
        permute_16gb.ADAPTERS.pop(name, None)
    if attn != "eager":
        # memory-efficient attention for prompts too long for eager on 16 GB
        # (base Llama-3 tokenises Sinhala into ~6x more tokens than SinLlama).
        # The attention class is fixed at load time, so load with it directly.
        from transformers import AutoModelForCausalLM
        _fp = AutoModelForCausalLM.from_pretrained
        permute_16gb.AutoModelForCausalLM.from_pretrained = \
            lambda *a, **kw: _fp(*a, **{**kw, "attn_implementation": attn})
        permute_16gb.SI.assert_eager = lambda *a, **kw: None
    model, tok = permute_16gb.load_model(path)
    print(f"  attention: {model.config._attn_implementation}, "
          f"{type(model.model.layers[0].self_attn).__name__}")
    emb = model.model.embed_tokens
    emb.to("cpu")
    emb.register_forward_pre_hook(lambda m, a: tuple(x.to("cpu") for x in a))
    emb.register_forward_hook(lambda m, a, out: out.to("cuda"))
    torch.cuda.empty_cache()
    return model, tok


@torch.no_grad()
def score(model, tok, jobs, ids, out, fails):
    oom = False
    try:
        enc = tok([j["prompt"] for j in jobs], return_tensors="pt", padding=True)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        lg = model(**enc, logits_to_keep=1).logits[:, -1, :].float().cpu()
        for b, j in enumerate(jobs):
            lp = torch.log_softmax(lg[b, ids], -1)
            a = out[j["key"]]
            for i in range(4):                       # original option i sits at (i+rot)%4
                a["sum"][i] += float(lp[(i + j["rot"]) % 4])
            a["k"] += 1
            if j["rot"] == 0:
                a["raw"] = int(lp.argmax())
    except torch.cuda.OutOfMemoryError:
        oom = True
    if not oom:
        return
    enc = lg = None
    torch.cuda.empty_cache()
    if len(jobs) == 1:
        fails.append(jobs[0]["key"])
        return
    h = len(jobs) // 2
    score(model, tok, jobs[:h], ids, out, fails)
    score(model, tok, jobs[h:], ids, out, fails)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="+", default=["sin_Sinh", "eng_Latn"])
    ap.add_argument("--budget", type=float, default=6e6)
    ap.add_argument("--arms", nargs="+", default=["passage", "nopassage"])
    ap.add_argument("--attn", default="eager", choices=("eager", "sdpa"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    name = os.path.basename(args.model.rstrip("/"))
    model, tok = load(args.model, args.attn)
    ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 5)]
    assert len(set(ids)) == 4, ids

    for lang in args.langs:
        rows = [json.loads(l) for l in open(f"{B}/{lang}.jsonl", encoding="utf-8")]
        T = TEMPLATES[lang]
        shots = pick_shots(rows)
        shot_links = {s["link"] for s in shots}
        test = [r for r in rows if r["link"] not in shot_links]
        for arm, wp in (("passage", True), ("nopassage", False)):
            if arm not in args.arms:
                continue
            t0 = time.time()
            prefix = "\n\n".join(render(T, s, options(s), wp, s["correct_answer_num"]) for s in shots)
            jobs = []
            for k, r in enumerate(test):
                o = options(r)
                for rot in range(4):
                    ro = [None] * 4
                    for i, x in enumerate(o):
                        ro[(i + rot) % 4] = x
                    jobs.append(dict(key=k, rot=rot, prompt=prefix + "\n\n" + render(T, r, ro, wp)))
            lens = [len(x) for x in tok([j["prompt"] for j in jobs])["input_ids"]]
            order = sorted(range(len(jobs)), key=lambda i: -lens[i])
            out = {k: {"sum": [0.0] * 4, "k": 0, "raw": None} for k in range(len(test))}
            fails, cur, mx = [], [], 0
            for i in order:
                m = max(mx, lens[i])
                if cur and ((len(cur) + 1) * m * m > args.budget or len(cur) >= 4):
                    score(model, tok, cur, ids, out, fails)
                    cur, m = [], lens[i]
                cur.append(jobs[i])
                mx = m
            if cur:
                score(model, tok, cur, ids, out, fails)
            preds, raw_ok, perm_ok, n = [], 0, 0, 0
            for k, r in enumerate(test):
                a = out[k]
                gold = int(r["correct_answer_num"])
                ok = a["k"] == 4 and a["raw"] is not None
                pr = a["raw"] + 1 if ok else -1
                pp = max(range(4), key=lambda i: a["sum"][i]) + 1 if ok else -1
                n += 1
                raw_ok += pr == gold
                perm_ok += pp == gold
                preds.append(dict(link=r["link"], question_number=r["question_number"], gold=gold,
                                  pred_raw=pr, pred=pp))
            m = dict(model=name, lang=lang, arm=arm, n=n, failed=len(set(fails)), attn=args.attn,
                     max_tokens=max(lens), shots=[(s["link"], s["question_number"]) for s in shots],
                     raw=100 * raw_ok / n, permuted=100 * perm_ok / n,
                     raw_correct=raw_ok, permuted_correct=perm_ok, seconds=round(time.time() - t0))
            print(json.dumps(m, ensure_ascii=False), flush=True)
            tag = f"{name}_{lang}_{arm}"
            json.dump(m, open(f"{args.out}/{tag}_metrics.json", "w"), ensure_ascii=False, indent=1)
            with open(f"{args.out}/{tag}_predictions.jsonl", "w", encoding="utf-8") as fh:
                for p in preds:
                    fh.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
