#!/usr/bin/env python3
"""Answer the COCO VQA questions with a SinLlama checkpoint (description + question in, answer out).

Runs on the 16 GB lab box: the 32 decoder layers sit on the GPU in bf16 (~13 GiB); the input
embedding and lm_head (139,336 x 4096 each) stay on the CPU. Prompts use the UltraChat SFT
template (prompt_template.txt), i.e. the model's own tokenizer chat template.

  python gen_sinllama.py --lang en --inp ../vqa_english.json --out sinllama_en_generated.json
  python gen_sinllama.py --lang si --inp ../vqa_sinhala.json --out sinllama_si_generated.json
  add --shots shots.json for k-shot (exemplars become earlier turns of the dialogue)
  add --adapter DIR for a LoRA fine-tune (ft_vqa.py), merged into the bf16 weights in memory

Greedy decoding, repetition_penalty 1.1, max_new_tokens 256 (dataset answers: p99 134 en /
149 si tokens). Resumable through <out>.jsonl.
"""
import argparse, json, os, time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PROMPTS = {
    "en": ("Read the following description and answer the question.\n\n"
           "Description: {description}\n\nQuestion: {question}"),
    "si": ("පහත විස්තරය කියවා ප්‍රශ්නයට පිළිතුරු දෙන්න.\n\n"
           "විස්තරය: {description}\n\nප්‍රශ්නය: {question}"),
}
STOP_STRINGS = ["### User:", "### Assistant:", "### System:"]
EOS_ID = 128001


class CPUEmbedding(nn.Module):
    """Embedding lookup on the CPU; the table is a plain tensor, so model.device stays cuda."""
    def __init__(self, weight, device):
        super().__init__()
        self.w, self.device = weight, device

    def forward(self, ids):
        return nn.functional.embedding(ids.cpu(), self.w).to(self.device)


class CPUHead(nn.Module):
    """lm_head on the CPU (generate only asks for the last position's logits).

    bf16 weights (the checkpoint dtype; reading them is the per-step bottleneck: 32 ms vs 53 ms
    for fp32 at batch 8) or fp32 (--head-fp32, as used in the 0-shot run). Logits returned in fp32.
    """
    def __init__(self, weight, device, fp32=False):
        super().__init__()
        self.w, self.device = (weight.float() if fp32 else weight), device

    def forward(self, h):
        return (h.to(self.w.dtype).cpu() @ self.w.T).float().to(self.device)


def load(model_dir, head_fp32=False, adapter=None):
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True, attn_implementation="sdpa")
    if adapter:   # trained on the nf4 base (QLoRA); merged into bf16 so decoding matches the 0-shot run
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        print(f"merged adapter {adapter}", flush=True)
    dev = torch.device("cuda")
    emb_w = model.model.embed_tokens.weight.data
    head_w = model.lm_head.weight.data
    model.model.embed_tokens = CPUEmbedding(emb_w, dev)
    model.lm_head = CPUHead(head_w, dev, fp32=head_fp32)
    model.model.layers.to(dev)
    model.model.norm.to(dev)
    model.model.rotary_emb.to(dev)
    model.eval()
    print(f"GPU after load: {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    return tok, model


def user_turn(lang, rec):
    return PROMPTS[lang].format(description=rec["description"].strip(), question=rec["question"].strip())


def build_prompt(tok, lang, rec, shots=()):
    """Few-shot = earlier user/assistant turns of the same dialogue (the SFT template is multi-turn)."""
    msgs = []
    for sh in shots:
        msgs += [{"role": "user", "content": user_turn(lang, sh)},
                 {"role": "assistant", "content": sh["answer"].strip()}]
    msgs.append({"role": "user", "content": user_turn(lang, rec)})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def clean(text):
    cut = len(text)
    for s in STOP_STRINGS:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "si"], required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.expanduser(
        "~/sinllama-continual-pretraining/models/SinLlama_wiki_uc_gen_mix"))
    ap.add_argument("--adapter", help="LoRA adapter dir to merge into --model")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--head-fp32", action="store_true", help="fp32 CPU lm_head (0-shot run setting)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="re-encode the shared few-shot prefix for every item (slow; for checking)")
    ap.add_argument("--ids", help="JSON list of test image_ids to restrict to (for checking)")
    ap.add_argument("--shots", help="JSON list of image_ids used as in-context examples "
                                     "(taken from --inp, in this order, and removed from the test set)")
    args = ap.parse_args()

    data = json.load(open(args.inp, encoding="utf-8"))
    shot_ids = json.load(open(args.shots)) if args.shots else []
    by_id = {r["image_id"]: r for r in data}
    shots = [by_id[i] for i in shot_ids]
    data = [r for r in data if r["image_id"] not in set(shot_ids)]
    if args.ids:
        keep = set(json.load(open(args.ids)))
        data = [r for r in data if r["image_id"] in keep]
    if args.limit:
        data = data[:args.limit]
    jsonl = os.path.splitext(args.out)[0] + ".jsonl"
    done = {}
    if os.path.exists(jsonl):
        for line in open(jsonl, encoding="utf-8"):
            if line.strip():
                j = json.loads(line)
                done[j["image_id"]] = j
    todo = [r for r in data if r["image_id"] not in done]
    print(f"{len(data)} records, {len(done)} done, {len(todo)} to generate", flush=True)

    if todo:
        torch.set_num_threads(args.threads)
        tok, model = load(args.model, args.head_fp32, args.adapter)
        prompts = {r["image_id"]: build_prompt(tok, args.lang, r, shots) for r in todo}
        if not done:
            print("--- example prompt ---\n" + prompts[todo[0]["image_id"]] + "\n---", flush=True)
        full = {k: tok(p, add_special_tokens=False).input_ids for k, p in prompts.items()}
        plen = {k: len(v) for k, v in full.items()}
        todo.sort(key=lambda r: plen[r["image_id"]], reverse=True)   # longest first: OOM shows early

        # Shared-prefix KV cache: every few-shot prompt starts with the same token run (the
        # exemplar turns + the next turn's fixed instruction). Encode it once; each batch is
        # prefix + pads + per-item suffix, with the pads masked (positions come from the mask).
        P, legacy = 0, None
        if shots and not args.no_prefix_cache:
            ref = full[todo[0]["image_id"]]
            P = min(plen.values()) - 1
            for v in full.values():
                while v[:P] != ref[:P]:
                    P -= 1
            with torch.inference_mode():
                legacy = model(input_ids=torch.tensor([ref[:P]], device="cuda"), use_cache=True,
                               num_logits_to_keep=1).past_key_values
                legacy = legacy.to_legacy_cache() if hasattr(legacy, "to_legacy_cache") else legacy
            print(f"shared prefix cached: {P} tokens", flush=True)

        def encode(recs):
            if not legacy:
                return tok([prompts[r["image_id"]] for r in recs], return_tensors="pt", padding=True,
                           add_special_tokens=False).to("cuda"), None
            suf = [full[r["image_id"]][P:] for r in recs]
            n = max(map(len, suf))
            ids = [full[recs[0]["image_id"]][:P] + [tok.pad_token_id] * (n - len(x)) + x for x in suf]
            mask = [[1] * P + [0] * (n - len(x)) + [1] * len(x) for x in suf]
            cache = DynamicCache.from_legacy_cache(tuple(
                (k.expand(len(recs), -1, -1, -1).contiguous(), v.expand(len(recs), -1, -1, -1).contiguous())
                for k, v in legacy))
            return {"input_ids": torch.tensor(ids, device="cuda"),
                    "attention_mask": torch.tensor(mask, device="cuda")}, cache

        def generate(recs):
            enc, cache = encode(recs)
            with torch.inference_mode():
                out = model.generate(**enc, past_key_values=cache, do_sample=False, temperature=None,
                                     top_p=None, max_new_tokens=args.max_new_tokens,
                                     repetition_penalty=args.repetition_penalty,
                                     eos_token_id=EOS_ID, pad_token_id=tok.pad_token_id,
                                     stop_strings=STOP_STRINGS, tokenizer=tok)
            res = []
            for r, row in zip(recs, out[:, enc["input_ids"].shape[1]:]):
                ids = row.tolist()
                hit_eos = EOS_ID in ids
                if hit_eos:
                    ids = ids[:ids.index(EOS_ID)]
                ids = [i for i in ids if i != tok.pad_token_id]
                raw = tok.decode(ids, skip_special_tokens=True)
                ans = clean(raw)
                res.append({
                    "image_id": r["image_id"],
                    "description": r["description"],
                    "question": r["question"],
                    "reference_answer": r["answer"],
                    "generated_answer": ans,
                    "prompt_tokens": plen[r["image_id"]],
                    "generated_tokens": len(ids),
                    "stop": "eos" if hit_eos else ("stop_string" if ans != raw.strip()
                                                   else "max_new_tokens"),
                })
            return res

        def safe_generate(recs):
            try:
                return generate(recs)
            except torch.cuda.OutOfMemoryError:
                if len(recs) == 1:
                    raise
            torch.cuda.empty_cache()   # retry outside except so the failed batch is freed
            h = len(recs) // 2
            print(f"  OOM at batch {len(recs)}, splitting", flush=True)
            return safe_generate(recs[:h]) + safe_generate(recs[h:])

        t0 = time.time()
        with open(jsonl, "a", encoding="utf-8") as f:
            for i in range(0, len(todo), args.batch):
                for j in safe_generate(todo[i:i + args.batch]):
                    done[j["image_id"]] = j
                    f.write(json.dumps(j, ensure_ascii=False) + "\n")
                f.flush()
                n = min(i + args.batch, len(todo))
                el = time.time() - t0
                print(f"{n}/{len(todo)}  {el / 60:.1f} min  ETA {el / n * (len(todo) - n) / 60:.1f} min",
                      flush=True)

    out = [done[r["image_id"]] for r in data]
    meta = {"model": os.path.basename(os.path.normpath(args.model)), "lang": args.lang,
            "adapter": args.adapter,
            "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens,
                         "repetition_penalty": args.repetition_penalty,
                         "stop": ["<|end_of_text|>"] + STOP_STRINGS},
            "prompt_format": PROMPTS[args.lang], "n": len(out),
            "n_shots": len(shots), "shot_image_ids": shot_ids,
            "shared_prefix_kv_cache": bool(shots) and not args.no_prefix_cache,
            "lm_head_dtype": "fp32" if args.head_fp32 else "bf16",
            "stop_counts": {s: sum(o["stop"] == s for o in out)
                            for s in ["eos", "stop_string", "max_new_tokens"]}}
    json.dump({"meta": meta, "records": out}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("wrote", args.out, meta["stop_counts"], flush=True)


if __name__ == "__main__":
    main()
