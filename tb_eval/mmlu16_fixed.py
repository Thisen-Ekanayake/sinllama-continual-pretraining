#!/usr/bin/env python3
"""permute_eval.py on the 16 GB lab box with the memory fixes of rag/rag_eval.py, without retrieval.

permute_16gb.py alone leaves English MMLU's long 5-shot prompts (up to ~3.2k tokens) failing: permute_eval
retries an OOM *inside* its except block while the failed attempt's activations are still referenced, so one
OOM cascades (1,813 of 14,042 items unscored on 2026-11-22). Here, exactly as in rag_eval.py:
  1) embed_tokens (139k x 4096 bf16) runs on CPU: a pure lookup, bit-identical output;
  2) the OOM retry runs after the except block (split the batch; a single item that still OOMs is re-scored
     at 768 tokens and counted in the FALLBACK line, never silently).
Scoring, prompts and the cyclic permutation are permute_eval's, unchanged.
usage (repo root): ~/tts_env/bin/python mmlu16_fixed.py <permute_eval args>
"""
import os
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/tb_eval"))
import permute_16gb  # noqa: E402,F401  (patches SI/EN.load_model)
import permute_eval as PE  # noqa: E402

_load = PE.SI.load_model


def load_model(path, load_in_8bit=False):
    model, tok = _load(path)
    emb = model.model.embed_tokens
    emb.to("cpu")
    emb.register_forward_pre_hook(lambda m, a: tuple(x.to("cpu") for x in a))
    emb.register_forward_hook(lambda m, a, out: out.to("cuda"))
    torch.cuda.empty_cache()
    fwd = model.forward

    def forward(*a, **kw):
        kw = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in kw.items()}
        return fwd(*a, **kw)
    model.forward = forward
    print(f"  embed_tokens -> cpu; GPU now {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    return model, tok


PE.SI.load_model = load_model
PE.EN.load_model = load_model
FALLBACK, FAILED = [], []


@torch.no_grad()
def score(model, tok, chunk, option_ids, max_len, out):
    oom = False
    try:
        enc = tok([j["prompt"] for j in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(model.device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float().cpu()
        for b, j in enumerate(chunk):
            n, rot = j["n"], j["rot"]
            lp = torch.log_softmax(logits[b, option_ids[:n]], dim=-1)
            acc = out[j["idx"]]
            for i in range(n):
                acc["sum"][i] += float(lp[(i + rot) % n])
            acc["k"] += 1
            if rot == 0:
                acc["raw"] = int(torch.argmax(lp).item())
    except torch.cuda.OutOfMemoryError:
        oom = True
    if not oom:
        return
    enc = logits = None
    torch.cuda.empty_cache()
    if len(chunk) > 1:
        mid = len(chunk) // 2
        score(model, tok, chunk[:mid], option_ids, max_len, out)
        score(model, tok, chunk[mid:], option_ids, max_len, out)
    elif max_len > 768:
        FALLBACK.append(chunk[0]["idx"])
        score(model, tok, chunk, option_ids, 768, out)
    else:
        FAILED.append(chunk[0]["idx"])
        out[chunk[0]["idx"]]["failed"] = True


PE._score_jobs = score

if __name__ == "__main__":
    PE.main()
    print(f"items re-scored at 768 tokens after OOM: {len(set(FALLBACK))}; failed outright: {len(set(FAILED))}")
