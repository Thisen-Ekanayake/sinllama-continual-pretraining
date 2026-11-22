#!/usr/bin/env python3
"""SinhalaMMLU with retrieved textbook passages (open-book), raw arm.

permute_eval.py + permute_16gb.py loader, unchanged except that each test
prompt becomes

    <3 few-shot blocks, as before>\n\n<CONTEXT header + top-k passages>\n\n<test block>

Exemplars get no passages (keeps the prompt within 16 GB); the template, the
option-digit scoring and the answer cue are untouched. Run with --rotations 1
(raw arm only: pred == pred_raw), so the result pairs item-for-item with the
no-retrieval pred_raw of the same model.

usage (repo root):
  ~/tts_env/bin/python rag_eval.py --retrieval ~/tb_eval/rag/bge-m3/retrieval.jsonl --k 3 \
      <permute_eval args: --lang sinhala --models ... --rotations 1 ...>
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/tb_eval"))
import permute_16gb  # noqa: E402,F401  (patches SI.load_model)
import permute_eval as PE  # noqa: E402

# A merged copy on disk wins over permute_16gb's parent + in-memory-LoRA recipe
# (e.g. models/SinLlama_textbook_mix from gs://sinllama_cpt, parent deleted).
for _n in list(permute_16gb.ADAPTERS):
    if os.path.exists(f"models/{_n}/model.safetensors.index.json"):
        del permute_16gb.ADAPTERS[_n]
        print(f"loading merged models/{_n} directly (not parent + LoRA)")

HEADER = "පහත දැක්වෙන්නේ ප්‍රශ්නයට අදාළ විය හැකි, පාසල් පෙළපොත්වලින් උපුටා ගත් කොටස් කිහිපයකි."


def pop_arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return default


RETRIEVAL = pop_arg("--retrieval")
K = int(pop_arg("--k", "3"))       # 0 = closed book (same scorer, no passages)
CHUNKS = pop_arg("--chunks", os.path.expanduser("~/tb_eval/rag/chunks.jsonl"))
assert RETRIEVAL, "--retrieval is required"

chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
retr = [json.loads(l) for l in open(RETRIEVAL, encoding="utf-8")]


def passage(c):
    return (c["heading"] + "\n" + c["text"]) if c["heading"] else c["text"]


def context(ids):
    return HEADER + "\n" + "\n".join(f"[{i+1}] {passage(chunks[j])}"
                                     for i, j in enumerate(ids[:K]))


_build = PE.SinhalaAdapter.build


def build(self, args):
    assert not args.limit, "--limit would desync records and retrieval"
    records = _build(self, args)
    if K == 0:
        print("RAG: k=0, closed book")
        return records
    assert len(records) == len(retr), (len(records), len(retr))
    pos = {}
    for r, x in zip(records, retr):
        p = pos.get(r["file"], 0)
        pos[r["file"]] = p + 1
        assert (x["file"], x["pos"]) == (r["file"], p), (x["file"], x["pos"], r["file"], p)
        test = self.render_test(r, r["choices"])
        assert r["prompt"].endswith(test)
        r["prompt"] = ((r["shot_prefix"] + "\n\n") if r["shot_prefix"] else "") \
            + context(x["ids"]) + "\n\n" + test
        r["rag_ids"] = x["ids"][:K]
    print(f"RAG: {RETRIEVAL}, k={K}; example prompt tail:\n"
          + records[0]["prompt"][-1500:] + "\n---")
    return records


PE.SinhalaAdapter.build = build

# --- memory -----------------------------------------------------------------
# 1) The token-embedding table (139k x 4096 bf16 = 1.1 GB) is a pure lookup, so
#    it moves to CPU (bit-identical output) to give eager attention room for
#    the ~1.7k-token RAG prompts on a 16 GB card.
# 2) permute_eval retries an OOM *inside* its except block, while the failed
#    attempt's activations are still referenced by the traceback, so one OOM
#    cascades into every retry. Same scoring code, retry moved outside.
_load = PE.SI.load_model


def load_model(path, load_in_8bit=False):
    model, tok = _load(path)
    emb = model.model.embed_tokens
    emb.to("cpu")
    emb.register_forward_pre_hook(lambda m, a: tuple(x.to("cpu") for x in a))
    emb.register_forward_hook(lambda m, a, out: out.to("cuda"))
    torch.cuda.empty_cache()
    fwd = model.forward             # permute_16gb's wrapper (logits_to_keep)

    def forward(*a, **kw):          # harness sends inputs to model.device (now cpu)
        kw = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in kw.items()}
        return fwd(*a, **kw)
    model.forward = forward
    print(f"  embed_tokens -> cpu; GPU now {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    return model, tok


PE.SI.load_model = load_model
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
    print(f"RAG items truncated to 768 tokens after OOM: {len(FALLBACK)}; "
          f"failed outright: {len(FAILED)}")
