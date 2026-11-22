#!/usr/bin/env python3
"""Embed the textbook passages and retrieve top-K for every SinhalaMMLU item.

Embedders (each used the way its model card says):
  bge-m3          XLM-R encoder, dense = normalised [CLS], max 512 tokens here
                  (passages are <= 1000 chars), no query instruction.
  HelaBERT_Large  BERT-MLM, SentencePiece ids (no CLS/SEP, as in its README),
                  mean-pooled last layer, normalised; input cleaned like its
                  pre-training text (Sinhala, digits, punctuation, ZWJ/ZWNJ);
                  max 512 positions.
Query = question + options (the same text the LLM is shown, minus the template).
Retrieval = exact cosine over all passages (33k, a single matmul).

Records come from SI.build_examples, so their order is exactly permute_eval's.
Output: <out>/<embedder>/passages.npy (fp16), retrieval.jsonl
        (one line per record: file, pos, valid, ids[K], scores[K]).

usage (repo root): ~/tts_env/bin/python embed_retrieve.py --embedder bge-m3|HelaBERT_Large
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "benchmark/mmlu")
import evaluate_sinhala_mmlu as SI   # noqa: E402

E = os.path.expanduser("~/tb_eval")
K = 10


class BGE:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        p = f"{E}/embedders/bge-m3"
        self.tok = AutoTokenizer.from_pretrained(p)
        self.model = AutoModel.from_pretrained(p, torch_dtype=torch.float16).cuda().eval()

    def n_tokens(self, texts):
        return [len(x) for x in self.tok(texts, add_special_tokens=True)["input_ids"]]

    @torch.no_grad()
    def encode(self, texts):
        enc = self.tok(texts, padding=True, truncation=True, max_length=512,
                       return_tensors="pt").to("cuda")
        h = self.model(**enc).last_hidden_state[:, 0]
        return F.normalize(h.float(), dim=-1)


KEEP = re.compile(r"[^඀-෿0-9\s.,?!:;()\[\]'\"%/\-‌‍]")


class Hela:
    def __init__(self):
        import sentencepiece as spm
        from transformers import BertModel
        p = f"{E}/embedders/HelaBERT_Large"
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{p}/tokenizer/unigram_32000_0.9995.model")
        self.pad = 0                     # config pad_token_id (sp.pad_id() is -1)
        self.model = BertModel.from_pretrained(f"{p}/HelaBERT_Large",
                                               add_pooling_layer=False).cuda().eval()

    @staticmethod
    def clean(t):
        t = unicodedata.normalize("NFC", t).replace("​", "")
        return re.sub(r"\s+", " ", KEEP.sub(" ", t)).strip()

    def ids(self, texts):
        return [self.sp.encode(self.clean(t), out_type=int) for t in texts]

    def n_tokens(self, texts):
        return [len(x) for x in self.ids(texts)]

    @torch.no_grad()
    def encode(self, texts):
        ids = [x[:512] or [self.pad] for x in self.ids(texts)]
        n = max(len(x) for x in ids)
        t = torch.full((len(ids), n), self.pad, dtype=torch.long)
        m = torch.zeros((len(ids), n), dtype=torch.long)
        for i, x in enumerate(ids):
            t[i, :len(x)] = torch.tensor(x)
            m[i, :len(x)] = 1
        t, m = t.cuda(), m.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.model(input_ids=t, attention_mask=m).last_hidden_state.float()
        mm = m.unsqueeze(-1).float()
        return F.normalize((h * mm).sum(1) / mm.sum(1), dim=-1)


def embed_all(enc, texts, bs):
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))   # less padding
    vecs = [None] * len(texts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        v = enc.encode([texts[i] for i in idx]).half().cpu()
        for j, i in enumerate(idx):
            vecs[i] = v[j]
    return torch.stack(vecs)


def query_text(r):
    opts = "\n".join(f"{i+1}. {SI.clean(c)}" for i, c in enumerate(r["choices"]))
    return SI.clean(r["question"]) + "\n" + opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", required=True, choices=("bge-m3", "HelaBERT_Large"))
    ap.add_argument("--chunks", default=f"{E}/rag/chunks.jsonl")
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()
    out = f"{E}/rag/{args.embedder}"
    os.makedirs(out, exist_ok=True)

    chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8")]
    ptexts = [(c["heading"] + "\n" + c["text"]) if c["heading"] else c["text"]
              for c in chunks]
    template = SI.load_template("benchmark/prompts/mmlu_sinhala.txt")
    records, _ = SI.build_examples("benchmark/mmlu/SinhalaMMLU", template, 3)
    qtexts = [query_text(r) for r in records]

    enc = BGE() if args.embedder == "bge-m3" else Hela()
    pl = np.array(enc.n_tokens(ptexts))
    ql = np.array(enc.n_tokens(qtexts))
    print(f"{args.embedder}: {len(chunks)} passages, tokens p50 {np.median(pl):.0f} "
          f"p99 {np.percentile(pl, 99):.0f} max {pl.max()} (>{512}: {(pl > 512).sum()}); "
          f"{len(qtexts)} queries, tokens p50 {np.median(ql):.0f} max {ql.max()} "
          f"(>{512}: {(ql > 512).sum()})")

    t0 = time.time()
    P = embed_all(enc, ptexts, args.bs)
    print(f"passages embedded in {time.time() - t0:.0f}s")
    Q = embed_all(enc, qtexts, args.bs)
    np.save(f"{out}/passages.npy", P.numpy())
    np.save(f"{out}/queries.npy", Q.numpy())

    Pg = P.cuda().float()
    pos = {}
    with open(f"{out}/retrieval.jsonl", "w", encoding="utf-8") as fh:
        for s in range(0, len(Q), 512):
            sim = Q[s:s + 512].cuda().float() @ Pg.T
            sc, ix = sim.topk(K, dim=-1)
            for j in range(len(sc)):
                r = records[s + j]
                p = pos.get(r["file"], 0)
                pos[r["file"]] = p + 1
                fh.write(json.dumps(dict(file=r["file"], pos=p, valid=r["valid"],
                                         ids=ix[j].tolist(),
                                         scores=[round(x, 4) for x in sc[j].tolist()]),
                                    ensure_ascii=False) + "\n")
    print(f"wrote {out}/retrieval.jsonl ({len(records)} records) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
