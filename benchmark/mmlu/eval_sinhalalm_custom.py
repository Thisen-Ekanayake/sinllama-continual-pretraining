#!/usr/bin/env python3
"""
SinhalaMMLU for HelaAI/SinhalaLM-V.0.0.1-it -- a small from-scratch Sinhala GPT
(8 layers, d=512, 15k BPE vocab, 512-token context) with its own PyTorch model
class and whitespace-BPE tokenizer. Neither loads through transformers, so
evaluate_sinhala_mmlu.py / permute_eval.py can't score it directly. This script
re-implements only the forward pass and the tokenizer (copied from the repo's
modeling_sinhalalm.py, which is a training script and can't be imported) and
reuses evaluate_sinhala_mmlu.py for everything else: dataset loading, prompt
template, few-shot selection, metrics and report files. Output files have the
same names and shape as the standard arms, so downstream tools work unchanged.

One run scores all three arms of the position-bias study with one model load:
  raw         next-token log-softmax over the answer digits, argmax
  calibrated  same, minus the content-free positional prior (--calibrate)
  permuted    full cyclic rotation of the options, mean score per answer text

Differences from the standard protocol, forced by the model:
  * 512-token context. The standard 3-shot prompts are often longer, so whole
    few-shot blocks are dropped from the front until the prompt fits (rather
    than cutting mid-block, which is what left-truncation would do). Shots used
    per item are recorded; an item whose test block alone is too long is
    left-truncated and flagged.
  * The tokenizer splits on whitespace, so newlines are invisible to the model;
    the prompt reaches it as one flat word sequence. Answer digits are the
    single tokens "1</w>".."9</w>" (asserted distinct below).
  * --format native wraps each prompt in the model's own SFT template
    (system header + "### ප්‍රශ්නය:" / "### පිළිතුර:"), 0-shot; results go to
    <name>_native_* files.

Usage (on the pod):
  python eval_sinhalalm_custom.py --model-dir /workspace/models/SinhalaLM-V.0.0.1-it \
      --data-root SinhalaMMLU --prompt-file ../prompts/mmlu_sinhala.txt
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from functools import lru_cache

import torch
import torch.nn as nn
from torch.nn import functional as F
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_sinhala_mmlu as SI  # noqa: E402

OUT_DIRS = {"raw": "SinhalaMMLU_results_final",
            "calibrated": "SinhalaMMLU_results_calibrated",
            "permuted": "SinhalaMMLU_results_permuted"}
SYSTEM_HEADER = "පහත ප්‍රශ්නයට සිංහලෙන් පිළිතුරු දෙන්න."
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


# ----------------------------------------------------------------------------- #
# Tokenizer + model, from the model repo's modeling_sinhalalm.py (inference only)
# ----------------------------------------------------------------------------- #
class FastBPETokenizer:
    def __init__(self, path):
        data = json.load(open(path, encoding="utf-8"))
        self.stoi = data["stoi"]

        def pairs(d):
            return {tuple(k.split("|||", 1)) if "|||" in k else tuple(k.split("|", 1)): v
                    for k, v in d.items()}
        self.merges = pairs(data["merges"])
        self.prio = (pairs(data["merge_priority"]) if "merge_priority" in data
                     else {p: i for i, p in enumerate(self.merges)})
        self.unk = self.stoi.get("<UNK>", 1)
        self.bos = self.stoi.get("<BOS>", 2)

    @lru_cache(maxsize=None)
    def _word(self, word):
        symbols = list(word) + ["</w>"]
        while len(symbols) > 1:
            best, pos = None, None
            for i in range(len(symbols) - 1):
                p = self.prio.get((symbols[i], symbols[i + 1]))
                if p is not None and (best is None or p < best):
                    best, pos = p, i
            if pos is None:
                break
            merged = self.merges[(symbols[pos], symbols[pos + 1])]
            symbols = symbols[:pos] + [merged] + symbols[pos + 2:]
        return tuple(self.stoi.get(t, self.unk) for t in symbols)

    def encode(self, text):
        return [i for w in text.split() for i in self._word(w)]


class RoPEEmbedding(nn.Module):
    def __init__(self, dim, max_seq=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        freqs = torch.outer(torch.arange(max_seq).float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None])
        self.register_buffer("sin", emb.sin()[None, None])

    @staticmethod
    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, x, seq_len):
        return x * self.cos[:, :, :seq_len] + self.rotate_half(x) * self.sin[:, :, :seq_len]


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.rope = RoPEEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
                   .permute(2, 0, 3, 1, 4).unbind(0))
        q, k = self.rope(q, T), self.rope(k, T)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.ffn = SwiGLU(d, int(d * 8 / 3))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class SinhalaGPT(nn.Module):
    def __init__(self, vocab_size, d, n_heads, n_layers):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.ln_f(x))


def load(model_dir, device):
    cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
    model = SinhalaGPT(cfg["vocab_size"], cfg["n_embed"], cfg["n_heads"], cfg["n_layers"])
    missing, unexpected = model.load_state_dict(
        load_file(os.path.join(model_dir, "model.safetensors")), strict=False)
    # lm_head is tied to tok_emb, so it may legitimately be absent; nothing else may
    assert set(missing) <= {"lm_head.weight"} and not unexpected, (missing, unexpected)
    tok = FastBPETokenizer(os.path.join(model_dir, "bpe_vocabulary.json"))
    return model.to(device).eval(), tok, cfg["block_size"]


# ----------------------------------------------------------------------------- #
# Prompt -> token ids under the 512-token budget
# ----------------------------------------------------------------------------- #
def to_native(block):
    """Repo template block -> the model's own SFT template (answer cue kept last)."""
    body, _, ans = block.rpartition("පිළිතුර:")
    return f"{SYSTEM_HEADER}\n\n### ප්‍රශ්නය:\n{body.strip()}\n\n### පිළිතුර:\n{ans.strip()}"


def fit(tok, shot_blocks, test_block, budget):
    """[BOS] + as many trailing few-shot blocks as fit + test block. Returns
    (ids, shots_used, truncated)."""
    test = tok.encode(test_block)
    shots = [tok.encode(b) for b in shot_blocks]
    while shots and 1 + sum(map(len, shots)) + len(test) > budget:
        shots.pop(0)                      # drop the earliest exemplar first
    ids = [i for s in shots for i in s] + test
    truncated = len(ids) + 1 > budget
    return [tok.bos] + ids[-(budget - 1):], len(shots), truncated


@torch.no_grad()
def score(model, seqs, cand_ids, device, batch_size):
    """Log-softmax over the answer-digit tokens at each sequence's last position.
    Right-padded: causal attention + absolute RoPE positions mean padding after
    the last real token cannot affect it."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [None] * len(seqs)
    cand = torch.tensor(cand_ids, device=device)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        L = max(len(seqs[i]) for i in idx)
        x = torch.zeros(len(idx), L, dtype=torch.long)
        for j, i in enumerate(idx):
            x[j, :len(seqs[i])] = torch.tensor(seqs[i])
        logits = model(x.to(device)).float()
        last = torch.tensor([len(seqs[i]) - 1 for i in idx], device=device)
        lp = logits[torch.arange(len(idx), device=device), last][:, cand].cpu()
        for j, i in enumerate(idx):
            out[i] = lp[j]
    return out


def write_arm(out_dir, name, records, m):
    os.makedirs(out_dir, exist_ok=True)
    json.dump(m, open(os.path.join(out_dir, f"{name}_metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"{name}_predictions.jsonl"), "w", encoding="utf-8") as fh:
        for r in records:
            row = {k: r.get(k) for k in ("file", "difficulty", "category", "subject_display",
                                         "n_choices", "gold", "valid", "pred", "correct",
                                         "pred_raw", "shots_used", "truncated")
                   if k in r}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    SI.write_txt(os.path.join(out_dir, f"{name}_results.txt"), name, m)
    SI.write_md(os.path.join(out_dir, f"{name}_results.md"), name, m)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-root", default="SinhalaMMLU")
    ap.add_argument("--prompt-file", default="../prompts/mmlu_sinhala.txt")
    ap.add_argument("--format", choices=["raw", "native"], default="raw")
    ap.add_argument("--kshot", type=int, default=None,
                    help="default 3 for raw (as config.yml), 0 for native")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="questions per TEST file (0 = all)")
    ap.add_argument("--out-root", default=".")
    args = ap.parse_args()
    k = args.kshot if args.kshot is not None else (3 if args.format == "raw" else 0)
    wrap = to_native if args.format == "native" else (lambda b: b)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    name = os.path.basename(args.model_dir.rstrip("/"))
    if args.format == "native":
        name += "_native"

    template = SI.load_template(args.prompt_file)
    records, missing = SI.build_examples(args.data_root, template, k)
    if args.limit:
        seen, kept = defaultdict(int), []
        for r in records:
            if seen[r["file"]] < args.limit:
                kept.append(r); seen[r["file"]] += 1
        records = kept
    if missing:
        print(f"WARNING: no few-shot match for: {sorted(missing)}")
    valid = [r for r in records if r["valid"]]
    print(f"{len(records)} questions ({len(valid)} valid), format={args.format}, k={k}")

    t0 = time.time()
    model, tok, block_size = load(args.model_dir, device)
    digits = [tok.stoi.get(f"{d}</w>") for d in range(1, 10)]
    assert None not in digits and len(set(digits)) == 9, f"bad digit tokens {digits}"
    for d in range(1, 10):
        assert tok.encode(str(d)) == [digits[d - 1]], f"'{d}' is not one token"
    print(f"model on {device}, block_size={block_size}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, digit ids {digits}")

    def build(r, choices, question=None):
        shots = [wrap(b) for b in r["shot_prefix"].split("\n\n")] if r["shot_prefix"] else []
        test = wrap(SI.render(template, r["subject_original"],
                              r["question"] if question is None else question,
                              choices, answer=None))
        return fit(tok, shots, test, block_size)

    # ---- jobs: every cyclic rotation of every item (rotation 0 = raw arm) ----
    jobs, seqs = [], []
    for ri, r in enumerate(valid):
        n = r["n_choices"]
        for rot in range(n):
            ids, used, trunc = build(r, _rotate(r["choices"], rot))
            if rot == 0:
                r["shots_used"], r["truncated"] = used, trunc
            jobs.append((ri, rot, n))
            seqs.append(ids)
    print(f"tokenized {len(seqs)} prompts in {time.time() - t0:.0f}s; "
          f"shots used {dict(sorted(_count(r['shots_used'] for r in valid).items()))}, "
          f"{sum(r['truncated'] for r in valid)} test blocks truncated")

    lp = score(model, seqs, digits, device, args.batch_size)

    # ---- content-free prior per (difficulty, subject, n_choices) ----
    cf_keys, cf_seqs = [], []
    for r in valid:
        key = SI.cf_key(r)
        if key not in cf_keys:
            cf_keys.append(key)
            cf_seqs.append(build(r, [SI.CF_QUESTION] * r["n_choices"], SI.CF_QUESTION)[0])
    cf_lp = score(model, cf_seqs, digits, device, args.batch_size)
    cf_bias = {key: torch.log_softmax(v[:key[2]], -1) for key, v in zip(cf_keys, cf_lp)}

    # ---- aggregate ----
    sums = defaultdict(lambda: torch.zeros(9))
    raw_scores = {}
    for (ri, rot, n), v in zip(jobs, lp):
        s = torch.log_softmax(v[:n], -1)
        if rot == 0:
            raw_scores[ri] = s
        # option originally at index i sits in slot (i + rot) % n
        sums[ri][:n] += s[[(i + rot) % n for i in range(n)]]

    arms = {}
    for arm in ("raw", "calibrated", "permuted"):
        recs = [dict(r) for r in records]
        vi = 0
        for r in recs:
            if not r["valid"]:
                continue
            n = r["n_choices"]
            if arm == "raw":
                s = raw_scores[vi]
            elif arm == "calibrated":
                s = raw_scores[vi] - cf_bias[SI.cf_key(r)]
            else:
                s = sums[vi][:n]
                r["pred_raw"] = int(raw_scores[vi].argmax()) + 1
            r["pred"] = int(s.argmax()) + 1
            r["correct"] = r["pred"] == r["gold"]
            vi += 1
        m = SI.compute_metrics(recs)
        m.update(format=args.format, kshot=k, calibrated=(arm == "calibrated"),
                 permuted=(arm == "permuted"), context=block_size,
                 shots_used=_count(r["shots_used"] for r in valid),
                 truncated_test_blocks=sum(r["truncated"] for r in valid))
        if arm == "permuted":
            m["rotations"] = "full"
        write_arm(os.path.join(args.out_root, OUT_DIRS[arm]), name, recs, m)
        arms[arm] = m
        print(f"  {arm:<11} {m['overall']['accuracy']:6.2f}%  "
              f"({m['overall']['correct']}/{m['overall']['total']})")
    dist = _count(int(v.argmax()) + 1 for v in raw_scores.values())
    print(f"raw-arm predicted-option distribution: {dict(sorted(dist.items()))}")
    print(f"done in {time.time() - t0:.0f}s")


def _rotate(choices, r):
    out = [None] * len(choices)
    for i, c in enumerate(choices):
        out[(i + r) % len(choices)] = c
    return out


def _count(it):
    c = defaultdict(int)
    for x in it:
        c[x] += 1
    return dict(c)


if __name__ == "__main__":
    main()
