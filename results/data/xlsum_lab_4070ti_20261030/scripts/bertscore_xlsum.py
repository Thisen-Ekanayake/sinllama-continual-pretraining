"""BERTScore (Zhang et al. 2020) for XL-Sum predictions with HelaBERT_Large.

Same algorithm as the `bert_score` package (no idf, no baseline rescaling --
there is no Sinhala baseline file): contextual embeddings from one hidden
layer, [CLS]/[SEP] excluded, cosine greedy matching; P = mean over candidate
tokens of the best reference match, R = the reverse, F1 = 2PR/(P+R).
HelaBERT ships a SentencePiece tokenizer, so inputs are built as in its
pretraining: <s>(=CLS) + sp.encode(NFC text) + </s>(=SEP).
Layer 9 of 12 is bert_score's default for 12-layer BERT models; the last
layer (12) is also reported as a sensitivity check.
Usage: python bertscore_xlsum.py preds_dir out_prefix [sinhala|english]
"""
import json, os, sys, unicodedata
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
DEV = os.environ.get("DEVICE", "cuda")
from transformers import BertModel

HELA = "/ml/HelaBERT_Large"
ROBERTA = "/ml/Roberta-Large"
MODELS = ["llama-3-8b", "SinLlama_v01", "SinLlama_wiki_uc_gen_mix", "SinLlama_textbook_mix"]
PAIRS = [("SinLlama_v01", "llama-3-8b"), ("SinLlama_wiki_uc_gen_mix", "SinLlama_v01"),
         ("SinLlama_textbook_mix", "SinLlama_wiki_uc_gen_mix"), ("SinLlama_textbook_mix", "llama-3-8b")]
LANG = sys.argv[3] if len(sys.argv) > 3 else "sinhala"
if LANG == "sinhala":
    LAYERS, MAIN = [9, 12], 9
    sp = spm.SentencePieceProcessor(model_file=f"{HELA}/tokenizer/unigram_32000_0.9995.model")
    bert = BertModel.from_pretrained(f"{HELA}/HelaBERT_Large", add_pooling_layer=False).to(DEV).eval()
    PAD = sp.pad_id()
    encode = lambda t: [sp.bos_id()] + sp.encode(unicodedata.normalize("NFC", t), out_type=int)[:510] + [sp.eos_id()]
else:
    # English: bert_score's default model/layer (roberta-large, layer 17), its
    # tokenisation (strip, add_prefix_space, <s> ... </s>); last layer 24 as the check.
    from transformers import AutoModel, AutoTokenizer
    LAYERS, MAIN = [17, 24], 17
    rtok = AutoTokenizer.from_pretrained(ROBERTA, add_prefix_space=True)
    bert = AutoModel.from_pretrained(ROBERTA, add_pooling_layer=False).to(DEV).eval()
    PAD = rtok.pad_token_id
    encode = lambda t: rtok(t.strip(), add_special_tokens=True, truncation=True, max_length=510).input_ids


@torch.no_grad()
def embed(texts, bs=64):
    """Per text: {layer: L2-normalised token embeddings without CLS/SEP}."""
    out = []
    for i in range(0, len(texts), bs):
        ids = [encode(t) for t in texts[i:i + bs]]
        n = max(map(len, ids))
        x = torch.tensor([s + [PAD] * (n - len(s)) for s in ids]).to(DEV)
        hs = bert(input_ids=x, attention_mask=(x != PAD).long(), output_hidden_states=True).hidden_states
        for j, s in enumerate(ids):
            out.append({l: torch.nn.functional.normalize(hs[l][j, 1:len(s) - 1].float(), dim=-1) for l in LAYERS})
    return out


def bertscore(cands, refs):
    ce, re_ = embed(cands), embed(refs)
    res = {l: {"P": [], "R": [], "F1": []} for l in LAYERS}
    for c, r in zip(ce, re_):
        for l in LAYERS:
            if len(c[l]) == 0 or len(r[l]) == 0:          # empty prediction scores 0
                p = rr = f = 0.0
            else:
                sim = r[l] @ c[l].T                        # ref tokens x cand tokens
                p, rr = sim.max(0).values.mean().item(), sim.max(1).values.mean().item()
                f = 2 * p * rr / (p + rr)
            res[l]["P"].append(p); res[l]["R"].append(rr); res[l]["F1"].append(f)
    return {l: {k: np.array(v) for k, v in d.items()} for l, d in res.items()}


preds_dir, out = Path(sys.argv[1]), sys.argv[2]
rng = np.random.default_rng(1234)
table, per_ex = [], {}
for m in MODELS:
    f = preds_dir / f"{m}.jsonl"
    if not f.exists():
        continue
    rows = sorted((json.loads(l) for l in open(f)), key=lambda r: r["id"])
    s = bertscore([r["prediction"] for r in rows], [r["reference"] for r in rows])
    per_ex[m] = s
    row = {"model": m, "n": len(rows)}
    for l in LAYERS:
        for k in ("P", "R", "F1"):
            v = s[l][k]
            boot = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
            row[f"L{l}_{k}"] = round(v.mean() * 100, 2)
            row[f"L{l}_{k}_ci"] = [round(np.percentile(boot, 2.5) * 100, 2), round(np.percentile(boot, 97.5) * 100, 2)]
    table.append(row)

for l in LAYERS:
    print(f"-- {'HelaBERT_Large' if LANG == 'sinhala' else 'roberta-large'} layer {l}" + (" (main)" if l == MAIN else " (sensitivity)"))
    for t in table:
        print(f"  {t['model']:28s} P {t[f'L{l}_P']:6.2f}  R {t[f'L{l}_R']:6.2f}  F1 {t[f'L{l}_F1']:6.2f}  {t[f'L{l}_F1_ci']}")

pairs = []
for a, b in PAIRS:
    if a not in per_ex or b not in per_ex:
        continue
    row = {"a": a, "b": b}
    for l in LAYERS:
        d = per_ex[a][l]["F1"] - per_ex[b][l]["F1"]
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
        p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
        row[f"L{l}_F1"] = {"delta": round(d.mean() * 100, 2), "p": round(float(max(p, 1e-4)), 4)}
    pairs.append(row)
    print(f"{a} - {b}: " + ", ".join(f"L{l} F1 {row[f'L{l}_F1']['delta']:+.2f} (p={row[f'L{l}_F1']['p']})" for l in LAYERS))

json.dump({"scores": table, "paired_bootstrap": pairs}, open(out + ".json", "w"), indent=1)
