#!/usr/bin/env python3
"""Build the bilingual (75% Sinhala / 25% English by tokens) instruct-tuning mix for SinLlama_wiki_uc_gen_mix.

Runs on the lab box (sources in ~/sinhala_instruct_datasets and ~/coco_train/chunks):

  python build_instruct_mix.py                 # -> ~/sinhala_instruct_datasets/mix/{train,val}.parquet + stats.json

Sources and treatment
  English (25% of tokens)
    vqa_en   COCO train2017 VQA, English (Qwen3.5-9B), all 100k, loop-fixed file
    uc_en    UltraChat-200k train_sft (shard 0), --uc-en dialogues; prompt_ids with an odd md5 (disjoint from uc_si)
  Sinhala (75% of tokens)
    vqa_si   COCO train2017 VQA, Sinhala (NLLB), all 100k, loop-fixed file
    aya_ds   Aya Dataset, language == Sinhala (human-written), deduplicated
    aya_coll Aya Collection (Sinhala split), only subsets with real generated answers, capped per subset
    lk       LKvaluesIT (si), --lk rows, at most one row per scenario, labels balanced; label kept in English
    uc_si    UltraChat-Sinhala train_sft (even md5 prompt_ids); the number of dialogues is SOLVED so that the
             Sinhala share of all tokens is --si-share (0.75)
  Shares are measured on rendered chat-template tokens (prompt + answers) of the SinLlama tokenizer.

Cleaning (Sinhala text from Aya and LKValues): lexicon-gated ZWJ repair (sinhala_normalize.py, the COCO pipeline's);
drop rows with "<unk>", empty fields, an answer with < 50% Sinhala letters, or > MAX_LEN tokens (UltraChat:
cut at the last complete user/assistant pair that fits instead). Exact-duplicate (user, answer) pairs dropped.
Validation: a fixed number of rows per source is held out (val.parquet keeps a 'source' column, so the trainer
reports per-source eval loss). The VQA val2017 benchmark images are disjoint from train2017.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import random
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

H = os.path.expanduser("~")
D = f"{H}/sinhala_instruct_datasets"
MODEL = f"{H}/sinllama-continual-pretraining/models/SinLlama_wiki_uc_gen_mix"
MAX_LEN = 2048

VQA_PROMPT = {  # = gen_sinllama.py PROMPTS (the wording used at test time)
    "en": ("Read the following description and answer the question.\n\n"
           "Description: {description}\n\nQuestion: {question}"),
    "si": ("පහත විස්තරය කියවා ප්‍රශ්නයට පිළිතුරු දෙන්න.\n\n"
           "විස්තරය: {description}\n\nප්‍රශ්නය: {question}"),
}
# Aya Collection subsets kept, with a cap on rows (the rest have one-word answers, garbled targets or mismatched tasks)
AYA_COLL_KEEP = {
    "Dolly-v2 (T)": 15000, "Flan-CoT-submix (T)": 20000, "SODA-inst (T)": 20000, "CNN-Daily-Mail (T)": 5000,
    "Flan-GEM-wiki-lingua (T)": 5000, "PIQA (T)": 5000, "NQ-Open (T)": 2500, "HotpotQA (T)": 2500,
}
VAL_N = {"vqa_si": 300, "vqa_en": 300, "uc_si": 200, "uc_en": 200, "aya_ds": 200, "aya_coll": 300, "lk": 100}
# the value label stays in English (as in the source file); the Sinhala answers name the value in Sinhala
LK_PROMPT = "පහත තත්ත්වය ශ්‍රී ලාංකීය අගයක් වන \"{value}\" සමඟ සම්බන්ධ වන්නේ කෙසේදැයි පැහැදිලි කරන්න.\n\nතත්ත්වය: {situation}"
SI = re.compile(r"[඀-෿]")


def si_frac(s):
    letters = [c for c in s if c.isalpha()]
    return sum(bool(SI.match(c)) for c in letters) / max(1, len(letters))


def md5_parity(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 2


class Renderer:
    def __init__(self, tok):
        self.tok = tok

    def n_tokens(self, msgs):
        return len(self.tok.apply_chat_template(msgs, tokenize=True))

    def fit(self, msgs):
        """Whole dialogue if it fits MAX_LEN, else the longest prefix of complete user/assistant pairs (None if
        not even the first pair fits)."""
        n = self.n_tokens(msgs)
        if n <= MAX_LEN:
            return msgs, n
        for k in range(len(msgs) - 2, 0, -2):
            n = self.n_tokens(msgs[:k])
            if n <= MAX_LEN:
                return msgs[:k], n
        return None, n


def pair(u, a):
    return [{"role": "user", "content": u.strip()}, {"role": "assistant", "content": a.strip()}]


def load_vqa(lang, path):
    """Loop-fixed file; the few records fix_loops.py could not repair (fix/report.json) are dropped."""
    rep = f"{H}/coco_train/fix/report.json"
    bad = {x["image_id"] for x in json.load(open(rep))["still_looping"]} if os.path.exists(rep) else set()
    out = []
    for r in json.load(open(path, encoding="utf-8")):
        if r["image_id"] in bad or not all(r[k].strip() for k in ("description", "question", "answer")):
            continue
        out.append({"id": f"coco{r['image_id']}",
                    "messages": pair(VQA_PROMPT[lang].format(description=r["description"].strip(),
                                                             question=r["question"].strip()), r["answer"])})
    return out


def load_aya_ds(norm):
    t = pq.read_table(glob.glob(f"{D}/aya_dataset/data/train-*.parquet")[0],
                      columns=["inputs", "targets", "language"]).to_pandas()
    t = t[t.language == "Sinhala"]
    return [{"id": f"ayads{i}", "messages": pair(norm(u), norm(a))}
            for i, (u, a) in enumerate(zip(t.inputs, t.targets))]


def load_aya_coll(norm, rng):
    by = collections.defaultdict(list)
    for fn in sorted(glob.glob(f"{D}/aya_collection_sinhala/sinhala/train-*.parquet")):
        pf = pq.ParquetFile(fn)
        for g in range(pf.num_row_groups):
            df = pf.read_row_group(g, columns=["id", "dataset_name", "inputs", "targets"]).to_pandas()
            df = df[df.dataset_name.isin(AYA_COLL_KEEP)]
            for i, dn, u, a in zip(df["id"], df.dataset_name, df.inputs, df.targets):
                by[dn].append((i, u, a))
    out, counts = [], {}
    for dn, rows in sorted(by.items()):
        rng.shuffle(rows)
        rows = rows[:int(AYA_COLL_KEEP[dn] * 1.3)]   # headroom for the filters, cut to the cap later
        counts[dn] = len(rows)
        out += [{"id": f"ayac{i}", "subset": dn, "messages": pair(norm(u), norm(a))} for i, u, a in rows]
    return out


def load_lk(n, rng):
    rows = json.load(open(f"{D}/lkvalues/train/lkvalues_it_si.json", encoding="utf-8"))
    rng.shuffle(rows)
    by_label = collections.defaultdict(list)
    seen = set()
    for r in rows:                 # one row per scenario (each scenario appears with several value labels)
        s = r["instruction"].strip()
        if s in seen or not r["input"].strip():
            continue
        seen.add(s)
        by_label[r["input"]].append(r)
    out, k = [], 0
    while len(out) < n and any(by_label.values()):     # round-robin over labels = balanced values
        for lab in sorted(by_label):
            if by_label[lab] and len(out) < n:
                r = by_label[lab].pop()
                out.append({"id": f"lk{k}", "messages": pair(
                    LK_PROMPT.format(value=lab.strip(), situation=r["instruction"].strip()), r["output"])})
                k += 1
    return out


def load_uc(path_glob, parity, lang):
    out = []
    for fn in sorted(glob.glob(path_glob)):
        pf = pq.ParquetFile(fn)
        for b in pf.iter_batches(batch_size=2000, columns=["prompt_id", "messages"]):
            for r in b.to_pylist():
                if md5_parity(r["prompt_id"]) != parity:
                    continue
                m = [{"role": x["role"], "content": x["content"].strip()} for x in r["messages"]]
                if not m or any(not x["content"] for x in m):
                    continue
                if [x["role"] for x in m] != ["user", "assistant"] * (len(m) // 2) or len(m) % 2:
                    continue
                out.append({"id": f"uc{lang}_{r['prompt_id'][:16]}", "messages": m})
    return out


def clean(rows, rend, sinhala, stats, name):
    c = collections.Counter()
    seen, out = set(), []
    for r in rows:
        m = r["messages"]
        text = "\n".join(x["content"] for x in m)
        if "<unk>" in text:
            c["unk"] += 1
            continue
        if sinhala and any(si_frac(x["content"]) < 0.5 for x in m if x["role"] == "assistant"):
            c["not_sinhala"] += 1
            continue
        key = (m[0]["content"], m[1]["content"])
        if key in seen:
            c["duplicate"] += 1
            continue
        seen.add(key)
        m2, n = rend.fit(m)
        if m2 is None:
            c["too_long"] += 1
            continue
        if len(m2) < len(m):
            c["truncated_turns"] += 1
        r["messages"], r["n_tokens"] = m2, n
        out.append(r)
    stats[name] = {"in": len(rows), "kept": len(out), **c}
    print(f"{name:9s} in {len(rows):7d} kept {len(out):7d} {dict(c)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{D}/mix")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--vqa-dir", default=f"{H}/coco_train/chunks")
    ap.add_argument("--uc-en", type=int, default=8000, help="English UltraChat dialogues")
    ap.add_argument("--lk", type=int, default=5000)
    ap.add_argument("--si-share", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    sys.path.insert(0, f"{H}/coco_train")
    from sinhala_normalize import SinhalaZwjNormalizer, load_lexicon, load_vocab_zwj
    zn = SinhalaZwjNormalizer(load_lexicon(None, f"{H}/coco_train/sinhala_lexicon.pkl"),
                              load_vocab_zwj(f"{H}/coco_train/unigram_32000_0.9995.vocab"), 5)
    norm = lambda s: zn.normalize(s.strip())  # noqa: E731
    rend = Renderer(AutoTokenizer.from_pretrained(args.model))
    stats = {}

    src = {
        "vqa_en": (clean(load_vqa("en", f"{args.vqa_dir}/vqa_train2017_english_100k_fixed.json"), rend, False, stats,
                         "vqa_en"), "en"),
        "vqa_si": (clean(load_vqa("si", f"{args.vqa_dir}/vqa_train2017_sinhala_100k_fixed.json"), rend, True, stats,
                         "vqa_si"), "si"),
        "aya_ds": (clean(load_aya_ds(norm), rend, True, stats, "aya_ds"), "si"),
        "lk": (clean(load_lk(args.lk + VAL_N["lk"], rng), rend, True, stats, "lk"), "si"),
    }
    coll = clean(load_aya_coll(norm, rng), rend, True, stats, "aya_coll")
    per = collections.defaultdict(list)
    for r in coll:
        per[r["subset"]].append(r)
    coll = [r for dn, rs in sorted(per.items()) for r in rs[:AYA_COLL_KEEP[dn] + VAL_N["aya_coll"] // len(per)]]
    stats["aya_coll_subsets"] = {dn: min(len(rs), AYA_COLL_KEEP[dn]) for dn, rs in per.items()}
    src["aya_coll"] = (coll, "si")

    uc_en = load_uc(f"{D}/ultrachat_en/data/train_sft-*.parquet", 1, "en")
    rng.shuffle(uc_en)
    src["uc_en"] = (clean(uc_en[:int((args.uc_en + VAL_N["uc_en"]) * 1.05)], rend, False, stats, "uc_en"), "en")
    src["uc_en"] = (src["uc_en"][0][:args.uc_en + VAL_N["uc_en"]], "en")
    uc_si = load_uc(f"{D}/ultrachat_si/*.parquet", 0, "si")
    rng.shuffle(uc_si)
    uc_si = clean(uc_si, rend, True, stats, "uc_si")

    # split off validation rows, then solve the number of Sinhala UltraChat dialogues for the target share
    train, val = {}, {}
    for name, (rows, lang) in src.items():
        rng.shuffle(rows)
        val[name], train[name] = rows[:VAL_N[name]], rows[VAL_N[name]:]
    val["uc_si"], uc_si_pool = uc_si[:VAL_N["uc_si"]], uc_si[VAL_N["uc_si"]:]
    tok = lambda rows: sum(r["n_tokens"] for r in rows)  # noqa: E731
    en_tok = sum(tok(train[n]) for n, (_, lang) in src.items() if lang == "en")
    si_fixed = sum(tok(train[n]) for n, (_, lang) in src.items() if lang == "si")
    need = en_tok * args.si_share / (1 - args.si_share) - si_fixed
    take, acc = 0, 0
    while take < len(uc_si_pool) and acc < need:
        acc += uc_si_pool[take]["n_tokens"]
        take += 1
    if acc < need:
        print(f"WARNING: Sinhala UltraChat pool too small: need {need / 1e6:.1f}M tokens, have {acc / 1e6:.1f}M")
    train["uc_si"] = uc_si_pool[:take]
    langs = {**{n: lang for n, (_, lang) in src.items()}, "uc_si": "si"}

    os.makedirs(args.out, exist_ok=True)
    for split, parts in (("train", train), ("val", val)):
        rows = [{"id": r["id"], "source": n, "lang": langs[n], "n_tokens": r["n_tokens"], "messages": r["messages"]}
                for n, rs in parts.items() for r in rs]
        if split == "train":
            rng.shuffle(rows)
        pq.write_table(pa.Table.from_pylist(rows), f"{args.out}/{split}.parquet", compression="zstd")

    tot = sum(tok(r) for r in train.values())
    summary = {n: {"lang": langs[n], "train_rows": len(train[n]), "val_rows": len(val[n]),
                   "train_tokens": tok(train[n]), "token_share": round(tok(train[n]) / tot, 4)} for n in train}
    lang_tok = collections.Counter()
    for n in train:
        lang_tok[langs[n]] += tok(train[n])
    stats["summary"] = summary
    stats["total"] = {"train_rows": sum(len(r) for r in train.values()), "val_rows": sum(len(r) for r in val.values()),
                      "train_tokens": tot, "si_share": round(lang_tok["si"] / tot, 4),
                      "en_share": round(lang_tok["en"] / tot, 4)}
    json.dump(stats, open(f"{args.out}/stats.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n{'source':9s} {'lang':4s} {'rows':>8s} {'tokens':>12s} share")
    for n, s in summary.items():
        print(f"{n:9s} {s['lang']:4s} {s['train_rows']:8d} {s['train_tokens']:12,d} {100 * s['token_share']:5.1f}%")
    print("total", stats["total"])


if __name__ == "__main__":
    main()
