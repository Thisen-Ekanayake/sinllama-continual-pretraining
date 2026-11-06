#!/usr/bin/env python3
"""Translate SinLlama's Sinhala VQA answers back to English with NLLB-200 3.3B (fp16).

Sentence-level, like ../translate_nllb.py, run in the other direction (sin_Sinh -> eng_Latn).
Each unique sentence goes through up to three attempts, stopping at the first that passes:
(1) as is, (2) final punctuation removed and re-appended, (3) split at , ; and each clause
translated as in (2). Checks: empty, ran to max_new_tokens or repeats a 4-gram 4+ times
(loop), "[" invented, or English/Sinhala character ratio < 0.6 on 8+ word sentences
(dropped clauses; the dataset's own en/si answer ratio is median 1.06, p1 0.87).

  python translate_si_en.py --inp sinllama_si_generated.json --out sinllama_si_translated_en.json
"""
import argparse, json, os, re, time
from collections import Counter

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# boundary after . ! ? (not "1." list numbers) followed by whitespace
SENT_SPLIT = re.compile(r"(?<![0-9])(?<=[.!?])\s+(?=\S)")
FINAL_PUNCT = re.compile(r"[.!?]+$")
CLAUSE_SPLIT = re.compile(r"(?<=[,;])\s+")
MIN_RATIO, MIN_RATIO_WORDS = 0.6, 8


def split_sentences(text):
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    return [[s.strip() for s in SENT_SPLIT.split(p) if s.strip()] for p in paras]


def looped(t):
    w = t.split()
    grams = Counter(tuple(w[i:i + 4]) for i in range(len(w) - 3))
    return bool(grams) and max(grams.values()) >= 4


def problem(src, tgt, hit_max):
    if not tgt.strip():
        return "empty"
    if hit_max or looped(tgt):
        return "loop"
    if "[" in tgt and "[" not in src:
        return "hallucination"
    if len(src.split()) >= MIN_RATIO_WORDS and len(tgt) / len(src) < MIN_RATIO:
        return "short"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.expanduser("~/nllb_fp16"))
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--tok-budget", type=int, default=1024)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--device", default="cuda", help="cpu only for smoke tests (loads fp32)")
    args = ap.parse_args()

    src_doc = json.load(open(args.inp, encoding="utf-8"))
    data = src_doc["records"]
    structure, unique = [], {}
    for r in data:
        paras = split_sentences(r["generated_answer"])
        for p in paras:
            for s in p:
                unique.setdefault(s, None)
        structure.append(paras)
    print(f"{len(data)} answers, {len(unique)} unique sentences", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, src_lang="sin_Sinh")
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=dtype,
                                                  low_cpu_mem_usage=True).to(args.device).eval()
    tgt_id = tok.convert_tokens_to_ids("eng_Latn")

    def generate(sents):
        enc = tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=1000).to(args.device)
        max_new = min(1000, int(enc.input_ids.shape[1] * 2.0) + 16)
        with torch.inference_mode():
            out = model.generate(**enc, forced_bos_token_id=tgt_id, num_beams=args.beams,
                                 max_new_tokens=max_new)
        texts = [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]
        hit_max = [(row != tok.pad_token_id).sum().item() >= max_new for row in out]
        return list(zip(texts, hit_max))

    def safe_generate(sents):
        try:
            return generate(sents)
        except torch.cuda.OutOfMemoryError:
            if len(sents) == 1:
                raise
        torch.cuda.empty_cache()     # retry outside except so the failed batch is freed
        h = len(sents) // 2
        return safe_generate(sents[:h]) + safe_generate(sents[h:])

    def translate_many(sents):
        sents = list(dict.fromkeys(sents))
        lens = {s: len(tok(s).input_ids) for s in sents}
        sents.sort(key=lambda s: lens[s], reverse=True)
        res, cur = {}, []
        for s in sents + [None]:
            if cur and (s is None or (len(cur) + 1) * lens[cur[0]] > args.tok_budget
                        or len(cur) >= args.max_batch):
                res.update(zip(cur, safe_generate(cur)))
                cur = []
            if s is not None:
                cur.append(s)
        return res

    def strip_punct(s):
        m = FINAL_PUNCT.search(s)
        return (s[:m.start()].rstrip(), m.group(0)) if m else (s, "")

    def with_punct(t, p):
        return t if not p or t[-1:] in ".!?" else t + p

    t0 = time.time()
    sents = list(unique)
    cands = {s: [] for s in sents}
    r1 = translate_many(sents)
    bad = []
    for s in sents:
        t, hm = r1[s]
        why = problem(s, t, hm)
        cands[s].append(("plain", t, why))
        if why:
            bad.append(s)
    print(f"stage 1: {len(sents)} sentences, {len(bad)} flagged, {(time.time() - t0) / 60:.1f} min", flush=True)
    if bad:
        r2 = translate_many([strip_punct(s)[0] for s in bad])
        still = []
        for s in bad:
            core, p = strip_punct(s)
            t, hm = r2[core]
            t = with_punct(t, p)
            why = problem(s, t, hm)
            cands[s].append(("nopunct", t, why))
            if why:
                still.append(s)
        bad = still
        print(f"stage 2: {len(bad)} still flagged", flush=True)
    if bad:
        plans = {s: [c for c in CLAUSE_SPLIT.split(strip_punct(s)[0]) if c.strip()] for s in bad}
        core = lambda c: c.rstrip(",; ")
        pieces = [core(c) for cl in plans.values() if len(cl) > 1 for c in cl]
        r3 = translate_many(pieces) if pieces else {}
        for s in bad:
            cl = plans[s]
            if len(cl) < 2:
                continue
            outs, broken = [], False
            for c in cl:
                t, hm = r3[core(c)]
                broken |= hm or not t or looped(t)
                sep = c.rstrip()[-1] if c.rstrip()[-1] in ",;" else ""
                outs.append(strip_punct(t.rstrip(",; "))[0] + sep)
            t = with_punct(" ".join(outs), strip_punct(s)[1])
            cands[s].append(("clauses", t, "clause_loop" if broken else problem(s, t, False)))

    final, stages, flags = {}, Counter(), []
    for s in sents:
        ok = [c for c in cands[s] if c[2] is None]
        if ok:
            stage, t, why = ok[0]
        else:
            stage, t, why = max(cands[s], key=lambda c: (c[2] not in ("loop", "empty", "clause_loop"), len(c[1])))
            flags.append({"src": s, "tgt": t, "stage": stage, "why": why,
                          "candidates": [{"stage": a, "tgt": b, "why": w} for a, b, w in cands[s]]})
        final[s] = t
        stages[stage] += 1

    flagged = {f["src"] for f in flags}
    out = []
    for r, paras in zip(data, structure):
        en = "\n\n".join(" ".join(final[s] for s in p) for p in paras)
        out.append({"image_id": r["image_id"], "question_si": r["question"],
                    "generated_answer_si": r["generated_answer"],
                    "translated_answer_en": en,
                    "translation_flagged": any(s in flagged for p in paras for s in p)})
    meta = {"translator": "facebook/nllb-200-3.3B (fp16)", "direction": "sin_Sinh -> eng_Latn",
            "beams": args.beams, "unit": "sentence", "n": len(out), "unique_sentences": len(sents),
            "accepted_stage_counts": dict(stages), "unresolved_sentences": len(flags),
            "source_file": os.path.basename(args.inp)}
    json.dump({"meta": meta, "records": out}, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.splitext(args.out)[0] + ".flags.jsonl", "w", encoding="utf-8") as f:
        for j in flags:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: stages {dict(stages)}, {len(flags)} unresolved, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
