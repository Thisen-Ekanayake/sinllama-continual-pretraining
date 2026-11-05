#!/usr/bin/env python3
"""Translate the VQA dataset (description, question, answer) from English to
Sinhala (sin_Sinh) with NLLB-200 3.3B.

NLLB is a sentence-level model, so every field is split into paragraphs and
sentences, each unique sentence is translated once, and the field is rebuilt
in the original order.

Known NLLB en->si failure modes handled here (see also
/ml/Datasets/UltraChat/docs/sinhala_zwj_repair.md):

  * ZWJ loss: NLLB's SentencePiece normaliser turns the Zero-Width Joiner of
    Sinhala conjuncts into a space (ප්‍ර -> ප් ර). Restored on the decoded text
    by the lexicon-gated SinhalaZwjNormalizer from the UltraChat pipeline
    (sinhala_normalize.py, lexicon + tokenizer vocab in resources/, copied from
    /ml/Datasets/UltraChat/{data/sinhala_lexicon.pkl,tokenizer/unigram_32000_0.9995.vocab}).
  * Hallucinated stock phrases: some ordinary sentences translate to
    "[23වන පිටුවේ පින්තූරය]" ("[picture on page 23]") or
    "[සංස්කරණය කරන ලද පිටුව]" ("[edited page]"), deterministically (same in
    fp16/bf16/fp32, beam or greedy). Dropping the final full stop avoids it.
  * Omission: long multi-clause sentences sometimes come back as a
    translation of only the last clause (target/source length ratio << 1).
  * Repetition loops (output runs to max_new_tokens).

Each sentence goes through up to three attempts, stopping at the first that
passes the checks: (1) as is, (2) final punctuation removed and re-appended,
(3) split into clauses, each translated as in (2), and re-joined. If all fail,
the best candidate is kept and the sentence is listed in <out>.flags.jsonl.

Raw NLLB output is cached in <out>.cache.jsonl, so an interrupted run resumes.

This is step 1 of 3. The JSON written here only has the ZWJ repair; the release file
comes from:
  2. repair_nllb.py   (GPU) re-translates the ~1.6% still broken (loops, residual
                      hallucinations, dropped clauses) -> <out>.cache.repair.jsonl
  3. postprocess_si.py (CPU) rebuilds <out> from both caches: ZWJ, NLLB's missing
                      rare letters ("සං signs ා" -> සංඥා), stem-prefix ZWJ

Usage (lab box, ~/tts_env):
  python translate_nllb.py --inp vqa_coco_val2017.json --out vqa_sinhala.json
  python translate_nllb.py --inp vqa_coco_val2017.json --out vqa_sample10_si.json --limit 10
"""
import argparse
import json
import os
import re
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sinhala_normalize import SinhalaZwjNormalizer, load_lexicon, load_vocab_zwj

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ("description", "question", "answer")
# sentence boundary: . ! ? (optionally followed by a closing quote/bracket), whitespace,
# then something that starts a sentence. "J.P.Morgan", "3.5 m" etc. have no space and stay whole.
SENT_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”’)]))\s+(?=[A-Z0-9\"“‘(])")
FINAL_PUNCT = re.compile(r"[.!?]+$")
# clause boundaries for the last-resort split: after , before a conjunction / participle,
# or at ; : and dashes
CLAUSE_SPLIT = re.compile(
    r"(?<=[,;:])\s+(?=(?:but|while|which|who|allowing|as|because|so|whereas|suggesting|"
    r"indicating|making|with|where|though|although|creating|giving|adding|since|yet|"
    r"providing|showing|likely|possibly|perhaps|then|leaving|helping)\b)"
    r"|(?<=;)\s+|\s+[—–]\s+|—")
HALLUCINATION = re.compile(r"පිටුවේ පින්තූරය|සංස්කරණය කරන ලද|^\s*\[")
MIN_RATIO, MIN_RATIO_WORDS = 0.55, 12   # median ratio of good translations ~0.98


def split_sentences(text):
    """-> list of paragraphs, each a list of sentences."""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    return [[s.strip() for s in SENT_SPLIT.split(p) if s.strip()] for p in paras]


def problem(src, tgt, hit_max):
    """Reason the translation looks broken, or None."""
    if not tgt.strip():
        return "empty"
    if hit_max:
        return "loop"
    if HALLUCINATION.search(tgt) and "[" not in src:
        return "hallucination"
    if len(src.split()) >= MIN_RATIO_WORDS and len(tgt) / len(src) < MIN_RATIO:
        return "short"
    return None


def split_clauses(s):
    # only at conjunction / dash boundaries: splitting a list at plain commas
    # ("No people, animals ..., or text are present") loses the negation
    parts = [p for p in CLAUSE_SPLIT.split(s) if p and p.strip()]
    return [p.strip() for p in parts] if len(parts) >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="vqa_coco_val2017.json")
    ap.add_argument("--out", default="vqa_sinhala.json")
    ap.add_argument("--model", default=os.path.expanduser("~/nllb"))
    ap.add_argument("--src", default="eng_Latn")
    ap.add_argument("--tgt", default="sin_Sinh")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--tok-budget", type=int, default=1024,
                    help="max (batch size x longest source length) per generate call")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--lexicon", default=os.path.join(HERE, "resources/sinhala_lexicon.pkl"))
    ap.add_argument("--vocab", default=os.path.join(HERE, "resources/unigram_32000_0.9995.vocab"))
    args = ap.parse_args()

    data = json.load(open(args.inp, encoding="utf-8"))
    if args.limit:
        data = data[:args.limit]

    structure, unique = [], {}
    for r in data:
        rec = {}
        for k in FIELDS:
            rec[k] = split_sentences(r[k])
            for p in rec[k]:
                for s in p:
                    unique.setdefault(s, None)
        structure.append(rec)

    base = os.path.splitext(args.out)[0]
    cache_path = base + ".cache.jsonl"
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    cache[j["src"]] = j
    todo = [s for s in unique if s not in cache]
    print(f"{len(data)} records, {len(unique)} unique sentences, {len(unique) - len(todo)} cached, "
          f"{len(todo)} to translate", flush=True)

    if todo:
        tok = AutoTokenizer.from_pretrained(args.model, src_lang=args.src)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                      low_cpu_mem_usage=True).cuda().eval()
        tgt_id = tok.convert_tokens_to_ids(args.tgt)

        def generate(sents):
            enc = tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=1000).to("cuda")
            max_new = min(1000, int(enc.input_ids.shape[1] * 2.5) + 16)
            with torch.inference_mode():
                out = model.generate(**enc, forced_bos_token_id=tgt_id, num_beams=args.beams,
                                     max_new_tokens=max_new)
            texts = [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]
            # never emitted EOS -> ran to max_new_tokens (decoder start + forced bos counted)
            hit_max = [(row != tok.pad_token_id).sum().item() >= max_new for row in out]
            return list(zip(texts, hit_max))

        def safe_generate(sents):
            try:
                return generate(sents)
            except torch.cuda.OutOfMemoryError:
                if len(sents) == 1:
                    raise
            # retry outside the except block: the live exception would otherwise keep
            # the failed batch's tensors referenced and every retry would OOM too
            torch.cuda.empty_cache()
            h = len(sents) // 2
            print(f"  OOM at batch {len(sents)}, splitting", flush=True)
            return safe_generate(sents[:h]) + safe_generate(sents[h:])

        def translate_many(sents):
            """Batched translation of arbitrary strings -> {s: (text, hit_max)}."""
            sents = list(dict.fromkeys(sents))
            lens = {s: len(tok(s).input_ids) for s in sents}
            sents.sort(key=lambda s: lens[s], reverse=True)   # longest first: OOM shows up early
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
            return t if not p or t.endswith(p[-1]) else t + p

        chunk = 2000          # sentences per round, so the cache is written as we go
        t0, done = time.time(), 0
        with open(cache_path, "a", encoding="utf-8") as fout:
            for c0 in range(0, len(todo), chunk):
                part = todo[c0:c0 + chunk]
                cands = {s: [] for s in part}          # (stage, text, reason)
                # stage 1: as is
                r1 = translate_many(part)
                bad = []
                for s in part:
                    t, hm = r1[s]
                    why = problem(s, t, hm)
                    cands[s].append(("plain", t, why))
                    if why:
                        bad.append(s)
                # stage 2: without final punctuation, re-appended
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
                # stage 3: clause split, each clause as in stage 2
                if bad:
                    def clause_core(c):       # clause without trailing , ; : . ! ?
                        return strip_punct(c.rstrip(",;: "))[0]

                    plans = {s: split_clauses(strip_punct(s)[0]) for s in bad}
                    pieces = [clause_core(c) for cl in plans.values() if cl for c in cl]
                    r3 = translate_many(pieces) if pieces else {}
                    for s in bad:
                        cl = plans[s]
                        if not cl:
                            continue
                        outs, broken = [], False
                        for c in cl:
                            t, hm = r3[clause_core(c)]
                            if HALLUCINATION.search(t) or hm or not t:
                                broken = True
                            sep = c.rstrip()[-1] if c.rstrip()[-1] in ",;:" else ""
                            outs.append(strip_punct(t.rstrip(",;: "))[0] + sep)
                        t = with_punct(" ".join(outs), strip_punct(s)[1])
                        why = "clause_hallucination" if broken else problem(s, t, False)
                        cands[s].append(("clauses", t, why))

                for s in part:
                    ok = [c for c in cands[s] if c[2] is None]
                    if ok:
                        stage, t, _ = ok[0]
                        j = {"src": s, "tgt": t, "stage": stage}
                    else:
                        # nothing passed: prefer non-hallucinated, then the longest
                        stage, t, why = max(cands[s], key=lambda c: (
                            c[2] not in ("hallucination", "clause_hallucination", "empty", "loop"), len(c[1])))
                        j = {"src": s, "tgt": t, "stage": stage, "flag": why,
                             "candidates": [{"stage": a, "tgt": b, "why": w} for a, b, w in cands[s]]}
                    cache[s] = j
                    fout.write(json.dumps(j, ensure_ascii=False) + "\n")
                fout.flush()
                done += len(part)
                el = time.time() - t0
                print(f"sentences {done}/{len(todo)} {el/60:.1f} min, "
                      f"ETA {el/done*(len(todo)-done)/60:.1f} min", flush=True)

    # ZWJ restoration on the decoded text, then rebuild the records
    norm = SinhalaZwjNormalizer(load_lexicon(None, args.lexicon), load_vocab_zwj(args.vocab), 5)
    fixed = {s: norm.normalize(cache[s]["tgt"]) for s in unique}
    out = []
    for r, rec in zip(data, structure):
        o = {"image_id": r["image_id"]}
        for k in FIELDS:
            o[k] = "\n\n".join(" ".join(fixed[s] for s in p) for p in rec[k])
        out.append(o)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    stages, flags = {}, []
    for s in unique:
        j = cache[s]
        stages[j["stage"]] = stages.get(j["stage"], 0) + 1
        if j.get("flag"):
            flags.append(j)
    with open(base + ".flags.jsonl", "w", encoding="utf-8") as f:
        for j in flags:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} records to {args.out}; sentences by accepted stage {stages}; "
          f"{len(flags)} unresolved (see {base}.flags.jsonl); ZWJ merges {norm.merges}")


if __name__ == "__main__":
    main()
