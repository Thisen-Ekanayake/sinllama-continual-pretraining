#!/usr/bin/env python3
"""Second pass over the NLLB sentence cache: re-translate the sentences whose
translation is still broken, with more strategies than translate_nllb.py.

Broken = hallucinated stock phrase ("[23වන පිටුවේ පින්තූරය]"), repetition loop
(tandem repeats, frequent 3-grams, or hit max length), or omission (target/source
length ratio < 0.55; typically one half of a two-part question is dropped).

Candidates per broken sentence (the existing translation stays a candidate):
  lc        first letter lower-cased, final punctuation removed and re-appended
  nr        n-gram blocking (no_repeat_ngram_size=3, repetition_penalty=1.15)
  lc_nr     both
  split     split at clause boundaries incl. ", and" / ", or" and before a second
            question word ("..., and how ..."); each part translated with lc_nr;
            a two-part question becomes two Sinhala questions
Among the passing deterministic candidates, the one whose length ratio is closest
to a normal translation's wins. Only when none passes:
  mbr       24 samples (temperature 0.6, top_p 0.9; 16 of the sentence, 8 of its
            lower-cased form); among the passing samples, pick the one with the
            highest mean chrF against all other candidates (minimum-Bayes-risk
            consensus), so a sample that drifts in meaning ("cat" -> "dog") loses.
            A single high-temperature sample was tried first and garbled ~1 in 3.
If nothing passes, the least-bad candidate is kept and listed in the report.

Writes <cache stem>.repair.jsonl (same records as the cache; later lines override
earlier ones in postprocess_si.py) and <cache stem>.repair_report.jsonl.

Usage (lab box, ~/tts_env):
  python repair_nllb.py --cache vqa_sinhala.cache.jsonl
"""
import argparse
import collections
import json
import math
import os
import re
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

HALLUCINATION = re.compile(r"පිටුවේ පින්තූරය|සංස්කරණය කරන ලද|^\s*\[")
FINAL_PUNCT = re.compile(r"[.!?]+$")
QWORD = r"(?:how|what|why|which|where|when|whether|who|does|do|is|are|can|could|would|might)"
SPLIT = re.compile(
    rf"(?<=,)\s+(?=(?:and|or|but|while|which|who|allowing|as|because|so|whereas|suggesting|"
    rf"indicating|making|with|where|though|although|creating|giving|since|yet|providing)\b)"
    rf"|(?<=;)\s+|\s+[—–]\s+|—|\s+(?=and\s+{QWORD}\b)")
TARGET_RATIO, MIN_RATIO, MIN_RATIO_WORDS = 0.95, 0.55, 12


def is_loop(t, hit_max=False):
    if hit_max:
        return True
    w = t.split()
    for n in (1, 2, 3):          # the same 1-3 word unit three times in a row
        for i in range(len(w) - 3 * n + 1):
            if w[i:i + n] == w[i + n:i + 2 * n] == w[i + 2 * n:i + 3 * n]:
                return True
    g = collections.Counter(tuple(w[i:i + 3]) for i in range(len(w) - 2))
    return bool(g) and max(g.values()) >= 4


def problem(src, tgt, hit_max=False):
    if not tgt.strip():
        return "empty"
    if HALLUCINATION.search(tgt) and "[" not in src:
        return "hallucination"
    if is_loop(tgt, hit_max):
        return "loop"
    if len(src.split()) >= MIN_RATIO_WORDS and len(tgt) / len(src) < MIN_RATIO:
        return "short"
    return None


def chrf(hyp, ref, n=6, beta=2.0):
    """Character n-gram F-score (chrF, Popovic 2015), whitespace removed."""
    h, r = hyp.replace(" ", ""), ref.replace(" ", "")
    precs, recs = [], []
    for k in range(1, n + 1):
        hg = collections.Counter(h[i:i + k] for i in range(len(h) - k + 1))
        rg = collections.Counter(r[i:i + k] for i in range(len(r) - k + 1))
        if not hg or not rg:
            continue
        m = sum((hg & rg).values())
        precs.append(m / sum(hg.values()))
        recs.append(m / sum(rg.values()))
    if not precs:
        return 0.0
    p, rc = sum(precs) / len(precs), sum(recs) / len(recs)
    return 0.0 if p + rc == 0 else (1 + beta ** 2) * p * rc / (beta ** 2 * p + rc)


SEVERITY = {None: 0, "short": 1, "loop": 2, "hallucination": 3, "empty": 4}


def strip_punct(s):
    m = FINAL_PUNCT.search(s)
    return (s[:m.start()].rstrip(), m.group(0)) if m else (s, "")


def with_punct(t, p):
    t = t.rstrip()
    return t if not p or t.endswith(p[-1]) else t + p


def lower_first(s):
    # keep acronyms / "I" (two capitals, or a lone I) as they are
    if len(s) > 1 and s[0].isupper() and not s[1].isupper() and not s.startswith("I "):
        return s[0].lower() + s[1:]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="vqa_sinhala.cache.jsonl")
    ap.add_argument("--model", default=os.path.expanduser("~/nllb"))
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--tok-budget", type=int, default=1024)
    ap.add_argument("--max-batch", type=int, default=32)
    args = ap.parse_args()

    cache = {}
    for line in open(args.cache, encoding="utf-8"):
        if line.strip():
            j = json.loads(line)
            cache[j["src"]] = j
    bad = [s for s, j in cache.items() if problem(s, j["tgt"]) or j.get("flag") in ("loop", "hallucination",
                                                                                     "clause_hallucination")]
    print(f"{len(cache)} cached sentences, {len(bad)} to repair "
          f"{collections.Counter(problem(s, cache[s]['tgt']) for s in bad)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True).cuda().eval()
    tgt_id = tok.convert_tokens_to_ids("sin_Sinh")

    def generate(sents, n_ret, kw):
        enc = tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=1000).to("cuda")
        max_new = min(1000, int(enc.input_ids.shape[1] * 2.5) + 16)
        with torch.inference_mode():
            out = model.generate(**enc, forced_bos_token_id=tgt_id, max_new_tokens=max_new,
                                 num_return_sequences=n_ret, **kw)
        texts = [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]
        hit = [(row != tok.pad_token_id).sum().item() >= max_new for row in out]
        return [list(zip(texts[i * n_ret:(i + 1) * n_ret], hit[i * n_ret:(i + 1) * n_ret]))
                for i in range(len(sents))]

    def safe_generate(sents, n_ret, kw):
        try:
            return generate(sents, n_ret, kw)
        except torch.cuda.OutOfMemoryError:
            if len(sents) == 1:
                raise
        torch.cuda.empty_cache()   # retry outside the except block (see translate_nllb.py)
        h = len(sents) // 2
        return safe_generate(sents[:h], n_ret, kw) + safe_generate(sents[h:], n_ret, kw)

    def run(sents, mode, n_samples=1):
        """-> {input: [(text, hit_max), ...]}"""
        kw = {"beam": dict(num_beams=args.beams),
              "nr": dict(num_beams=args.beams, no_repeat_ngram_size=3, repetition_penalty=1.15),
              "sample": dict(do_sample=True, temperature=0.6, top_p=0.9, num_beams=1)}[mode]
        n_ret = n_samples if mode == "sample" else 1
        budget = args.tok_budget // n_ret if mode == "sample" else args.tok_budget
        sents = list(dict.fromkeys(sents))
        lens = {s: len(tok(s).input_ids) for s in sents}
        sents.sort(key=lambda s: lens[s], reverse=True)
        res, cur = {}, []
        for s in sents + [None]:
            if cur and (s is None or (len(cur) + 1) * lens[cur[0]] > budget or len(cur) >= args.max_batch):
                res.update(zip(cur, safe_generate(cur, n_ret, kw)))
                cur = []
            if s is not None:
                cur.append(s)
        return res

    torch.manual_seed(0)
    t0 = time.time()
    cands = {s: [("original", cache[s]["tgt"], problem(s, cache[s]["tgt"]) or cache[s].get("flag"))]
             for s in bad}
    core = {s: strip_punct(s) for s in bad}

    lc_in = {s: lower_first(core[s][0]) for s in bad}
    r_lc = run(list(lc_in.values()), "beam")
    r_nr = run([core[s][0] for s in bad], "nr")
    r_lcnr = run(list(lc_in.values()), "nr")
    for s in bad:
        p = core[s][1]
        for name, (t, hm) in (("lc", r_lc[lc_in[s]][0]), ("nr", r_nr[core[s][0]][0]),
                              ("lc_nr", r_lcnr[lc_in[s]][0])):
            t = with_punct(t, p)
            cands[s].append((name, t, problem(s, t, hm)))

    # clause split
    plans = {}
    for s in bad:
        parts = [x.strip() for x in SPLIT.split(core[s][0]) if x and x.strip()]
        if len(parts) >= 2:
            plans[s] = parts
    piece_in = {}
    for s, parts in plans.items():
        q = core[s][1].endswith("?")
        for x in parts:
            x0 = re.sub(r"^(?:and|or)\s+", "", x.rstrip(",;: ")) if q else x.rstrip(",;: ")
            piece_in[(s, x)] = (x0[0].upper() + x0[1:] + "?") if q else lower_first(x0)
    r_pc = run(list(piece_in.values()), "nr") if piece_in else {}
    for s, parts in plans.items():
        q = core[s][1].endswith("?")
        outs, broken = [], None
        for x in parts:
            t, hm = r_pc[piece_in[(s, x)]][0]
            why = problem(x, t, hm)
            if why in ("hallucination", "loop", "empty"):
                broken = why
            if q:
                outs.append(with_punct(t, "?"))
            else:
                sep = x.rstrip()[-1] if x.rstrip()[-1] in ",;:" else ","
                outs.append(strip_punct(t.rstrip(",;: "))[0] + sep)
        t = " ".join(outs) if q else with_punct(" ".join(outs).rstrip(",;: "), core[s][1])
        cands[s].append(("split", t, broken or problem(s, t)))

    def score(src, c):
        _, t, why = c
        ratio = max(len(t), 1) / len(src)
        return (SEVERITY.get(why, 3), abs(math.log(ratio / TARGET_RATIO)))

    # MBR sampling only where no deterministic candidate passes
    need = [x for x in bad if not any(c[2] is None for c in cands[x])]
    print(f"{len(bad) - len(need)} fixed deterministically, {len(need)} need MBR sampling", flush=True)
    if need:
        r_s1 = run([core[x][0] for x in need], "sample", 16)
        r_s2 = run([lc_in[x] for x in need], "sample", 8)
        for x in need:
            p = core[x][1]
            pool = r_s1[core[x][0]] + r_s2[lc_in[x]]
            samples = [(f"mbr{k}", with_punct(t, p), problem(x, with_punct(t, p), hm))
                       for k, (t, hm) in enumerate(pool)]
            refs = [c[1] for c in cands[x] + samples if c[2] in (None, "short")]
            ok = [c for c in samples if c[2] is None]
            if ok and len(refs) > 1:
                def consensus(c):
                    return sum(chrf(c[1], r) for r in refs if r is not c[1]) / (len(refs) - 1)
                best = max(ok, key=consensus)
                cands[x].append(("mbr", best[1], None))
            cands[x].extend(c for c in samples if c[2] is not None)  # report only

    stem = os.path.splitext(args.cache)[0].replace(".cache", "")
    fixed_n, outcome = 0, collections.Counter()
    with open(stem + ".cache.repair.jsonl", "w", encoding="utf-8") as fout, \
            open(stem + ".repair_report.jsonl", "w", encoding="utf-8") as frep:
        for s in bad:
            best = min(cands[s], key=lambda c: score(s, c))
            name, t, why = best
            outcome[(name if why is None else f"unresolved:{why}")] += 1
            j = {"src": s, "tgt": t, "stage": f"repair_{name}"}
            if why:
                j["flag"] = why
            fout.write(json.dumps(j, ensure_ascii=False) + "\n")
            frep.write(json.dumps({"src": s, "chosen": name, "why": why,
                                   "candidates": [{"name": a, "tgt": b, "why": w} for a, b, w in cands[s]]},
                                  ensure_ascii=False) + "\n")
    print(f"repaired in {(time.time() - t0) / 60:.1f} min: {dict(outcome.most_common())}")


if __name__ == "__main__":
    main()
