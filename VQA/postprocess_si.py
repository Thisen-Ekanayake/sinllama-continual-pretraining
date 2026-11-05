#!/usr/bin/env python3
"""Rebuild vqa_sinhala.json from the NLLB sentence cache with text-level repairs.

Repairs (in order, all lexicon-gated with the UltraChat Sinhala lexicon):
  1. ZWJ restoration (SinhalaZwjNormalizer; ප් ර -> ප්‍ර).
  2. Rare-letter restoration. NLLB almost never generates some Sinhala letters
     (ඥ ඈ ඬ ඵ ඝ ...; ඓ is not even in its vocabulary) and copies the English
     source word into the gap instead: "සං signs ා" (සංඥා), "away ත්" (ඈත්),
     "නිහ quiet" (නිහඬ), "historical තිහාසික" (ඓතිහාසික). A lowercase English
     word that also occurs in the source sentence and sits next to a Sinhala
     fragment is replaced by the one letter that turns fragment(s) + letter into
     an attested word (lexicon frequency >= MIN_FREQ).
  3. ZWJ again (the merged word can expose new conjunct sites), then the stem-prefix
     extension for conjuncts in inflections the lexicon lacks (see make_stem_prefix_joiner).
Also collapses doubled final punctuation left by re-appending the source's mark (".?").

Runs locally (no GPU):
  python postprocess_si.py --en vqa_english.json \
      --cache vqa_sinhala.cache.jsonl vqa_sinhala.cache.repair.jsonl --out vqa_sinhala.json
"""
import argparse
import collections
import json
import os
import re

from sinhala_normalize import SinhalaZwjNormalizer, load_lexicon, load_vocab_zwj

FIELDS = ("description", "question", "answer")
MIN_FREQ = 5
SI = r"඀-෿‍"
# letters NLLB never / rarely generates (share in its output vs in the lexicon:
# 0 for ඃ ඍ ඎ ඓ ඖ ඞ ඣ ඦ ඪ ෟ ෲ ෳ, 0.02-0.5 for ඤ ෛ ඕ ඨ ඌ ඵ ෞ ඊ ඬ ඝ); ඥ ඈ are seen as gaps too.
# Common letters are excluded: allowing them produced false merges (away ත් -> පත්);
# so are ඕ and ඊ, which only ever won ambiguous gaps wrongly (thick න -> ඕන, not ඝන).
FILL = list("ඃඍඎඓඖඞඣඦඪෟෲෳඤෛඨඌඵෞඬඝඥඈ")
COMBINING = re.compile(r"^[ඁ-ඃ්-ෟෲෳ]")
LATIN_GAP = re.compile(rf"(?:(?P<L>[{SI}]+) )?(?P<w>[a-z]+)(?: (?P<R>[{SI}]+))?")


ZWJ = "\u200d"
HAL_SPACE = re.compile(rf"([{SI}]+\u0DCA) ([{SI}]+)")


def make_stem_prefix_joiner(W):
    """ZWJ for conjuncts in inflections the lexicon lacks (the stem-prefix extension
    proposed in UltraChat docs/sinhala_zwj_repair.md section 8). Join L + ZWJ + R when R
    is not a word by itself and an attested ZWJ word starts with L + ZWJ + R[0]:
    දෘශ් යතාවයක් -> දෘශ්‍යතාවයක්, හයිඩ් රන්ට් -> හයිඩ්‍රන්ට්, while
    ටෙනිස් රැකට්ටුවක් (no word starts with ටෙනිස්‍ර) and බස් රථය (රථය is a word) stay."""
    prefixes = set()
    for w, f in W.items():
        if f >= MIN_FREQ and ZWJ in w:
            for i, ch in enumerate(w):
                if ch == ZWJ and i + 1 < len(w):
                    prefixes.add(w[:i + 2])

    def join(text, stats):
        def repl(m):
            L, R = m.group(1), m.group(2)
            if R[0] in "රයව" and W.get(R, 0) < MIN_FREQ and (L + ZWJ + R[0]) in prefixes:
                stats["stem_prefix"] += 1
                return L + ZWJ + R
            return m.group(0)
        prev = None
        while prev != text:            # chains: one site per pass can expose the next
            prev, text = text, HAL_SPACE.sub(repl, text)
        return text
    return join


def split_sentences_factory():
    # same splitter as translate_nllb.py
    sent = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”’)]))\s+(?=[A-Z0-9\"“‘(])")

    def split(text):
        paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
        return [[s.strip() for s in sent.split(p) if s.strip()] for p in paras]
    return split


def make_rare_letter_fixer(W):
    def is_word(x):
        return W.get(x, 0) >= MIN_FREQ

    def fix(src, tgt, stats):
        src_words = set(re.findall(r"[a-z]+", src.lower()))

        def from_source(w):
            # the filler is a source word, a piece of one ("ness" < "greatness", "age" <
            # "message"), or another form of one ("historical" ~ "historic")
            # or shares a stem with one ("signs" ~ "signage")
            return w in src_words or any(
                (len(w) >= 3 and w in x) or (len(x) >= 5 and x in w)
                or (len(os.path.commonprefix([w, x])) >= max(4, 0.6 * len(w))) for x in src_words)

        def repl(m):
            L, w, R = m.group("L"), m.group("w"), m.group("R")
            if not L and not R:
                return m.group(0)
            # without a source match, only when the gap is unmistakable: the letter must
            # join BOTH neighbours into an attested word
            # (ශ්‍රේෂ් ness ත්වයට from "grandeur" -> ශ්‍රේෂ්ඨත්වයට)
            matched = from_source(w)
            if not matched and not (L and R):
                return m.group(0)
            # a right fragment starting with a vowel sign / virama cannot stand alone and
            # must be absorbed; other non-word fragments are preferred but optional
            # (the left neighbour is often a real but unlisted inflection, e.g. නරඹන්නාගෙන්)
            must_R = bool(R) and COMBINING.match(R) is not None
            want_L = bool(L) and not is_word(L)
            want_R = bool(R) and not is_word(R)
            best = None
            for ch in FILL:
                for use_L in ((True, False) if L else (False,)):
                    for use_R in ((True, False) if R else (False,)):
                        if not (use_L or use_R) or (must_R and not use_R):
                            continue
                        if not matched and not (use_L and use_R):
                            continue
                        cand = (L if use_L else "") + ch + (R if use_R else "")
                        f = W.get(cand, 0)
                        if f >= MIN_FREQ:
                            satisfied = (use_L or not want_L) + (use_R or not want_R)
                            key = (satisfied, use_L + use_R, f)
                            if best is None or key > best[0]:
                                best = (key, cand, use_L, use_R, ch)
            if best is None:
                stats["unresolved"][w] += 1
                return m.group(0)
            _, cand, use_L, use_R, ch = best
            stats["fixed"][ch] += 1
            stats["examples"].setdefault(w, (m.group(0), cand))
            return ((L + " ") if (L and not use_L) else "") + cand + ((" " + R) if (R and not use_R) else "")
        return LATIN_GAP.sub(repl, tgt)
    return fix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", default="vqa_english.json")
    ap.add_argument("--cache", nargs="+", default=["vqa_sinhala.cache.jsonl", "vqa_sinhala.cache.repair.jsonl"],
                    help="sentence caches; later files override earlier ones (repair pass last)")
    ap.add_argument("--out", default="vqa_sinhala.json")
    ap.add_argument("--lexicon", default="/ml/Datasets/UltraChat/data/sinhala_lexicon.pkl")
    ap.add_argument("--vocab", default="/ml/Datasets/UltraChat/tokenizer/unigram_32000_0.9995.vocab")
    args = ap.parse_args()

    W = load_lexicon(None, args.lexicon)
    norm = SinhalaZwjNormalizer(W, load_vocab_zwj(args.vocab), MIN_FREQ)
    fix_rare = make_rare_letter_fixer(W)
    join_stem = make_stem_prefix_joiner(W)
    split = split_sentences_factory()

    cache = {}
    for path in args.cache:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    cache[j["src"]] = j      # later lines / files (repairs) override earlier ones

    stats = {"fixed": collections.Counter(), "unresolved": collections.Counter(), "examples": {},
             "stem_prefix": 0}
    fixed = {}
    for s, j in cache.items():
        t = j["tgt"]
        # re-appending the source's final mark can double it ("... .?"); keep the source's
        m = re.search(r"[.!?]{2,}$", t.rstrip())
        if m and not re.search(r"[.!?]{2,}$", s.rstrip()):
            sm = re.search(r"[.!?]$", s.rstrip())
            t = t.rstrip()[:m.start()] + (sm.group(0) if sm else m.group(0)[-1])
        t = norm.normalize(t)
        t = fix_rare(s, t, stats)
        fixed[s] = join_stem(norm.normalize(t), stats)

    en = json.load(open(args.en, encoding="utf-8"))
    out, missing = [], 0
    for r in en:
        o = {"image_id": r["image_id"]}
        for k in FIELDS:
            paras = []
            for p in split(r[k]):
                if any(s not in fixed for s in p):
                    missing += 1
                paras.append(" ".join(fixed.get(s, "") for s in p))
            o[k] = "\n\n".join(paras)
        out.append(o)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"wrote {len(out)} records to {args.out} ({missing} paragraphs with untranslated sentences)")
    print(f"ZWJ merges: {norm.merges} lexicon + {stats['stem_prefix']} stem-prefix")
    print(f"rare letters restored: {sum(stats['fixed'].values())} {dict(stats['fixed'].most_common())}")
    print(f"latin gaps left: {sum(stats['unresolved'].values())} {stats['unresolved'].most_common(15)}")
    for w, (a, b) in list(stats["examples"].items())[:25]:
        print(f"  {a!r} -> {b!r}")


if __name__ == "__main__":
    main()
