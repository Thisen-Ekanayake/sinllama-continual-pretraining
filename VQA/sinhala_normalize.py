"""Stage 9 post-decode — restore the Zero-Width Joiner NLLB strips from Sinhala.

**Root cause (measured, not assumed).** NLLB-200's SentencePiece *normalizer*
rewrites the Sinhala conjunct joiner ``U+0DCA U+200D <C>`` to
``U+0DCA U+0020 <C>`` at *encode* time::

    tok.backend_tokenizer.normalizer.normalize_str("්‍ර")  ->  "් ර"  (0DCA 0020 0DBB)
    "ප්‍රධාන"  --encode->decode-->  "ප් රධාන"   (U+200D count 1 -> 0)

So the ZWJ is gone before the model ever sees it: the model never learns to emit
``U+200D`` and it **cannot be recovered from the generated token ids**. A real
word boundary (``ග්ලූටන් රහිත``, gluten-free) round-trips unchanged, which is why
the only signal of corruption is a space *inside* a conjunct. The fix therefore
has to be **orthographic restoration on the decoded string**; this module does
that and ``translate.py`` applies it to every decoded segment, so the written
data is correct at the source (no separate post-hoc repair pass needed).

**Method.** ``hal + space + letter`` is ambiguous (corruption vs. real boundary),
so a space is turned back into ``U+200D`` only when the ZWJ-joined form is a real
Sinhala word, decided against a lexicon with three confidence tiers:

  T1 corpus     ZWJ-join occurs >= threshold times in a clean Sinhala corpus
  T2 vocab      ZWJ-join is a SentencePiece ZWJ piece (secondary oracle)
  T3 structural lone consonant+hal + ``ර``/``ය`` whose 4-char cluster is attested
                (recovers loanwords/inflections absent from the corpus)

Consecutive joints are merged maximally so multi-conjunct words (``ද්‍රව්‍ය``,
``බ්‍රිතාන්‍ය``) are rebuilt. Only a single ``U+0020`` is ever swapped for
``U+200D``; nothing else in the text is touched.

This is the canonical implementation of the algorithm. The batch JSONL repair
tool ``tools/fix_sinhala_zwj.py`` imports the same core, and the guarantees
(surgical-diff invariant, idempotency, decision logic) are machine-checked by
``tools/test_fix_sinhala_zwj.py``. Full write-up: ``docs/sinhala_zwj_repair.md``.
"""
from __future__ import annotations

import io
import os
import pickle
import re
from collections import Counter

ZWJ = "‍"  # U+200D
# A "word" = maximal run of Sinhala letters/marks + ZWJ (conjunct words stay whole).
WORD = re.compile(r"[඀-෿‍]+")
LETTER = re.compile(r"[ක-ෆ]")
# A repair chain: a hal-ending run, then >=1 space-separated runs (all but last end in hal).
CHAIN = re.compile(r"[඀-෿‍]*්(?: [඀-෿‍]*්)* [඀-෿]+")
# Attested conjunct cluster, used to gate Tier-3.
CLUSTER = re.compile(r"[ක-ෆ]්‍[රය]")


# --------------------------------------------------------------------------- #
# Core algorithm (pure, stdlib-only; validated by tools/test_fix_sinhala_zwj.py)
# --------------------------------------------------------------------------- #
def build_lexicon(corpus_path, min_count=2):
    """Word-frequency dict from a clean ZWJ-preserving Sinhala corpus."""
    cnt = Counter()
    with io.open(corpus_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            for w in WORD.findall(line):
                if LETTER.search(w):
                    cnt[w] += 1
    return {w: c for w, c in cnt.items() if c >= min_count}


def load_lexicon(corpus_path, cache_path, min_count=2):
    """Load the lexicon from a pickle cache, building (and caching) it if absent."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    W = build_lexicon(corpus_path, min_count)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(W, f, protocol=pickle.HIGHEST_PROTOCOL)
    return W


def load_vocab_zwj(vocab_path):
    """Set of ZWJ-bearing SentencePiece pieces (▁ stripped) from a .vocab file."""
    s = set()
    if vocab_path and os.path.exists(vocab_path):
        for line in io.open(vocab_path, encoding="utf-8"):
            p = line.split("\t")[0].lstrip("▁")
            if ZWJ in p:
                s.add(p)
    return s


def is_lone_hal(tok):
    """True if ``tok`` is a single consonant + hal (never a standalone word)."""
    s = tok.replace(ZWJ, "")
    return len(s) == 2 and s[1] == "්" and "ක" <= s[0] <= "ෆ"


def valid_conjunct_clusters(W):
    """Attested ``C ් ZWJ ර|ය`` clusters; gates Tier-3 so it cannot invent
    non-clusters such as ය්‍ර. Derived from real ZWJ words in the lexicon."""
    cl = set()
    for w in W:
        if ZWJ in w:
            cl.update(CLUSTER.findall(w))
    return cl


def new_stats():
    return {"merge_sites": 0, "kept_sites": 0, "tier": Counter(),
            "merged_forms": {}, "t3_audit": Counter(), "residual": Counter()}


def repair_text(text, W, vocab_zwj, threshold, stats, valid_clusters=None):
    """Restore ZWJ in ``text``; record decisions in ``stats``. Returns new text."""
    valid_clusters = valid_clusters if valid_clusters is not None else valid_conjunct_clusters(W)

    def ok(cand):
        if W.get(cand, 0) >= threshold:
            return "T1"
        if cand in vocab_zwj:
            return "T2"
        return None

    def fix_chain(m):
        toks = m.group(0).split(" ")
        out, i = [], 0
        while i < len(toks):
            best = None
            for j in range(len(toks), i + 1, -1):           # longest join first
                cand = ZWJ.join(toks[i:j])
                tier = ok(cand)
                if tier:
                    best = (j, cand, tier); break
            if best is None and i + 1 < len(toks) and is_lone_hal(toks[i]) \
                    and toks[i + 1][:1] in ("ර", "ය") \
                    and (toks[i] + ZWJ + toks[i + 1][0]) in valid_clusters:
                best = (i + 2, ZWJ.join(toks[i:i + 2]), "T3")  # lone-hal + attested cluster
            if best:
                j, cand, tier = best
                stats["merge_sites"] += (j - i - 1)
                stats["tier"][tier] += (j - i - 1)
                stats["merged_forms"][cand] = (W.get(cand, 0), tier)
                if tier == "T3":
                    stats["t3_audit"][cand] += 1
                out.append(cand); i = j
            else:
                if i + 1 < len(toks) and toks[i].endswith("්"):
                    stats["kept_sites"] += 1
                    if is_lone_hal(toks[i]):
                        stats["residual"][(toks[i], toks[i + 1])] += 1
                out.append(toks[i]); i += 1
        return " ".join(out)

    return CHAIN.sub(fix_chain, text)


# --------------------------------------------------------------------------- #
# Pipeline-facing normalizer
# --------------------------------------------------------------------------- #
class SinhalaZwjNormalizer:
    """Stateful normalizer for the translation hot loop.

    Holds the lexicon, vocab oracle and attested clusters once; ``normalize()``
    is then a cheap regex pass per decoded segment, accumulating job-level
    statistics in ``self.stats``.
    """

    def __init__(self, lexicon, vocab_zwj=frozenset(), threshold=5):
        self.W = lexicon
        self.vocab = set(vocab_zwj)
        self.threshold = threshold
        self.clusters = valid_conjunct_clusters(lexicon)
        self.stats = new_stats()

    def normalize(self, text: str) -> str:
        return repair_text(text, self.W, self.vocab, self.threshold,
                           self.stats, self.clusters)

    @property
    def merges(self) -> int:
        return self.stats["merge_sites"]

    @classmethod
    def from_config(cls) -> "SinhalaZwjNormalizer":
        """Build/load from the pipeline config (corpus + tokenizer vocab)."""
        from pipeline import config
        cache = str(config.ZWJ_LEXICON_CACHE)
        corpus = str(config.ZWJ_CORPUS)
        if not os.path.exists(cache) and not os.path.exists(corpus):
            raise FileNotFoundError(
                f"ZWJ lexicon needs either a cache ({cache}) or the corpus "
                f"({corpus}); set UC_ZWJ_CORPUS / UC_ZWJ_LEXICON_CACHE, or "
                f"disable with UC_ZWJ_FIX=0.")
        W = load_lexicon(corpus, cache)
        vocab = load_vocab_zwj(str(config.ZWJ_VOCAB))
        return cls(W, vocab, config.ZWJ_MIN_FREQ)


# --------------------------------------------------------------------------- #
# CLI: build (cache lexicon) | demo (normalize a string) | selftest
# --------------------------------------------------------------------------- #
_SELFTEST = [  # (input, expected) — mirrors the unit cases in the test suite
    ("ප් රධාන", "ප්‍රධාන"),
    ("ද් රව් ය", "ද්‍රව්‍ය"),
    ("ග්ලූටන් රහිත", "ග්ලූටන් රහිත"),     # real compound: kept
    ("එක් රටක්", "එක් රටක්"),             # two words: kept
    ("ක් පමණ", "ක් පමණ"),               # stray ක් + function word: kept
    ("ට් රෑම්", "ට්‍රෑම්"),               # T3 ra loanword
    ("ක් වන", "ක් වන"),                 # lone-hal + ව excluded from T3: kept
]


def main() -> int:
    import argparse
    from pipeline import config
    from pipeline.common import StepTimer, get_logger
    log = get_logger("zwj_normalize")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="build & cache the Sinhala lexicon, then exit")
    pd = sub.add_parser("demo", help="normalize --text (or stdin) and show the diff")
    pd.add_argument("--text", default=None)
    sub.add_parser("selftest", help="run the built-in decision-logic checks")
    args = ap.parse_args()

    if args.cmd == "selftest":
        # Synthetic lexicon attesting exactly the clusters the cases need.
        W = {w: 1000 for w in ["ප්‍රධාන", "ද්‍රව්‍ය", "ට්‍රක්", "එක්", "රටක්",
                               "ග්ලූටන්", "රහිත", "පමණ", "වන"]}
        n = SinhalaZwjNormalizer(W)
        bad = 0
        for src, exp in _SELFTEST:
            got = n.normalize(src)
            ok = got == exp
            bad += not ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {src!r} -> {got!r}"
                  + ("" if ok else f"  (expected {exp!r})"))
        print(f"{len(_SELFTEST)-bad}/{len(_SELFTEST)} passed")
        return 1 if bad else 0

    with StepTimer(log, "load Sinhala ZWJ lexicon"):
        norm = SinhalaZwjNormalizer.from_config()
    log.info("lexicon=%d types | vocab ZWJ pieces=%d | attested ර/ය clusters=%d",
             len(norm.W), len(norm.vocab), len(norm.clusters))

    if args.cmd == "build":
        log.info("cache ready at %s", config.ZWJ_LEXICON_CACHE)
        return 0

    # demo
    import sys
    text = args.text if args.text is not None else sys.stdin.read()
    out = norm.normalize(text)
    cps = lambda s: " ".join("%04X" % ord(c) for c in s)
    print("in :", text)
    print("out:", out)
    print(f"ZWJ {text.count(ZWJ)} -> {out.count(ZWJ)} | merges this call: {norm.merges}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
