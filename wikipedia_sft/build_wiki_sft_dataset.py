"""Wrap Sinhala Wikipedia articles into the UltraChat SFT chat template.

Stage 3: continues models/SinLlama_uc_gen_bilingual (the "uc_gen" model) on
encyclopedic Sinhala knowledge, in the same instruction format stages 1-2
already trained -- see sft/prompt_template.txt. Wikipedia articles are not
dialogues, so each one becomes a single-turn synthetic conversation:

    ### User:
    {a Sinhala question asking about the article's title}

    ### Assistant:
    {the article body, verbatim}<|end_of_text|>

The question is drawn from a fixed pool of Sinhala phrasings (PROMPTS below),
chosen deterministically per article id (sha1(id) % len(PROMPTS)) rather than
one fixed phrasing for every row -- otherwise the model would learn "any
statement of fact starts with this exact sentence" instead of the underlying
world knowledge. Deterministic, not random-per-run: rebuilding the dataset
gives every article the same question again.

Source data is wikipedia_sft/fetch_wikipedia_dump.sh's short_articles.json /
medium_articles.json (150-2000 word articles; stubs carry too little signal
and long articles mostly overflow max_seq_length and would truncate an
assistant turn mid-sentence -- see run_sft_uc.py's truncation docstring).

Usage
-----
    python wikipedia_sft/build_wiki_sft_dataset.py --dry-run   # stats only
    python wikipedia_sft/build_wiki_sft_dataset.py             # writes train/eval parquet + manifest.json

Output schema (consumed by sft/build_uc_dataset.py, which only reads
`messages`; the rest is provenance for debugging):
    id, title, category, prompt, messages: [{role, content}]
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Deliberately varied so the model does not learn one fixed instruction
# phrasing as a prerequisite for reciting facts. All are plain, grammatical
# Sinhala requests for information about {title}.
PROMPTS = [
    "{title} ගැන කියන්න.",
    "{title} යනු කුමක්ද?",
    "{title} ගැන විස්තර කරන්න.",
    "{title} පිළිබඳව ඔබ දන්නා දේ කියන්න.",
    "{title} ගැන කෙටි විස්තරයක් දෙන්න.",
    "{title} ගැන තොරතුරු ලබා දෙන්න.",
    "{title} පිළිබඳ විස්තරයක් ලබා දෙන්න.",
    "{title} යනු මොකක්ද කියා පැහැදිලි කරන්න.",
]

OUT_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("title", pa.string()),
    pa.field("category", pa.string()),
    pa.field("prompt", pa.string()),
    pa.field("messages", pa.list_(pa.struct([
        pa.field("content", pa.string()),
        pa.field("role", pa.string()),
    ]))),
])

_BLANK_PARENS = re.compile(r"\(\s*\)")
_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")

# --------------------------------------------------------------------------
# Extraction artifacts.
#
# fetch_wikipedia_dump.sh no longer passes --links, so freshly extracted text
# carries none of the anchor markup -- but an `extracted/` tree from an older
# run (or from Wikipedia_Dataset/wikiextractor/extract.sh) does, and that is
# what put `&lt;a href="%E0%B7%81..."&gt;` into 67.1% of the first
# SinLlama_wiki run's training rows. Everything below is therefore written to
# clean an already-extracted tree as well as a fresh one.
#
# Tags are stripped BY NAME, never by a generic `<\w+>` rule. An audit of the
# corpus found `fml`, `act`, `hdmv`, `vhml`, `person`, `snoj` and friends in
# articles *about* markup languages, where the tags are the subject matter and
# must survive. Only tags that are genuinely MediaWiki/HTML furniture are
# listed. Paired forms keep their inner text; void/orphan forms are dropped.
# --------------------------------------------------------------------------
_STRIP_TAGS = (
    "a", "b", "i", "u", "s", "br", "hr", "p", "em", "strong", "span", "div",
    "sup", "sub", "small", "big", "code", "pre", "tt", "font", "center",
    "blockquote", "poem", "nowiki", "math", "chem", "ul", "ol", "li", "dl",
    "dt", "dd", "table", "tr", "td", "th", "caption", "ref", "references",
    "gallery", "score", "mapframe", "templatestyles", "link", "onlyinclude",
    "includeonly", "noinclude", "timeline", "imagemap", "syntaxhighlight",
)
_TAG_ALT = "|".join(_STRIP_TAGS)
# Escaped (WikiExtractor's --html-safe default) and raw spellings of each.
_TAG_PAIRED_ESC = re.compile(
    rf"&lt;({_TAG_ALT})\b(?:(?!&gt;).)*?&gt;(.*?)&lt;/\1&gt;", re.S | re.I)
_TAG_PAIRED_RAW = re.compile(
    rf"<({_TAG_ALT})\b[^>]*>(.*?)</\1>", re.S | re.I)
# Unpaired: void tags, and tags left dangling by an article truncated mid-link.
_TAG_ORPHAN_ESC = re.compile(
    rf"&lt;/?(?:{_TAG_ALT})\b(?:(?!&gt;).)*?(?:&gt;|$)", re.S | re.I)
_TAG_ORPHAN_RAW = re.compile(rf"</?(?:{_TAG_ALT})\b[^>]*>", re.S | re.I)

# Wiki markup that survives extraction in a small tail of articles.
_WIKILINK_PIPED = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]")   # [[Target|shown]] -> shown
_WIKILINK_PLAIN = re.compile(r"\[\[([^\]|]*)\]\]")           # [[shown]]        -> shown
# Innermost {{...}} only; applied repeatedly so nested infobox templates
# ({{#if:|{{legend0|{{IIJ/P/TC }}...}}}}) unwind from the inside out. A single
# non-nested pass leaves the outer braces and their pipe-delimited parameter
# lines behind, which is what survived the first cleaning attempt in ~6 country
# articles.
_TEMPLATE_INNER = re.compile(r"\{\{[^{}]*\}\}")
# Whatever the unwind cannot balance: stray braces, and the parameter lines of
# a half-expanded infobox (" |capital | =", "| Belfast", "{| class=...").
_BRACE_ORPHAN = re.compile(r"\{\{|\}\}|\{\||\|\}")
_TABLE_LINE = re.compile(r"^\s*(?:\||!|\{\||\|\}).*$", re.M)
_WIKI_QUOTES = re.compile(r"'{2,5}")                         # ''italic''/'''bold'''
# Footnote/reference leftovers: "[1]", "[12]" — never legitimate Sinhala prose.
_FOOTNOTE = re.compile(r"\[\d{1,3}\]")
# Bare external URLs, overwhelmingly in "මූලාශ්‍ර." (sources) sections. Dropped
# rather than kept: an SFT target that recites youtube.com/watch?v=... teaches
# the model to emit URLs, which is exactly what this stage should not do.
# `\S*` not `\S+`, and a tolerated space after the scheme: the corpus contains
# "http:// www.google.com", which `\S+` cannot match at all.
_BARE_URL = re.compile(r"https?:\s*//\s*\S*")
# Percent-encoded Sinhala left behind by a link whose tag was already stripped.
_URL_ENCODED = re.compile(r"(?:%[0-9A-Fa-f]{2}){3,}[-\w%]*")
# A leaked table-of-contents header at the very start of an article body.
_TOC_HEAD = re.compile(r"\A\s*අන්තර්ගතය\s*\.\s*\n")


def resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def normalize_text(text: str) -> str:
    """Strip extraction artifacts from WikiExtractor's plain-text output.

    Order matters. Tags are removed while still escaped, because unescaping
    first would turn a literal `&lt;fml&gt;` code sample in an article *about*
    a markup language into something a tag rule could eat. Only the tag names
    in _STRIP_TAGS are touched, so those samples survive either way.

    Measured on the 11,087-article short+medium corpus (2026-08-25): anchors in
    67.0% of articles, URL-encoded runs in 62.6%, bare URLs in 6.0%, empty
    parens 3.6%, footnote markers 1.5%, HTML entities 1.6%, and a long tail of
    wiki markup, tables and refs under 1% each.
    """
    # 1. Markup tags, escaped form first, then any raw survivors.
    for _ in range(2):  # nested pairs, e.g. an <a> inside a <ref>
        text = _TAG_PAIRED_ESC.sub(r"\2", text)
        text = _TAG_PAIRED_RAW.sub(r"\2", text)
    text = _TAG_ORPHAN_ESC.sub("", text)
    text = _TAG_ORPHAN_RAW.sub("", text)

    # 2. Entities. After tag removal, what is left is content: &amp; -> &, and
    #    a code sample's &lt;fml&gt; back to its intended <fml>. Looped because
    #    parts of the corpus are double-escaped ("S&amp;amp;P"), which one pass
    #    leaves as a visible "&amp;".
    for _ in range(3):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    # Unescaping can expose a tag that was double-escaped in the source.
    text = _TAG_PAIRED_RAW.sub(r"\2", text)
    text = _TAG_ORPHAN_RAW.sub("", text)

    # 3. Wiki markup that survived extraction.
    text = _WIKILINK_PIPED.sub(r"\1", text)
    text = _WIKILINK_PLAIN.sub(r"\1", text)
    for _ in range(6):  # unwind nested templates from the inside out
        collapsed = _TEMPLATE_INNER.sub("", text)
        if collapsed == text:
            break
        text = collapsed
    text = _TABLE_LINE.sub("", text)
    # Removing an unbalanced "{{" can leave its parameter line ("|name=...")
    # starting the line, which the pass above has already gone by -- so sweep
    # table lines once more afterwards.
    text = _BRACE_ORPHAN.sub("", text)
    text = _TABLE_LINE.sub("", text)
    text = _WIKI_QUOTES.sub("", text)

    # 4. Reference furniture and bare links.
    text = _FOOTNOTE.sub("", text)
    text = _BARE_URL.sub("", text)
    text = _URL_ENCODED.sub("", text)
    text = _TOC_HEAD.sub("", text)

    # 5. Whitespace and the empty "()" left by template stripping.
    text = _BLANK_PARENS.sub("", text)
    text = _TRAILING_WS.sub("\n", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def prompt_for(article_id: str) -> str:
    idx = int(hashlib.sha1(article_id.encode("utf-8")).hexdigest(), 16) % len(PROMPTS)
    return PROMPTS[idx]


def load_articles(paths: list[Path]) -> list[dict[str, Any]]:
    articles: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            raise SystemExit(
                f"missing source file: {path}\n"
                f"  run `bash wikipedia_sft/fetch_wikipedia_dump.sh` first."
            )
        for a in json.loads(path.read_text(encoding="utf-8")):
            articles[a["id"]] = a  # de-dupe defensively; ids are unique per category
    return list(articles.values())


# Articles this short after cleaning were mostly link lists or infobox shells:
# the prose that remains is a single sentence, which is a poor SFT target for a
# "tell me about X" prompt. Measured at 200: 2 of 11,087 articles drop out.
MIN_CHARS = 200


def build_row(article: dict[str, Any]) -> dict[str, Any] | None:
    title = (article.get("title") or "").strip()
    text = normalize_text(article.get("text") or "")
    if not title or not text:
        return None
    if len(text) < MIN_CHARS:
        return None
    prompt = prompt_for(article["id"])
    return {
        "id": article["id"],
        "title": title,
        "category": article.get("category", ""),
        "prompt": prompt,
        "messages": [
            {"role": "user", "content": prompt.format(title=title)},
            {"role": "assistant", "content": text},
        ],
    }


def write(table: pa.Table, path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "wikipedia_sft" / "config.yaml"))
    ap.add_argument("--limit", type=int, default=None, help="use only the first N articles (prototyping)")
    ap.add_argument("--seed", type=int, default=None, help="override wiki_source.seed")
    ap.add_argument("--eval-articles", type=int, default=None, help="override wiki_source.eval_articles")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))["wiki_source"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    eval_n = args.eval_articles if args.eval_articles is not None else cfg.get("eval_articles", 500)
    out_dir = resolve(cfg["out_dir"])
    source_paths = [resolve(p) for p in cfg["articles"]]

    print(f"reading: {', '.join(str(p) for p in source_paths)}")
    raw = load_articles(source_paths)
    raw.sort(key=lambda a: a["id"])  # stable order regardless of source file iteration
    if args.limit:
        raw = raw[: args.limit]

    rows, dropped = [], 0
    for article in raw:
        row = build_row(article)
        if row is None:
            dropped += 1
            continue
        rows.append(row)

    by_category: dict[str, int] = {}
    for r in rows:
        by_category[r["category"]] = by_category.get(r["category"], 0) + 1
    print(f"kept {len(rows):,} / {len(raw):,} articles "
          f"({dropped} dropped: no title, or under {MIN_CHARS} chars once cleaned)")
    for cat, n in sorted(by_category.items(), key=lambda kv: -kv[1]):
        print(f"  {cat}: {n:,}")

    if eval_n >= len(rows):
        raise SystemExit(f"wiki_source.eval_articles ({eval_n}) >= available articles ({len(rows)})")

    rng = random.Random(seed)
    eval_idx = set(rng.sample(range(len(rows)), eval_n))
    train_rows = [r for i, r in enumerate(rows) if i not in eval_idx]
    eval_rows = [r for i, r in enumerate(rows) if i in eval_idx]
    rng.shuffle(train_rows)  # Trainer shuffles anyway; defensive, as in gen/build_mixed_gen.py

    train_table = pa.Table.from_pylist(train_rows, schema=OUT_SCHEMA)
    eval_table = pa.Table.from_pylist(eval_rows, schema=OUT_SCHEMA)

    # `_clean` in the name, not an in-place overwrite of train_wiki.parquet.
    # sft/build_uc_dataset.py keys its tokenized-dataset cache on
    # `{filename_stem}_{template_fingerprint}_len{max_seq}`, so reusing the old
    # filename after changing the text would silently reload the stale tokens
    # built from the markup-laden corpus and train on them again. Same reason
    # sft/clean_ultrachat.py writes *_clean.parquet.
    train_path = out_dir / "train_wiki_clean.parquet"
    eval_path = out_dir / "eval_wiki_clean.parquet"
    write(train_table, train_path, args.dry_run)
    write(eval_table, eval_path, args.dry_run)
    print(f"\n-> {train_path if not args.dry_run else '(dry run)'}: {train_table.num_rows:,} rows")
    print(f"-> {eval_path if not args.dry_run else '(dry run)'}: {eval_table.num_rows:,} rows")

    if not args.dry_run:
        manifest = {
            "source_files": [str(p) for p in source_paths],
            "seed": seed,
            "eval_articles": eval_n,
            "prompts": PROMPTS,
            "kept": len(rows),
            "dropped": dropped,
            "by_category": by_category,
            "train_rows": train_table.num_rows,
            "eval_rows": eval_table.num_rows,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"\nmanifest: {out_dir / 'manifest.json'}")
        print("Next: python sft/run_sft_uc.py --config wikipedia_sft/config.yaml --preview 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
