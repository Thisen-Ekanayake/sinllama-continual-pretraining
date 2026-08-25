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

# Wiki link markup, keeping the anchor's visible text and dropping the tag.
# fetch_wikipedia_dump.sh no longer passes --links, so freshly extracted text
# has none of this -- but an `extracted/` tree produced by an older run (or by
# Wikipedia_Dataset/wikiextractor/extract.sh) still does, and it is what put
# `&lt;a href="%E0%B7%81..."&gt;` into 67.1% of the first SinLlama_wiki run's
# training rows. Both spellings are handled: --html-safe (on by default)
# escapes the tags, a run with it off leaves them literal.
# The opening tag is "anything up to the closing bracket" rather than href
# alone: external links carry extra attributes (rel="mw:ExtLink" title="...").
_ANCHOR_ESC = re.compile(r"&lt;a\s(?:(?!&gt;).)*?&gt;(.*?)&lt;/a&gt;", re.S)
_ANCHOR_RAW = re.compile(r"<a\s[^>]*>(.*?)</a>", re.S)
# An article truncated mid-link leaves an unpaired tag the rules above cannot
# match; drop those outright so no markup survives into a training example.
_ANCHOR_ORPHAN = re.compile(
    r"&lt;/?a(?:\s(?:(?!&gt;).)*?)?(?:&gt;|$)|</?a(?:\s[^>]*)?>", re.S)


def resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def normalize_text(text: str) -> str:
    """Light cleanup of WikiExtractor's plain-text output.

    Strips wiki link markup down to its visible text (see _ANCHOR_ESC), then
    tidies whitespace. Template stripping (e.g. a name's untranslated English
    form) sometimes leaves an empty "()", which also goes. Nothing else is
    rewritten.
    """
    text = _ANCHOR_ESC.sub(r"\1", text)
    text = _ANCHOR_RAW.sub(r"\1", text)
    text = _ANCHOR_ORPHAN.sub("", text)
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


def build_row(article: dict[str, Any]) -> dict[str, Any] | None:
    title = (article.get("title") or "").strip()
    text = normalize_text(article.get("text") or "")
    if not title or not text:
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
    print(f"kept {len(rows):,} / {len(raw):,} articles ({dropped} dropped: missing title or text)")
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

    train_path = out_dir / "train_wiki.parquet"
    eval_path = out_dir / "eval_wiki.parquet"
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
