"""Build the textbook CPT corpus: Sinhala school textbooks (grades 6-11) as plain
text, one document per chapter (lesson), for raw causal-LM LoRA training with no
chat template (textbook_cpt/run_textbook_cpt.py).

Same source and cleaning as textbook_sft/build_textbook_sft_dataset.py
(iter_sections -> clean_block: garbage, footers, figure furniture, math
normalisation, cross-book duplicate blocks), but nothing is turned into
instructions: a chapter is written back as Markdown,

    # <chapter>

    ## <section>

    <block>

    ...

Exercise / activity sections are kept: as running text they are part of the
book. Chapters shorter than MIN_DOC chars are dropped (title pages, stubs).

Eval: chapters held out by sha1 bucket on subject/grade/chapter (eval_percent),
so no lesson is on both sides; held-out chapters whose 8-gram overlap with train
is >= 50% (the same passage in two grades' books) are dropped from eval.

Usage
-----
    python textbook_cpt/build_textbook_cpt_dataset.py --dry-run
    python textbook_cpt/build_textbook_cpt_dataset.py

Output (data.out_dir): train.parquet, eval.parquet, manifest.json.
Schema: id, subject, grade, chapter, text.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "textbook_sft"))
from build_textbook_sft_dataset import (  # noqa: E402
    bucket, iter_sections, load_books, load_tokenizer, resolve, word_ngrams,
)

MIN_DOC = 300  # chars

DOC_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("subject", pa.string()),
    pa.field("grade", pa.int32()),
    pa.field("chapter", pa.string()),
    pa.field("text", pa.string()),
])


def book_docs(path: Path, book: dict, seen_blocks: set, drops: Counter) -> list[dict]:
    """One document per chapter; the untitled front matter before the first
    `# ` chapter is its own document."""
    docs: list[dict] = []
    for sec in iter_sections(path.read_text(encoding="utf-8"), seen_blocks, drops):
        if not docs or docs[-1]["chapter"] != sec["chapter"]:
            docs.append({"id": f"{book['id']}:{len(docs)}", "subject": book["subject"],
                         "grade": book["grade"], "chapter": sec["chapter"], "parts": []})
            if sec["chapter"]:
                docs[-1]["parts"].append(f"# {sec['chapter']}")
        if sec["heading"]:
            docs[-1]["parts"].append(f"## {sec['heading']}")
        docs[-1]["parts"] += sec["blocks"]
    out = []
    for d in docs:
        d["text"] = "\n\n".join(d.pop("parts")).strip()
        if len(d["text"]) < MIN_DOC:
            drops["doc_too_short"] += 1
            continue
        out.append(d)
    return out


def write(rows: list[dict], path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=DOC_SCHEMA), path, compression="zstd")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "textbook_cpt" / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--tokenizer", default=None, help="tokenizer for the token counts")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))["textbook_cpt"]
    seed = cfg.get("seed", 42)
    md_root = resolve(cfg["markdown_dir"])
    out_dir = resolve(cfg["out_dir"])
    arts = {k.split("/")[0] for k in json.loads((md_root / "_report.json").read_text())}

    drops: Counter = Counter()
    seen_blocks: set[str] = set()
    docs: list[dict] = []
    for path, book in load_books(md_root, arts):
        docs += book_docs(path, book, seen_blocks, drops)

    is_eval = lambda d: bucket(f"{d['subject']}/{d['grade']}/{d['chapter']}:{seed}", 100) < cfg["eval_percent"]
    train = [d for d in docs if not is_eval(d)]
    held = [d for d in docs if is_eval(d)]
    train_grams = set().union(*(word_ngrams(d["text"]) for d in train))
    eval_ = [d for d in held if not ((g := word_ngrams(d["text"])) and len(g & train_grams) / len(g) >= 0.5)]
    drops["eval_overlaps_train"] = len(held) - len(eval_)

    tok = load_tokenizer({}, args.tokenizer)
    count = lambda ds: sum(len(tok(d["text"], add_special_tokens=False).input_ids) + 2 for d in ds)  # + BOS/EOS
    stats = {
        "train_docs": len(train), "eval_docs": len(eval_),
        "train_chars": sum(len(d["text"]) for d in train), "eval_chars": sum(len(d["text"]) for d in eval_),
        "train_tokens": count(train), "eval_tokens": count(eval_),
    }
    print(f"textbook docs: {len(docs):,}; dropped " + ", ".join(f"{k}={v:,}" for k, v in drops.most_common()))
    for k, v in stats.items():
        print(f"  {k:<13} {v:>12,}")
    print("  train by subject: " + ", ".join(f"{s}={n}" for s, n in
                                              Counter(d["subject"] for d in train).most_common()))

    write(train, out_dir / "train.parquet", args.dry_run)
    write(eval_, out_dir / "eval.parquet", args.dry_run)
    if not args.dry_run:
        manifest = {"seed": seed, "eval_percent": cfg["eval_percent"], "min_doc": MIN_DOC,
                    "dropped": dict(drops), **stats,
                    "eval_ids": [d["id"] for d in eval_]}
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"\n-> {out_dir}: train.parquet, eval.parquet, manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
