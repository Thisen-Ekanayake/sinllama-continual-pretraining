"""Build the v2 stage-3 corpus: Wikipedia rows whose answer length matches the
question, mixed ~50/50 with Sinhala UltraChat replay.

Why a v2
--------
build_wiki_sft_dataset.py (the corpus behind SinLlama_wiki_clean) answers every
"tell me about {title}" phrasing -- including "give a SHORT description" -- with
the entire cleaned article (median ~2,300 chars). On the 100-prompt Gemini-judged
probe (inference/judge_comparison_UC_Gen_vs_wiki_clean.txt) wiki_clean lost to
its own parent UC_Gen on every criterion, overall 3.61 vs 5.11, conciseness 3.45
vs 5.81: it learned "any prompt -> a full article". This builder keeps the same
articles and cleaner but changes what the answers look like.

Wikipedia rows (one per article, kind chosen deterministically from sha1(id))
-------------------------------------------------------------------------------
    lead    50%  "what is X / tell me about X"  -> the lead section (paragraphs
                 before the first section heading), capped at LEAD_CAP chars
                 on a paragraph boundary
    short   25%  "briefly / in a few sentences" -> the first 2-3 sentences
    detail  25%  "describe X in detail"         -> the full article, capped at
                 DETAIL_CAP chars on a paragraph boundary
An article too short to have a distinct detail answer falls back to `lead`.

Article quality filter (on top of the old cleaner's MIN_CHARS)
--------------------------------------------------------------
    not_sinhala     < 60% of letters are Sinhala script (English articles)
    low_prose       < 60% of characters sit in prose paragraphs (>= 80 chars
                    with a sentence end); catches lists, tables, link farms
    short_lead      lead section < 150 chars
    run_on          median sentence > 350 chars (tangled, unreadable text)

UltraChat replay
----------------
Sinhala UltraChat `gen` dialogues -- the data UC_Gen was trained on -- through
gen/build_mixed_gen.py's prepare() (drop the trailing unanswered user turn,
clean MT artifacts, repair). Sampled to `ultrachat_ratio` x the number of
Wikipedia train rows. A separate Sinhala test_gen slice is written as a second
eval set, so the trainer reports eval_wiki_loss and eval_uc_loss.

Usage
-----
    python wikipedia_sft/build_wiki_mix_dataset.py --config wikipedia_sft/config_v2.yaml --dry-run
    python wikipedia_sft/build_wiki_mix_dataset.py --config wikipedia_sft/config_v2.yaml

Output (wiki_mix.out_dir): train_mix.parquet, eval_wiki.parquet, eval_uc.parquet,
manifest.json. Schema: id, source, kind, title, messages: [{role, content}]
(the trainer reads only `messages`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "wikipedia_sft"))
sys.path.insert(0, str(REPO_ROOT / "gen"))
from build_wiki_sft_dataset import MIN_CHARS, load_articles, normalize_text, resolve  # noqa: E402
from build_mixed_gen import prepare  # noqa: E402

PROMPTS = {
    "lead": [
        "{title} ගැන කියන්න.",
        "{title} යනු කුමක්ද?",
        "{title} පිළිබඳව ඔබ දන්නා දේ කියන්න.",
        "{title} ගැන තොරතුරු ලබා දෙන්න.",
        "{title} යනු මොකක්ද කියා පැහැදිලි කරන්න.",
    ],
    "short": [
        "{title} ගැන කෙටි විස්තරයක් දෙන්න.",
        "{title} ගැන කෙටියෙන් කියන්න.",
        "{title} ගැන වාක්‍ය කිහිපයකින් පැහැදිලි කරන්න.",
        "{title} යනු කුමක්දැයි කෙටියෙන් පැහැදිලි කරන්න.",
    ],
    "detail": [
        "{title} ගැන විස්තර කරන්න.",
        "{title} පිළිබඳ සවිස්තරාත්මක විස්තරයක් ලබා දෙන්න.",
        "{title} ගැන විස්තරාත්මකව ලියන්න.",
        "{title} පිළිබඳ විස්තරයක් ලබා දෙන්න.",
    ],
}
# sha1(id) % 4 -> 0,1 lead / 2 short / 3 detail  == 50 / 25 / 25
KIND_BY_BUCKET = ["lead", "lead", "short", "detail"]

LEAD_CAP = 1500
DETAIL_CAP = 2000
SHORT_SENTENCES = (2, 3)
SHORT_MAX = 600

SINHALA = re.compile(r"[඀-෿]")
LATIN = re.compile(r"[A-Za-z]")
# A section heading after normalize_text: one short line, a single trailing
# full stop, no sentence inside it ("පිහිටිම.", "උද්‍යානයේ භෞතික ලක්ෂණ.").
HEADING = re.compile(r"^[^.?!]{1,60}\.$")
# Sentence end: . ? ! followed by whitespace. A full stop right after a 1-2
# character token is an abbreviation or initial (ක්‍රි.ව., ඩී. එස්.), not an end.
SENT_END = re.compile(r"(?<=[.?!])\s+")
ABBREV_TAIL = re.compile(r"(?:^|[\s.(])\S{1,2}\.$")
SENTENCE_LINE = re.compile(r"[.?!][\"'”’)]?$")
EDIT_LINK = re.compile(r"\s*\[\s*සංස්කරණය\s*\]")  # "[ edit ]" section links

ROW_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("source", pa.string()),
    pa.field("kind", pa.string()),
    pa.field("title", pa.string()),
    pa.field("messages", pa.list_(pa.struct([
        pa.field("content", pa.string()),
        pa.field("role", pa.string()),
    ]))),
])


def bucket(key: str, n: int) -> int:
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % n


def paragraphs(text: str) -> list[str]:
    """Prose paragraphs and section headings; everything else is dropped.

    A line that neither is a heading nor ends like a sentence is an image
    caption, a repeated title or an image-alignment leftover ("right", "thumb")
    that the old cleaner lets through -- not something an answer should say.
    """
    out = []
    for p in EDIT_LINK.sub("", text).split("\n"):
        p = p.strip()
        if p and (HEADING.match(p) or SENTENCE_LINE.search(p)):
            out.append(p)
    return out


def sentences(par: str) -> list[str]:
    out, buf = [], ""
    for piece in SENT_END.split(par):
        buf = f"{buf} {piece}".strip() if buf else piece
        if re.search(r"[.?!]$", buf) and not ABBREV_TAIL.search(buf):
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def cap_paragraphs(pars: list[str], cap: int) -> str:
    """Whole paragraphs up to `cap` chars; always at least the first one."""
    kept, total = [], 0
    for p in pars:
        if kept and total + len(p) > cap:
            break
        kept.append(p)
        total += len(p) + 1
    return "\n".join(kept)


def split_article(text: str) -> tuple[list[str], list[str]]:
    """(lead paragraphs, all paragraphs). Headings stay in the full list."""
    pars = paragraphs(text)
    lead = []
    for p in pars:
        if HEADING.match(p):
            if lead:
                break
            continue  # a heading before any prose is a stray title line
        lead.append(p)
    return lead, pars


def quality(text: str, lead: list[str], pars: list[str]) -> str | None:
    """None if the article is usable, else the drop reason."""
    si, la = len(SINHALA.findall(text)), len(LATIN.findall(text))
    if si + la == 0 or si / (si + la) < 0.6:
        return "not_sinhala"
    prose = sum(len(p) for p in pars if len(p) >= 80 and re.search(r"[.?!]", p))
    # Against the whole cleaned text, so captions and list lines that
    # paragraphs() already threw away still count as non-prose.
    if prose / max(1, len(text)) < 0.6:
        return "low_prose"
    if sum(len(p) for p in lead) < 150:
        return "short_lead"
    sents = [s for p in pars if not HEADING.match(p) for s in sentences(p)]
    if sents and statistics.median(len(s) for s in sents) > 350:
        return "run_on"
    return None


def short_answer(lead: list[str], key: str) -> str | None:
    sents = sentences(" ".join(lead))
    want = SHORT_SENTENCES[bucket(key + ":n", len(SHORT_SENTENCES))]
    out = ""
    for s in sents[:want]:
        if out and len(out) + len(s) > SHORT_MAX:
            break
        out = f"{out} {s}".strip()
    return out if len(out) >= 60 else None


def build_wiki_row(article: dict[str, Any], drops: Counter) -> dict[str, Any] | None:
    title = (article.get("title") or "").strip()
    text = normalize_text(article.get("text") or "")
    if not title or len(text) < MIN_CHARS:
        drops["too_short"] += 1
        return None
    lead, pars = split_article(text)
    reason = quality(text, lead, pars)
    if reason:
        drops[reason] += 1
        return None

    aid = article["id"]
    kind = KIND_BY_BUCKET[bucket(aid, len(KIND_BY_BUCKET))]
    lead_text = cap_paragraphs(lead, LEAD_CAP)
    answer = None
    if kind == "short":
        answer = short_answer(lead, aid)
    elif kind == "detail":
        full = cap_paragraphs(pars, DETAIL_CAP)
        # Not worth a "detail" row if it says nothing the lead does not.
        answer = full if len(full) >= len(lead_text) + 300 else None
    if answer is None:
        kind, answer = "lead", lead_text

    prompts = PROMPTS[kind]
    prompt = prompts[bucket(aid + ":p", len(prompts))].format(title=title)
    return {
        "id": aid, "source": "wiki", "kind": kind, "title": title,
        "messages": [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": answer}],
    }


def ultrachat_rows(path: str, n: int, seed: int, drops: Counter) -> list[dict[str, Any]]:
    """n cleaned Sinhala dialogues sampled uniformly from `path`."""
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    # Oversample ids so dropped dialogues do not leave us short.
    want = set(random.Random(seed).sample(range(total), min(total, int(n * 1.15) + 50)))
    rows, offset, hits = [], 0, Counter()
    for batch in pf.iter_batches(batch_size=4000, columns=["prompt_id", "messages"]):
        for i, row in enumerate(batch.to_pylist()):
            if offset + i not in want:
                continue
            kept, reason = prepare(row["messages"], True, hits)
            if kept is None:
                drops[reason] += 1
                continue
            rows.append({"id": row["prompt_id"], "source": "ultrachat", "kind": "dialogue",
                         "title": "", "messages": kept})
        offset += batch.num_rows
    random.Random(seed + 1).shuffle(rows)
    return rows[:n]


def write(rows: list[dict[str, Any]], path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=ROW_SCHEMA), path, compression="zstd")


def describe(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    kinds = Counter(r["kind"] for r in rows)
    lens: dict[str, list[int]] = {}
    for r in rows:
        lens.setdefault(r["kind"], []).append(len(r["messages"][-1]["content"]))
    stats = {k: {"rows": kinds[k], "answer_chars_median": int(statistics.median(v)),
                 "answer_chars_p90": sorted(v)[int(0.9 * len(v))]} for k, v in lens.items()}
    print(f"  {name}: {len(rows):,} rows")
    for k, s in sorted(stats.items()):
        print(f"    {k:8s} {s['rows']:6,}  answer chars median {s['answer_chars_median']:5,}  "
              f"p90 {s['answer_chars_p90']:5,}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "wikipedia_sft" / "config_v2.yaml"))
    ap.add_argument("--limit", type=int, default=None, help="use only the first N articles (prototyping)")
    ap.add_argument("--show", type=int, default=0, help="print N example wiki rows per kind")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))["wiki_mix"]
    seed = cfg.get("seed", 42)
    out_dir = resolve(cfg["out_dir"])

    raw = load_articles([resolve(p) for p in cfg["articles"]])
    raw.sort(key=lambda a: a["id"])
    if args.limit:
        raw = raw[: args.limit]

    wiki_drops: Counter = Counter()
    rows = [r for a in raw if (r := build_wiki_row(a, wiki_drops)) is not None]
    print(f"wikipedia: kept {len(rows):,} / {len(raw):,} articles; dropped "
          + ", ".join(f"{k}={v:,}" for k, v in wiki_drops.most_common()))

    rng = random.Random(seed)
    eval_idx = set(rng.sample(range(len(rows)), cfg["eval_articles"]))
    wiki_train = [r for i, r in enumerate(rows) if i not in eval_idx]
    wiki_eval = [r for i, r in enumerate(rows) if i in eval_idx]

    uc_drops: Counter = Counter()
    n_uc = round(len(wiki_train) * cfg["ultrachat_ratio"])
    uc_train = ultrachat_rows(str(resolve(cfg["ultrachat_train"])), n_uc, seed, uc_drops)
    uc_eval = ultrachat_rows(str(resolve(cfg["ultrachat_eval"])), cfg["ultrachat_eval_rows"], seed, uc_drops)
    print(f"ultrachat: {len(uc_train):,} train / {len(uc_eval):,} eval dialogues; dropped "
          + (", ".join(f"{k}={v:,}" for k, v in uc_drops.most_common()) or "none"))

    train = wiki_train + uc_train
    random.Random(seed + 2).shuffle(train)

    print()
    manifest: dict[str, Any] = {
        "seed": seed, "lead_cap": LEAD_CAP, "detail_cap": DETAIL_CAP, "short_max": SHORT_MAX,
        "prompts": PROMPTS, "wiki_dropped": dict(wiki_drops), "ultrachat_dropped": dict(uc_drops),
        "wiki_train": describe(wiki_train, "wiki train"),
        "wiki_eval": describe(wiki_eval, "wiki eval"),
        "ultrachat_train_rows": len(uc_train), "ultrachat_eval_rows": len(uc_eval),
        "train_rows": len(train),
    }
    print(f"  ultrachat train: {len(uc_train):,} dialogues")
    print(f"  train total: {len(train):,} ({100 * len(uc_train) / len(train):.1f}% ultrachat)")

    for kind in ("lead", "short", "detail"):
        for r in [r for r in wiki_train if r["kind"] == kind][: args.show]:
            print(f"\n--- {kind}: {r['messages'][0]['content']}\n{r['messages'][1]['content'][:700]}")

    write(train, out_dir / "train_mix.parquet", args.dry_run)
    write(wiki_eval, out_dir / "eval_wiki.parquet", args.dry_run)
    write(uc_eval, out_dir / "eval_uc.parquet", args.dry_run)
    if not args.dry_run:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"\n-> {out_dir}: train_mix.parquet, eval_wiki.parquet, eval_uc.parquet, manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
