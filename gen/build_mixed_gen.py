"""Build the bilingual UltraChat `gen` training mix for the stage-2 SFT run.

Stage 1 (`sft/`) trained SinLlama_v02 on the Sinhala `sft` split and produced a
model that answers *English* prompts in Sinhala -- only 1.1% of its assistant
turns were majority-Latin, so it learned "reply in Sinhala" unconditionally
(docs/uc-instruct-evaluation.md section 5). This script builds the corpus that
undoes that: the `gen` split, whose prompt_ids are disjoint from the `sft` split
already seen, with a quarter of it swapped to the original English.

The mix
-------
Take the prompt_id intersection of the Sinhala and English files, sort it, and
assign `idx % en_stride == 0` to English and the rest to Sinhala. That gives
exactly 1/en_stride English and puts each prompt_id in exactly one language, so
no dialogue is ever trained on twice in two languages.

The rule keys on prompt_id rather than row index deliberately: only 1 of the
28,300 shared ids sits at the same position in both files, so splitting on file
order would give an overlap of whatever the two orderings happen to share.

Per-dialogue pipeline
---------------------
1. **Drop trailing non-assistant turns.** `gen` dialogues have odd turn counts
   (3/5/7/9/11/13) and 100% of them end on a *user* turn -- that is what makes
   them the generation splits. The final user turn has no reference answer, so
   keeping it would trail every example with unsupervised tokens. Dropping it
   leaves a dialogue structurally identical to an `sft` one.
2. **Clean the Sinhala only.** The Sinhala side carries the JW300
   machine-translation artifacts documented in docs/ultrachat-cleaning.md
   (30.2% of gen turns modified), so it goes through
   `sft.clean_ultrachat.clean_text`. English is the original UltraChat text and
   must not: the RULES are Sinhala regexes that would no-op, but TIDY's
   `[ \\t]{2,} -> " "` collapses indentation and would corrupt English code
   blocks and markdown.
3. **`repair()`** -- truncate at the first empty turn back to the last intact
   assistant turn, drop what cannot be saved. Note this is also why step 1 must
   come first: `repair()` rejects anything not ending assistant-side, which
   before step 1 is every single `gen` dialogue.

Usage
-----
    python gen/build_mixed_gen.py --dry-run --limit 5000   # stats, writes nothing
    python gen/build_mixed_gen.py --split eval             # eval slices only
    python gen/build_mixed_gen.py                          # the real thing

Outputs `train_gen_mixed.parquet`, `eval_gen_si.parquet`, `eval_gen_en.parquet`
and a `manifest.json` recording the assignment rule, per-language counts, drop
reasons and cleaner hit counts, into `mix.out_dir`.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sft"))
from clean_ultrachat import clean_text, repair  # noqa: E402

# The two source files disagree on the list-element field name -- Sinhala uses
# `element`, the HF English parquet uses `item` -- so concatenating them needs
# one explicit schema rather than whatever the first file happened to carry.
OUT_SCHEMA = pa.schema([
    pa.field("prompt", pa.string()),
    pa.field("prompt_id", pa.string()),
    pa.field("messages", pa.list_(pa.struct([
        pa.field("content", pa.string()),
        pa.field("role", pa.string()),
    ]))),
    pa.field("language", pa.string()),
])


def resolve(path: str) -> str:
    """Config paths are relative to the repo root, not the caller's cwd."""
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else REPO_ROOT / p)


def expand(pattern: str | list[str], what: str) -> list[str]:
    """Resolve a config path that may be a glob (English ships as 3 shards).

    A list is a fallback chain, not a union: the first pattern that matches
    anything wins. That lets one config work both here -- where the English
    test_gen is already committed -- and on the pod, where gen/fetch_data.sh
    puts the HF snapshot somewhere else.
    """
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    for p in patterns:
        hits = sorted(glob.glob(resolve(p)))
        if hits:
            return hits
    tried = "\n".join(f"    {resolve(p)}" for p in patterns)
    raise SystemExit(
        f"no files matched {what} = {pattern!r}\n  tried:\n{tried}\n"
        f"  English data is not in the repo -- run `bash gen/fetch_data.sh` first."
    )


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------


def read_ids(files: list[str]) -> list[str]:
    return pq.ParquetDataset(files).read(columns=["prompt_id"])["prompt_id"].to_pylist()


def assign_languages(si_files: list[str], en_files: list[str], stride: int,
                     limit: int | None) -> tuple[dict[str, str], dict[str, Any]]:
    """Map each shared prompt_id to exactly one of 'si' / 'en'."""
    si_ids, en_ids = set(read_ids(si_files)), set(read_ids(en_files))
    shared = sorted(si_ids & en_ids)
    if limit:
        shared = shared[:limit]
    if not shared:
        raise SystemExit("the Sinhala and English files share no prompt_id -- wrong pair?")

    assignment = {pid: ("en" if i % stride == 0 else "si") for i, pid in enumerate(shared)}
    stats = {
        "si_file_ids": len(si_ids),
        "en_file_ids": len(en_ids),
        "shared_ids": len(shared),
        "si_only": len(si_ids - en_ids),
        "en_only": len(en_ids - si_ids),
        "assigned_si": sum(1 for v in assignment.values() if v == "si"),
        "assigned_en": sum(1 for v in assignment.values() if v == "en"),
    }
    return assignment, stats


# --------------------------------------------------------------------------
# Per-dialogue preparation
# --------------------------------------------------------------------------


def prepare(messages: list[dict], clean: bool, hits: Counter) -> tuple[list[dict] | None, str | None]:
    """Return (messages, drop_or_note). messages is None when the row is dropped."""
    # 1. gen dialogues end on a user turn with no reference answer.
    trimmed = list(messages or [])
    while trimmed and trimmed[-1].get("role") != "assistant":
        trimmed.pop()
    if len(trimmed) < 2:
        return None, "no_assistant_turn"

    # 2. Sinhala gets the MT-artifact cleaner; English gets whitespace only, so
    #    that repair()'s empty-turn detection sees the same thing either way.
    for m in trimmed:
        content = m.get("content") or ""
        m["content"] = clean_text(content, hits) if clean else content.strip()

    # 3. Truncate at an empty turn / reject what cannot be saved.
    kept, reason = repair(trimmed)
    if not kept or len(kept) < 2:
        return None, reason or "too_short"
    return kept, reason


def build_rows(files: list[str], assignment: dict[str, str], language: str,
               clean: bool, batch_size: int = 2000) -> Iterator[tuple[pa.Table, Counter, Counter, int]]:
    """Stream one language's parquet, yielding prepared batches as arrow tables.

    Genuinely streaming (`iter_batches`, not `read()`): the Sinhala train_gen
    file is 3.1 GB of text decoded, and this runs alongside the model download.
    """
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=batch_size, columns=["prompt", "prompt_id", "messages"]):
            hits: Counter = Counter()
            drops: Counter = Counter()
            n_trunc = 0
            out = []
            for row in batch.to_pylist():
                if assignment.get(row["prompt_id"]) != language:
                    continue
                kept, reason = prepare(row["messages"], clean, hits)
                if kept is None:
                    drops[reason] += 1
                    continue
                if reason:
                    n_trunc += 1
                prompt = row.get("prompt") or ""
                out.append({
                    "prompt": clean_text(prompt) if clean else prompt.strip(),
                    "prompt_id": row["prompt_id"],
                    "messages": kept,
                    "language": language,
                })
            yield (pa.Table.from_pylist(out, schema=OUT_SCHEMA) if out
                   else OUT_SCHEMA.empty_table()), hits, drops, n_trunc


def collect(files: list[str], assignment: dict[str, str], language: str,
            clean: bool) -> tuple[pa.Table, dict[str, Any]]:
    tables, hits, drops = [], Counter(), Counter()
    n_trunc = 0
    for table, h, d, t in build_rows(files, assignment, language, clean):
        if table.num_rows:
            tables.append(table)
        hits += h
        drops += d
        n_trunc += t
    combined = pa.concat_tables(tables) if tables else OUT_SCHEMA.empty_table()
    return combined, {
        "kept": combined.num_rows,
        "truncated_at_empty_turn": n_trunc,
        "dropped": dict(drops),
        "cleaner_hits": dict(hits),
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def shuffled(table: pa.Table, seed: int) -> pa.Table:
    """Interleave the two languages.

    Trainer's sampler shuffles anyway, so this is defensive -- it just means a
    run that stops mid-epoch has still seen both languages. Peak memory is
    roughly 2x the table during the take(); pass --no-shuffle to skip it.
    """
    order = list(range(table.num_rows))
    random.Random(seed).shuffle(order)
    return table.take(order)


def write(table: pa.Table, path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def summarize(name: str, table: pa.Table, stats: dict[str, Any]) -> None:
    print(f"\n  {name}: {stats['kept']:,} dialogues kept")
    if stats["truncated_at_empty_turn"]:
        print(f"    truncated at an empty turn : {stats['truncated_at_empty_turn']:,}")
    if stats["dropped"]:
        print("    dropped                    : "
              + ", ".join(f"{k}={v:,}" for k, v in sorted(stats["dropped"].items())))
    if stats["cleaner_hits"]:
        top = sorted(stats["cleaner_hits"].items(), key=lambda kv: -kv[1])[:5]
        print("    cleaner hits               : "
              + ", ".join(f"{k}={v:,}" for k, v in top))
    else:
        print("    cleaner hits               : none (expected for English)")
    if table.num_rows:
        turns = [len(m) for m in table["messages"].to_pylist()[:5000]]
        bad = sum(1 for m in table["messages"].to_pylist()[:5000] if m[-1]["role"] != "assistant")
        print(f"    turns/dialogue             : mean {sum(turns) / len(turns):.1f}, "
              f"max {max(turns)}; not assistant-last: {bad}")


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "gen" / "config.yaml"))
    ap.add_argument("--split", choices=["train", "eval", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N shared prompt_ids (prototyping)")
    ap.add_argument("--en-stride", type=int, default=None, help="override mix.en_stride")
    ap.add_argument("--seed", type=int, default=None, help="override mix.seed")
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))["mix"]
    stride = args.en_stride or cfg.get("en_stride", 4)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    clean_si = cfg.get("clean_sinhala", True)
    out_dir = Path(resolve(cfg["out_dir"]))
    manifest: dict[str, Any] = {
        "assignment_rule": f"sorted(shared prompt_ids)[i] -> 'en' if i % {stride} == 0 else 'si'",
        "en_stride": stride,
        "seed": seed,
        "clean_sinhala": clean_si,
        "limit": args.limit,
        "splits": {},
    }

    def do(split: str, si_key: str, en_key: str) -> tuple[pa.Table, pa.Table]:
        si_files = expand(cfg[si_key], f"mix.{si_key}")
        en_files = expand(cfg[en_key], f"mix.{en_key}")
        print(f"\n{'=' * 70}\n{split}\n{'=' * 70}")
        print(f"  sinhala: {', '.join(Path(f).name for f in si_files)}")
        print(f"  english: {', '.join(Path(f).name for f in en_files)}")

        assignment, ids = assign_languages(si_files, en_files, stride, args.limit)
        print(f"\n  prompt_ids: {ids['si_file_ids']:,} si, {ids['en_file_ids']:,} en, "
              f"{ids['shared_ids']:,} shared "
              f"({ids['si_only']:,} si-only, {ids['en_only']:,} en-only dropped)")
        print(f"  assigned  : {ids['assigned_si']:,} si / {ids['assigned_en']:,} en "
              f"({100 * ids['assigned_en'] / ids['shared_ids']:.1f}% english)")

        si_tbl, si_stats = collect(si_files, assignment, "si", clean_si)
        en_tbl, en_stats = collect(en_files, assignment, "en", clean=False)
        summarize("sinhala", si_tbl, si_stats)
        summarize("english", en_tbl, en_stats)

        si_pids = set(si_tbl["prompt_id"].to_pylist())
        en_pids = set(en_tbl["prompt_id"].to_pylist())
        overlap = si_pids & en_pids
        if overlap:
            raise SystemExit(f"BUG: {len(overlap)} prompt_ids landed in both languages")
        print(f"\n  cross-language prompt_id overlap: 0 of {len(si_pids) + len(en_pids):,} ✓")

        manifest["splits"][split] = {"ids": ids, "si": si_stats, "en": en_stats}
        return si_tbl, en_tbl

    train_pids: set[str] = set()
    eval_pids: set[str] = set()

    if args.split in ("train", "both"):
        si_tbl, en_tbl = do("train", "si_train", "en_train")
        table = pa.concat_tables([si_tbl, en_tbl])
        train_pids = set(table["prompt_id"].to_pylist())
        if not args.no_shuffle:
            table = shuffled(table, seed)
        dest = out_dir / "train_gen_mixed.parquet"
        write(table, dest, args.dry_run)
        manifest["splits"]["train"]["output"] = {"path": str(dest), "rows": table.num_rows}
        print(f"\n  -> {dest if not args.dry_run else '(dry run)'}: {table.num_rows:,} dialogues")

    if args.split in ("eval", "both"):
        si_tbl, en_tbl = do("eval", "si_eval", "en_eval")
        eval_pids = set(si_tbl["prompt_id"].to_pylist()) | set(en_tbl["prompt_id"].to_pylist())
        rng = random.Random(seed)
        for lang, tbl, n_key in (("si", si_tbl, "eval_si"), ("en", en_tbl, "eval_en")):
            n = min(cfg.get(n_key, 0) or tbl.num_rows, tbl.num_rows)
            idx = rng.sample(range(tbl.num_rows), n)
            sliced = tbl.take(sorted(idx))
            dest = out_dir / f"eval_gen_{lang}.parquet"
            write(sliced, dest, args.dry_run)
            manifest["splits"]["eval"].setdefault("output", {})[lang] = {
                "path": str(dest), "rows": sliced.num_rows}
            print(f"  -> {dest if not args.dry_run else '(dry run)'}: {sliced.num_rows:,} dialogues")

    # train_gen and test_gen take their membership verbatim from the upstream
    # ultrachat_200k split, so this should be empty -- but it is the one check
    # that would catch a mis-set si_eval/en_eval pointing back at the train file.
    if train_pids and eval_pids:
        leak = train_pids & eval_pids
        if leak:
            raise SystemExit(f"LEAK: {len(leak):,} prompt_ids appear in both train and eval")
        manifest["train_eval_overlap"] = 0
        print(f"\ntrain/eval prompt_id overlap: 0 ✓")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"\nmanifest: {out_dir / 'manifest.json'}")
        print("Next: python sft/run_sft_uc.py --config gen/config.yaml --preview 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
