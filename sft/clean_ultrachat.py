"""Strip machine-translation artifacts from UltraChat_Sinhala before SFT.

The corpus was machine-translated by a system contaminated with the JW300 /
Watchtower parallel corpus, and it leaks that corpus's boilerplate into
unrelated text. The damage is mechanical and positional, so most of it comes out
with a handful of anchored substitutions. See docs/ultrachat-cleaning.md for the
evidence and docs/uc-instruct-evaluation.md for what the uncleaned data did to
the trained model.

    python sft/clean_ultrachat.py --dry-run        # measure, write nothing
    python sft/clean_ultrachat.py                  # -> *_clean.parquet
    python sft/clean_ultrachat.py --in a.parquet --out b.parquet

Writes to a NEW filename rather than editing in place, deliberately: the
tokenized-dataset cache in build_uc_dataset.py is keyed on
`{filename_stem}_{template_fingerprint}_len{max_seq}`, so an in-place edit would
silently reuse the stale cache built from the dirty text.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------
# Rules. Order matters -- see ORDERING below. Each is
# (name, compiled_pattern, replacement, why).
# --------------------------------------------------------------------------

# Junk words the translator wedges between a list marker and the real content.
INJECTED_WORDS = [
    "සෞඛ්‍යය",     # "health"  -- 22,743 hits
    "මෘදු",         # "soft"    --  9,065
    "සීමාව",        # "limit"   --  2,984
    "සයිටම්",       #           --  1,704
    "මයික්",        # "Mike"    --    329
    "මිනීමැරුම්",   # "murder"  --    115
]

# Numerals the translator turned into Sinhala sentences ("it is five").
WORD_NUMERALS = {
    "පහයි": "5",
    "හයයි": "6",
}

ORDERING = """\
1. word-numerals first: a line reading "පහයි. සෞඛ්‍යය foo" only matches the
   injected-word rule's "^\\d+\\." anchor after it has become "5. සෞඛ්‍යය foo".
2. injected-word stripping second, now that every list item starts with a digit.
3. bracket-boilerplate removal third -- position-independent and mutually
   disjoint, so it cannot disturb the line-anchored rules above.
4. whitespace tidy last, to clean up after the removals -- boilerplate spans
   cluster in runs of 2+, so removal routinely leaves doubled spaces.

Deliberately NOT included, having been measured and rejected:
  * Unicode normalization. The corpus is already 100% NFC (0 of 41,719 sampled
    turns change under normalize("NFC")).
  * Any ZWJ rule. 100% of 535,574 U+200D occurrences are correctly preceded by
    U+0DCA, and inserting ZWJ into bare ්ර / ්ය would corrupt කාර්ය, පර්යේෂණ,
    සූර්ය, දුම්රිය and thousands of other correctly spelled words.
  * Legacy visual-order vowel signs (627 occ), exploded rakaransaya clusters
    (191 occ), fraction slashes (98 occ). All real, all under 0.4% of turns,
    and all requiring codepoint transposition -- not worth the risk here.
  * "රූප සටහන" (338 turns) is legitimate and means "diagram"; empty "[]" (973)
    is Java/Python code. Both are false friends -- do not add rules for them.
"""


def _build_rules() -> list[tuple[str, re.Pattern, str, str]]:
    rules: list[tuple[str, re.Pattern, str, str]] = []

    for word, digit in WORD_NUMERALS.items():
        rules.append((
            f"word_numeral_{digit}",
            re.compile(rf"(?m)^(\s*){word}\.(\s)"),
            rf"\g<1>{digit}.\g<2>",
            f'list number {digit} was translated as the sentence "{word}." '
            f"-- anchored to line start so genuine mid-sentence prose is untouched",
        ))

    rules.append((
        "injected_list_word",
        re.compile(rf"(?m)^(\s*\d+\.\s+)(?:{'|'.join(map(re.escape, INJECTED_WORDS))})\s+"),
        r"\1",
        "a junk word is wedged between the list marker and the content in "
        "99.2% of numbered list items",
    ))

    rules.append((
        "jw300_caption",
        # Superset of the plain "[Nවන පිටුවේ පින්තූරය]" form: also catches
        # "[සංස්කරණය කරන ලද පිටුවේ පින්තූරය]" and "[පිටුව Nවන පිටුවේ පින්තූරය]",
        # +690 spans per 76k turns over the strict form, and tolerates stray
        # whitespace inside the brackets.
        re.compile(r"\s*\[\s*(?:පිටුව\s*)?(?:\d+\s*වන|සංස්කරණය\s+කරන(?:\s+ලද)?)"
                   r"\s*පිටුවේ\s*පින්තූරය\s*\]\s*"),
        " ",
        '"[Picture on page N]" -- JW300/Watchtower furniture, in 33% of '
        "assistant turns, never relevant to the conversation",
    ))

    rules.append((
        "mediawiki_edit_link",
        re.compile(r"\s*\[සංස්කරණය(?:\s+කරන්න|\s+කරන\s+ලද\s+පිටපත)?\]\s*"),
        " ",
        '"[edit]" -- a MediaWiki section-edit link, injected free-floating into '
        "recipes, travel guides and fiction where it can carry no meaning; "
        "bracket-anchored, so the 1,404 running-word uses of සංස්කරණය "
        '("edition") in the same sample are provably untouched',
    ))

    return rules


RULES = _build_rules()

# Applied after RULES, to repair spacing the removals leave behind.
TIDY = [
    ("tidy_spaces", re.compile(r"[ \t]{2,}"), " "),
    ("tidy_space_before_punct", re.compile(r" +([,.!?;:])"), r"\1"),
    ("tidy_blank_lines", re.compile(r"\n{3,}"), "\n\n"),
    ("tidy_trailing_ws", re.compile(r"(?m)[ \t]+$"), ""),
]


def clean_text(text: str, hits: Counter | None = None) -> str:
    """Apply every rule in order. Counts substitutions into `hits` if given."""
    if not text:
        return text
    for name, pat, repl, _why in RULES:
        text, n = pat.subn(repl, text)
        if n and hits is not None:
            hits[name] += n
    for name, pat, repl in TIDY:
        text, n = pat.subn(repl, text)
        if n and hits is not None:
            hits[name] += n
    return text.strip()


# --------------------------------------------------------------------------
# Dialogue-level repair: damage a substitution cannot fix.
#
# Some turns are nothing but caption boilerplate -- '[25වන පිටුවේ පින්තූරය]' and
# no other text -- so cleaning leaves them empty. Rather than discard the whole
# conversation, truncate to the longest prefix that still ends on an assistant
# turn; 54% of affected dialogues survive that way. Drop only what is left.
# --------------------------------------------------------------------------

def repair(messages: list[dict]) -> tuple[list[dict], str | None]:
    """Return (kept_messages, drop_reason). drop_reason is None when kept."""
    if not messages:
        return [], "empty"

    empty = next((i for i, m in enumerate(messages)
                  if not (m.get("content") or "").strip()), None)
    if empty is not None:
        # Dialogues alternate user-first, so an assistant turn is at an odd
        # index; the longest valid prefix ends at the last odd index below
        # `empty`.
        keep = empty if empty % 2 == 0 else empty - 1
        if keep < 2:
            return [], "no_content_before_empty_turn"
        messages = messages[:keep]

    if messages[-1].get("role") != "assistant":
        return [], "not_assistant_last"
    return messages, ("truncated_at_empty_turn" if empty is not None else None)


# --------------------------------------------------------------------------

def process(src: Path, dst: Path, dry_run: bool, batch_size: int = 2000):
    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    writer = None
    hits: Counter = Counter()
    drops: Counter = Counter()
    n_rows = n_kept = n_turns = n_changed = n_trunc = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        rows = batch.to_pylist()
        out = []
        for row in rows:
            n_rows += 1
            msgs = row.get("messages") or []
            for m in msgs:
                n_turns += 1
                before = m.get("content") or ""
                after = clean_text(before, hits)
                if after != before:
                    n_changed += 1
                m["content"] = after

            msgs, reason = repair(msgs)
            if not msgs:
                drops[reason] += 1
                continue
            if reason:
                n_trunc += 1
            row["messages"] = msgs
            if row.get("prompt"):
                row["prompt"] = clean_text(row["prompt"])
            n_kept += 1
            out.append(row)

        if not dry_run and out:
            table = pa.Table.from_pylist(out, schema=schema)
            if writer is None:
                dst.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(dst, schema)
            writer.write_table(table)

    if writer is not None:
        writer.close()

    return {"rows": n_rows, "kept": n_kept, "turns": n_turns,
            "turns_changed": n_changed, "truncated": n_trunc,
            "hits": hits, "drops": drops}


def report(src: Path, dst: Path, stats: dict, dry_run: bool):
    print(f"\n{'=' * 70}\n{src.name} -> {dst.name if not dry_run else '(dry run, nothing written)'}\n{'=' * 70}")
    print(f"dialogues   : {stats['rows']:>9,} in   {stats['kept']:>9,} kept"
          f"   ({stats['rows'] - stats['kept']:,} dropped)")
    print(f"turns       : {stats['turns']:>9,}      {stats['turns_changed']:>9,} modified"
          f"   ({100 * stats['turns_changed'] / max(stats['turns'], 1):.1f}%)")
    print(f"truncated   : {stats['truncated']:>9,} dialogues cut back to their last "
          f"intact assistant turn")
    if stats["drops"]:
        print("dropped     : " + ", ".join(f"{k}={v:,}" for k, v in stats["drops"].most_common()))
    print("\nsubstitutions:")
    width = max((len(n) for n in stats["hits"]), default=10)
    for name, n in stats["hits"].most_common():
        print(f"  {name:<{width}}  {n:>9,}")
    if not stats["hits"]:
        print("  (none)")


def residual_check(path: Path, limit_batches: int = 4):
    """Re-scan a cleaned file for artifacts that should now be gone."""
    item = re.compile(r"(?m)^\s*\d+\.\s+")
    inj = re.compile("|".join(map(re.escape, INJECTED_WORDS)))
    wordnum = re.compile(r"(?m)^\s*(?:" + "|".join(WORD_NUMERALS) + r")\.")
    caption = re.compile(r"\[\s*\d+\s*වන\s*පිටුවේ")
    n_items = n_bad = n_wordnum = n_caption = 0
    for i, batch in enumerate(pq.ParquetFile(path).iter_batches(batch_size=4000)):
        if i >= limit_batches:
            break
        for row in batch.to_pylist():
            for m in row["messages"]:
                t = m["content"]
                for mt in item.finditer(t):
                    n_items += 1
                    n_bad += bool(inj.match(t, mt.end()))
                n_wordnum += len(wordnum.findall(t))
                n_caption += len(caption.findall(t))
    print(f"\nresidual scan of {path.name} (first {limit_batches * 4000:,} dialogues):")
    print(f"  numbered list items          : {n_items:,}")
    print(f"  ...still carrying a junk word: {n_bad:,} "
          f"({100 * n_bad / max(n_items, 1):.2f}%)")
    print(f"  line-start word numerals     : {n_wordnum:,}")
    print(f"  page-picture captions        : {n_caption:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", nargs="*",
                    default=["UltraChat_Sinhala/train_sft.parquet",
                             "UltraChat_Sinhala/test_sft.parquet"])
    ap.add_argument("--out", dest="dst", nargs="*", default=None,
                    help="defaults to each input with a _clean suffix")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rules", action="store_true", help="print the rule table and exit")
    args = ap.parse_args()

    if args.rules:
        print(ORDERING)
        for i, (name, pat, repl, why) in enumerate(RULES, 1):
            print(f"{i}. {name}\n   why : {why}\n   pat : {pat.pattern}\n   sub : {repl!r}\n")
        return 0

    srcs = [Path(p) for p in args.src]
    if args.dst:
        dsts = [Path(p) for p in args.dst]
        if len(dsts) != len(srcs):
            print("--out must have the same number of paths as --in", file=sys.stderr)
            return 2
    else:
        dsts = [p.with_name(f"{p.stem}_clean{p.suffix}") for p in srcs]

    for src, dst in zip(srcs, dsts):
        if not src.exists():
            print(f"missing input: {src}", file=sys.stderr)
            return 2
        stats = process(src, dst, args.dry_run)
        report(src, dst, stats, args.dry_run)
        if not args.dry_run:
            residual_check(dst)

    if not args.dry_run:
        print("\nNext: point data.train_file / data.eval_file in sft/config.yaml at the "
              "_clean files and re-run sft/run_sft_uc.sh into a fresh output_dir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
