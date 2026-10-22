"""Build the textbook SFT corpus: Sinhala school textbooks (grades 6-11) as
single-turn instruction rows, plus Sinhala UltraChat replay matched in tokens.

Same recipe as wikipedia_sft/build_wiki_mix_dataset.py (v2): every source unit
becomes one `messages: [user, assistant]` row whose prompt is drawn
deterministically (sha1) from a varied Sinhala pool, answers are length-matched
to the question, and UltraChat replay keeps instruction-following in training.

Replay: the cleaned UltraChat SFT split (sft/clean_ultrachat.py output,
UltraChat_Sinhala_cleaned/), already artifact-cleaned, so it is only trimmed and
repaired here. Every textbook row is used; UltraChat dialogues are sampled until
their templated token count reaches `ultrachat_token_ratio` x the textbook train
tokens (1.0 = equal tokens). Rows whose prompt also occurs in the replay eval set
are excluded from replay train.
The chat template itself is applied by sft/run_sft_uc.py (see config.yaml).

Source: textbooks/markdown/<subject>/grade_<N>[_part_<K>].md, produced by
textbooks/extract_arts.py and textbooks/extract_technical.py. A book is a list
of blocks; `# ` lines start a chapter (lesson), `## ` lines start a section.

Row kinds
---------
    detail          "explain {topic}"            -> the section, capped at DETAIL_CAP
    short           "briefly, {topic}"           -> its first 2-3 sentences
    list            "what are {topic}?"          -> a list-shaped section
    definition      "what is X?"                 -> the book's "X යනු ... වේ." sentence
    worked_example  a maths/science නිදසුන        -> its worked solution
    verse           "explain this poem: <verse>" -> the book's අර්ථය (meaning) gloss
    summary         "summarise lesson {chapter}" -> the chapter's සාරාංශය section
Exercise / activity / assignment sections are dropped: they carry questions
without answers.

Cleaning (numbers from textbooks/analyze_dataset.ipynb)
-------------------------------------------------------
    garbage blocks  (cid:NN) dumps, U+FFFD, private-use chars (14.7k chars)
    footers         leftover "නොමිලේ බෙදා හැරීම සඳහා ය" notices (892)
    furniture       figure placeholders, captions, dotted answer lines,
                    fragments of <= 3 words, stray-letter diagram labels
    math            rule-extracted ^{..} / _{..} wrapped as $..$ LaTeX;
                    unbalanced $ stripped
    duplicates      exact duplicate blocks (keep first); identical answers

Usage
-----
    python textbook_sft/build_textbook_sft_dataset.py --dry-run --show 2
    python textbook_sft/build_textbook_sft_dataset.py

Output (textbook_mix.out_dir): train_mix.parquet, eval_textbook.parquet,
eval_uc.parquet, manifest.json. Schema: id, source, kind, title, subject, grade,
messages: [{role, content}] (the trainer reads only `messages`).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "wikipedia_sft"))
sys.path.insert(0, str(REPO_ROOT / "gen"))
sys.path.insert(0, str(REPO_ROOT / "sft"))
from build_wiki_mix_dataset import bucket, describe, sentences  # noqa: E402
from build_wiki_sft_dataset import resolve  # noqa: E402
from build_mixed_gen import prepare  # noqa: E402
from build_uc_dataset import ChatTemplate  # noqa: E402

# ------------------------------------------------------------------- prompts

PROMPTS = {
    "detail": [
        "{topic} පැහැදිලි කරන්න.",
        "{topic} ගැන විස්තර කරන්න.",
        "{topic} පිළිබඳ විස්තරයක් ලබා දෙන්න.",
        "{topic} ගැන ඔබ දන්නා දේ කියන්න.",
        "{topic} පිළිබඳව පැහැදිලි කිරීමක් කරන්න.",
    ],
    "detail_ctx": [  # generic heading: name the lesson it belongs to
        "{chapter} පාඩමේ “{topic}” කොටස පැහැදිලි කරන්න.",
        "{chapter} සම්බන්ධයෙන් {topic} ගැන විස්තර කරන්න.",
        "{chapter} යටතේ {topic} පිළිබඳ කියන්න.",
    ],
    "short": [
        "{topic} කෙටියෙන් පැහැදිලි කරන්න.",
        "{topic} ගැන කෙටි විස්තරයක් දෙන්න.",
        "{topic} ගැන වාක්‍ය කිහිපයකින් කියන්න.",
    ],
    "list": [
        "{topic} මොනවාද?",
        "{topic} ලැයිස්තුගත කරන්න.",
        "{topic} සඳහන් කරන්න.",
        "{topic} මොනවාදැයි පෙන්වා දෙන්න.",
    ],
    "definition": [
        "{term} යනු කුමක්ද?",
        "{term} යන්න හඳුන්වන්න.",
        "{term} යන්නෙන් අදහස් වන්නේ කුමක්ද?",
        "{term} අර්ථ දක්වන්න.",
    ],
    "worked_example": [
        "{question}",
        "පහත ගැටලුව විසඳන්න.\n\n{question}",
        "මෙය පියවරෙන් පියවර විසඳන්න.\n\n{question}",
    ],
    "verse": [
        "පහත කවියේ අර්ථය පැහැදිලි කරන්න.\n\n{verse}",
        "මෙම පද්‍යයේ තේරුම කියන්න.\n\n{verse}",
        "පහත කවිය සරල සිංහලෙන් පැහැදිලි කරන්න.\n\n{verse}",
    ],
    "summary": [
        "{chapter} පාඩමේ සාරාංශය දෙන්න.",
        "{chapter} පිළිබඳ කෙටි සාරාංශයක් ලියන්න.",
        "{chapter} පාඩමේ ප්‍රධාන කරුණු සාරාංශ කරන්න.",
    ],
}
# sha1(section) % 4 -> detail 50% / short 25% / detail 25% (list-shaped sections use `list`)
KIND_BY_BUCKET = ["detail", "detail", "short", "detail"]

DETAIL_CAP = 2000     # chars, on a block boundary (as wiki v2)
SHORT_MAX = 600
MIN_ANSWER = 150      # detail / list / summary / example answers
MIN_SHORT = 60

# ------------------------------------------------------------------ sections

SKIP_SECTION = re.compile(
    r"ක්‍රියාකාරකම|ක්‍රියාකරකම|පැවරුම|අභ්‍යාස|පුහුණුවට|ප්‍රශ්න|පුනරීක්ෂණ|අවබෝධය|"
    r"අරමුණු|මෙම පාඩම අධ්‍යයනය|පාරිභාෂික|උත්තරය|ඔබ ඉගෙන|ඉගෙනුම් ඵල|^PB$")
SUMMARY = re.compile(r"සාරාංශ")
EXAMPLE = re.compile(r"^නිදසුන\s*\d*\s*[:ඃ]?$")
# Headings that name a role, not a topic: the prompt has to name the lesson too.
GENERIC = re.compile(
    r"^(හැඳින්වීම|හැඳින්වීමක්|නිදසුන්|ජීවිතයට බිඳක්|අමතර දැනුමට|පියවර|ප්‍රයෝජන|"
    r"වැදගත්කම|ලක්ෂණ|උදාහරණ|සටහන|විශේෂ සටහන|මතක තබා ගන්න|විස්තරය|පසුබිම|"
    r"ක්‍රමය|ක්‍රම|අදහස|ආකාරය|කරුණු|නිගමනය|විවරණය|භාවිතය|වර්ග|කාර්යය|ව්‍යුහය)$")

# ------------------------------------------------------------------ cleaning

GARBAGE = re.compile(r"\(cid:\d+\)|�|[-]")
FOOTER = re.compile(r"නොමිල[ේයෙ]*\s*බෙදා\s*හැරීම[^.\n]*\.?(\s*\d{1,3}\b)?")
DOTS = re.compile(r"\.{6,}|…{3,}|_{6,}")
FIGURE = re.compile(r"^\[රූපය[^\]]*\]$")
SINHALA_CHAR = re.compile(r"[඀-෿]")
STRAY = re.compile(r"[අ-ෆ]")
LIST_LINE = re.compile(r"^(\d+|[a-zA-Z]|[ivx]+)[.)] |^\((\d+|[ivx]+|[a-z])\) |^[-*•] ")
SCRIPT_TOKEN = re.compile(r"[^\s\u0d80-\u0dff$]*[\^_]\{[^}\s]*\}[^\s\u0d80-\u0dff$]*")
MATH_ONLY_LINE = re.compile(r"^[\s\dA-Za-z().,+\-–−×÷=<>≤≥≠±√π°%/:;\[\]^_{}|]+$")
PARTICLES = set("දයමවහ")
# running head / foot: "64 විද්‍යාව | අම්ල හා භස්ම", "අම්ල හා භස්ම | විද්‍යාව 65"
RUNNING_HEAD = re.compile(r"^(\d{1,3}\s+[^|\n]{2,40}\|[^|\n]{2,60}|[^|\n]{2,60}\|[^|\n]{2,40}\s+\d{1,3})$")
OBJECTIVES = re.compile(r"නිපුණතාව(?:ය)? ලබා ගත හැකි|කිරීමටත්,?$|හැකියාව ලැබේ\.?$|"
                        r"ඔබට හැකි වනු ඇත|මෙම පාඩම අධ්‍යයනය කිරීමෙන්")
NUMERIC = re.compile(r"^[\d.,:/x-]+$")
SENT_END_RX = re.compile(r"[.?!:;)”\"'’]\s*$")
POINTER_END = re.compile(r"පහත[^.]*[.:]\s*$")  # "... as follows." with the list cut off


def fix_sinhala(text: str) -> str:
    """Repair legacy-font conversion slips that survive extraction."""
    text = re.sub(r"([\u0d9a-\u0dc6])\u0dda\u200d([රය])", "\\1\u0dca\u200d\\2\u0dda", text)  # බේ‍ර -> බ්‍රේ
    text = re.sub(r"([\u0d9a-\u0dc6])\u0dd9\u200d([රය])", "\\1\u0dca\u200d\\2\u0dd9", text)  # බෙ‍ර -> බ්‍රෙ
    text = text.replace("\u200d\u0dca", "\u0dca\u200d")                     # ZWJ before hal
    text = re.sub(r"\u200d(?![රයෂ])", "", text)                              # stray joiners
    text = re.sub(r"(?<=[\u0d80-\u0dff])–(?=\s|$)", ".", text)               # '¡' = full stop
    return text


def sparse_table(table: str) -> bool:
    """Graph-paper grids and drawings detected as tables: mostly empty cells."""
    cells = [c.strip() for ln in table.split("\n") if not ln.startswith("|---")
             for c in ln.strip().strip("|").split("|")]
    return len(cells) >= 12 and sum(bool(c) for c in cells) / len(cells) < 0.4


def collapsed_table(table: str) -> bool:
    """A table whose cells hold a whole column of numbers/dates: the extractor merged
    rows, so the table no longer means anything."""
    for line in table.split("\n"):
        for cell in line.split("|"):
            toks = cell.split()
            if len(toks) >= 4 and sum(bool(NUMERIC.match(t)) for t in toks) >= 0.5 * len(toks):
                return True
    return False


def latexify(expr: str) -> str:
    expr = expr.replace("–", "-").replace("−", "-").replace("×", r"\times ").replace("÷", r"\div ")
    expr = expr.replace("≤", r"\leq ").replace("≥", r"\geq ").replace("≠", r"\neq ")
    expr = expr.replace("π", r"\pi ").replace("√", r"\sqrt ").replace("°", r"^\circ")
    return re.sub(r"\s+", " ", expr).strip()


def normalize_math(block: str) -> str:
    """Rule-extracted pages mark exponents as ^{3} / indices as _{2} without LaTeX
    delimiters; VLM pages already use $..$. Bring the former into the same form."""
    if not re.search(r"[\^_]\{", block):
        return block
    if "$" in block:  # VLM page: wrap only scripts left outside its $..$ spans
        parts = re.split(r"(\$\$.*?\$\$|\$[^$]*\$)", block, flags=re.S)
        return "".join(p if p.startswith("$") else
                       SCRIPT_TOKEN.sub(lambda m: f"${latexify(m.group(0))}$", p) for p in parts)
    out = []
    for line in block.split("\n"):
        if MATH_ONLY_LINE.match(line) and re.search(r"[\^_]\{|[=×÷]", line):
            out.append(f"${latexify(line)}$")
        else:
            out.append(SCRIPT_TOKEN.sub(lambda m: f"${latexify(m.group(0))}$", line))
    return "\n".join(out)


def clean_block(block: str) -> str | None:
    """Cleaned block, or None if the block is furniture or noise."""
    b = block.strip()
    if not b or GARBAGE.search(b) or FIGURE.match(b):
        return None
    if b.startswith("*") and b.endswith("*") and not b.startswith("* "):
        return None  # figure / table caption: the figure is not in the text
    b = FOOTER.sub("", b).replace("`ඵ", "ළු")
    b = re.sub(r"(?m)^[\x01-\x08]\s*", "- ", b)
    b = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", b).strip()
    b = fix_sinhala(b)
    if RUNNING_HEAD.match(b):
        return None
    if OBJECTIVES.search(b):
        return None  # "after this lesson you will be able to ..." furniture
    if b.startswith("|") and (collapsed_table(b) or sparse_table(b)):
        return None
    if not b or DOTS.search(b):
        return None  # fill-in-the-blank lines belong to exercises
    if (b.count("$") - 2 * b.count("$$")) % 2:
        b = b.replace("$", "")
    if "\n" in b and not b.startswith(("|", "$$")):
        lines = [ln for ln in b.split("\n") if not stray_line(ln)]
        if not lines:
            return None
        b = "\n".join(lines)
    b = drop_instructions(b)
    if not b.startswith(("#", "|", "$")) and len(b.split()) <= 3:
        return None  # fragment: ")", "( )", diagram label
    return normalize_math(b)


def stray_line(line: str) -> bool:
    """Diagram labels: single stray letters, or a line of 1-2 short Latin/number tokens."""
    toks = line.split()
    if not toks:
        return True
    stray = sum(len(t) == 1 and bool(STRAY.match(t)) and t not in PARTICLES for t in toks)
    label = len(toks) <= 2 and all(re.fullmatch(r"[A-Za-z0-9.,()°%+-]{1,4}", t) for t in toks)
    return label or stray / len(toks) > 0.5


INSTRUCTION = re.compile(r"(කරන්න|සොයන්න|ලියන්න|බලන්න|සාකච්ඡා කරන්න|අඳින්න|නිරීක්ෂණය කරන්න|"
                         r"සකස් කරන්න|පිළිතුරු සපයන්න|දක්වන්න|සඳහන් කරන්න)\s*[.?]?$")


def drop_instructions(block: str) -> str:
    """Remove sentences addressed to the student (activity steps); keep the rest."""
    if block.startswith(("|", "$", "#")) or "\n" in block or LIST_LINE.match(block):
        return block  # numbered steps of a procedure are content
    return " ".join(s for s in sentences(block) if not INSTRUCTION.search(s.strip())).strip()


BROKEN = {
    "broken_orphan_sign": re.compile(r"(?:^|[\s(“\"'])[\u0dca-\u0ddf]"),
    "broken_kombu": re.compile(r"(?:^|[^\u0d9a-\u0dc6\u200d\u0dca\u0dcf-\u0ddf])\u0dd9"),
    "broken_backtick": re.compile(r"`"),
    "broken_control": re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
    "broken_legacy_glyph": re.compile(r"[\u00a1-\u00b6\u00b8-\u00ba\u00bc-\u00d6\u00d8-\u00f6\u00f8-\u024f]"),
    "broken_raw_script": re.compile(r"[\w)][\^_]\{"),
}


def broken_text(r: dict) -> str | None:
    """Name of the first extraction defect in a row, or None."""
    text = "\n".join(m["content"] for m in r["messages"])
    no_math = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", " ", text, flags=re.S)
    for name, rx in BROKEN.items():
        if rx.search(no_math if name == "broken_raw_script" else text):
            return name
    words = r["messages"][1]["content"].split()
    if len(words) > 30 and len(set(words)) / len(words) < 0.25:
        return "broken_repetition"
    return None


def word_ngrams(text: str, n: int = 8) -> set[str]:
    w = re.findall(r"[\u0d80-\u0dff\u200d]+", text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def norm(text: str) -> str:
    return re.sub(r"\W+", "", text)


# ------------------------------------------------------------------ structure

def strip_heading(h: str) -> str:
    h = re.sub(r"^#+\s*", "", h)
    h = re.sub(r"^[\d.()\s]+(?=\S)", "", h)          # "1.2 ", "(3) "
    h = re.sub(r"^\(?([a-zA-Z]|[ivx]+)[.)]\s+", "", h)  # "a. ", "(ii) "
    h = h.strip(" :ඃ.-–").strip()
    return re.sub(r"\s+(හා|සහ|හෝ|ද)$", "", h)  # title cut at a line break: "... විවිධත්වය හා"


def iter_sections(text: str, seen_blocks: set[str], drops: Counter) -> Iterator[dict[str, Any]]:
    """Yield {chapter, heading, blocks} for every `## ` section (and the untitled
    part of a chapter before its first section)."""
    chapter, heading, blocks = "", "", []
    def flush():
        if blocks:
            return {"chapter": chapter, "heading": heading, "blocks": list(blocks)}
    for raw in re.split(r"\n\s*\n", text):
        raw = raw.strip()
        if raw.startswith("# "):
            if (s := flush()): yield s
            chapter, heading, blocks = strip_heading(raw), "", []
            continue
        if raw.startswith("#"):
            if (s := flush()): yield s
            heading, blocks = strip_heading(raw), []
            continue
        b = clean_block(raw)
        if b is None:
            drops["block_noise"] += 1
            continue
        key = norm(b)
        if len(key) > 40:
            if key in seen_blocks:
                drops["block_duplicate"] += 1
                continue
            seen_blocks.add(key)
        blocks.append(b)
    if (s := flush()): yield s


def is_list_block(b: str) -> bool:
    return bool(LIST_LINE.match(b))


def cap_blocks(blocks: list[str], cap: int) -> str:
    """Whole blocks up to `cap` chars; always at least the first one. A trailing
    block that only announces what follows ("පහත දැක්වේ.") is dropped."""
    kept, total = [], 0
    for b in blocks:
        if kept and total + len(b) > cap:
            break
        kept.append(b)
        total += len(b) + 2
    while len(kept) > 1 and (
            (POINTER_END.search(kept[-1]) and "\n" not in kept[-1])
            or (len(kept[-1].split()) <= 8 and not SENT_END_RX.search(kept[-1])
                and not kept[-1].startswith(("|", "$")) and not LIST_LINE.match(kept[-1]))):
        kept.pop()  # an announcement of omitted content, or a trailing label
    return "\n\n".join(kept)


def on_topic(topic: str, answer: str) -> bool:
    """The answer mentions the topic: some topic word (>= 3 letters) shares its first
    4 characters with a word of the answer. Catches captions and page headers that
    were taken for section headings."""
    stems = {w[:4] for w in re.findall(r"[\u0d80-\u0dff\u200d]{3,}", topic)}
    words = {w[:4] for w in re.findall(r"[\u0d80-\u0dff\u200d]{3,}", answer)}
    return not stems or bool(stems & words)


def sinhala_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(bool(SINHALA_CHAR.match(c)) for c in letters) / max(1, len(letters))


# ------------------------------------------------------------------ row builders

VERSE_SUBJECTS = ("sinhala", "buddhism", "christianity", "catholicism", "islam")


def unwrap(answer: str, subject: str) -> str:
    """Line breaks only carry meaning (verse) in the language and religion books;
    elsewhere they are prose wrapped around a figure, so re-join them."""
    if subject.startswith(VERSE_SUBJECTS):
        return answer
    out = []
    for b in answer.split("\n\n"):
        if "\n" in b and not b.startswith(("|", "$$")):
            lines = b.split("\n")
            joined = [lines[0]]
            for ln in lines[1:]:
                if LIST_LINE.match(ln):
                    joined.append(ln)
                else:
                    joined[-1] += " " + ln
            b = "\n".join(joined)
        out.append(b)
    return "\n\n".join(out)


def row(rid: str, kind: str, title: str, book: dict, prompt: str, answer: str) -> dict[str, Any]:
    if kind != "verse":
        answer = unwrap(answer, book["subject"])
    return {"id": rid, "source": "textbook", "kind": kind, "title": title,
            "subject": book["subject"], "grade": book["grade"],
            "messages": [{"role": "user", "content": prompt.strip()},
                         {"role": "assistant", "content": answer.strip()}]}


def pick(kind: str, key: str) -> str:
    pool = PROMPTS[kind]
    return pool[bucket(key + ":p", len(pool))]


DEFINITION = re.compile(r"^(?P<term>[^\s.,:;()]+(?: [^\s.,:;()]+){0,3}) යනු (?P<rest>.{15,400}?(?:ය|යි|වේ|වෙයි)\.)$")


def definition_rows(sec_id: str, sec: dict, book: dict, seen_terms: set[str]) -> list[dict]:
    out = []
    for b in sec["blocks"]:
        if b.startswith(("|", "$", "-")) or "\n" in b:
            continue
        for s in sentences(b):
            m = DEFINITION.match(s.strip())
            if not m:
                continue
            term = m["term"]
            if term in seen_terms or len(term) < 3 or not SINHALA_CHAR.search(term):
                continue
            seen_terms.add(term)
            rid = f"{sec_id}:def:{term}"
            out.append(row(rid, "definition", term, book,
                           pick("definition", rid).format(term=term), s.strip()))
    return out


VERSE_GLOSS = re.compile(r"^(අර්ථය|තේරුම|අදහස)\s*[:ඃ-]\s*")
ANWAYA = re.compile(r"^අන්වය\s*[:ඃ]")
INLINE_VERSE = re.compile(r"^\d+\.\s+(?P<verse>[^:]{20,400}?)\s+අන්වය\s*[:ඃ]\s*.+?\s+"
                          r"අර්ථය\s*[:ඃ]\s*(?P<gloss>.+?)(?=\s+විස්තර\b|$)", re.S)


def verse_rows(sec_id: str, sec: dict, book: dict) -> list[dict]:
    """A verse block followed by the book's meaning gloss for it."""
    out, bl = [], sec["blocks"]
    if not book["subject"].startswith("sinhala"):
        return out
    for i, b in enumerate(bl):
        # stanza, prose order and meaning merged into one paragraph by the extractor
        m = INLINE_VERSE.match(b)
        if m and len(m["gloss"]) >= MIN_SHORT:
            rid = f"{sec_id}:verse:{i}"
            out.append(row(rid, "verse", sec["chapter"], book,
                           pick("verse", rid).format(verse=m["verse"].strip()), m["gloss"].strip()))
    for i, b in enumerate(bl[:-1]):
        lines = b.split("\n")
        cut = next((j for j, ln in enumerate(lines) if ANWAYA.match(ln)), len(lines))
        lines = [ln for ln in lines[:cut] if not re.fullmatch(r"\d+\.?", ln.strip())]
        if (not (2 <= len(lines) <= 12) or b.startswith(("|", "$", "-")) or LIST_LINE.match(b)
                or max(len(ln) for ln in lines) > 70):
            continue
        for nxt in bl[i + 1:i + 3]:  # the අන්වය (prose order) may sit in between
            if VERSE_GLOSS.match(nxt):
                verse = "\n".join(lines)
                gloss = VERSE_GLOSS.sub("", nxt)
                if len(gloss) >= MIN_SHORT:
                    rid = f"{sec_id}:verse:{i}"
                    out.append(row(rid, "verse", sec["chapter"], book,
                                   pick("verse", rid).format(verse=verse), gloss))
                break
    return out


def example_row(sec_id: str, sec: dict, book: dict) -> dict | None:
    """නිදසුන N: first block is the problem, the rest its worked solution."""
    bl = sec["blocks"]
    if len(bl) < 2:
        return None
    question, solution = bl[0], "\n\n".join(bl[1:])
    if len(question) < 20 or len(solution) < 40 or not re.search(r"\$|=|\d", solution):
        return None
    rid = f"{sec_id}:example"
    return row(rid, "worked_example", sec["chapter"], book,
               pick("worked_example", rid).format(question=question), cap_blocks(bl[1:], DETAIL_CAP))


def section_row(sec_id: str, sec: dict, book: dict, drops: Counter) -> dict | None:
    blocks = [b for b in sec["blocks"] if not VERSE_GLOSS.match(b)]
    if not blocks:
        return None
    heading, chapter = sec["heading"], sec["chapter"]
    if heading and re.match(r"[A-Za-z]", heading):
        drops["latin_heading"] += 1  # labels / captions picked up as headings
        return None
    topic = heading or chapter
    if not topic:
        drops["no_topic"] += 1
        return None
    generic = bool(GENERIC.match(heading)) or (heading and len(heading.split()) == 1 and len(heading) < 8)
    list_share = sum(len(b) for b in blocks if is_list_block(b)) / max(1, sum(len(b) for b in blocks))

    kind = KIND_BY_BUCKET[bucket(sec_id, len(KIND_BY_BUCKET))]
    if list_share >= 0.6:
        kind = "list"
    answer = None
    if kind == "short":
        prose = next((b for b in blocks if not b.startswith(("|", "$", "-")) and "\n" not in b
                      and not is_list_block(b)), "")
        sents = sentences(prose)
        want = 2 + bucket(sec_id + ":n", 2)
        answer = ""
        for s in sents[:want]:
            if answer and len(answer) + len(s) > SHORT_MAX:
                break
            answer = f"{answer} {s}".strip()
        if len(answer) < MIN_SHORT or POINTER_END.search(answer):
            kind, answer = "detail", None
    if answer is None:
        answer = cap_blocks(blocks, DETAIL_CAP)
        if len(answer) < MIN_ANSWER:
            drops["section_too_short"] += 1
            return None
    if sinhala_share(answer) < (0.3 if book["kind"] == "technical" else 0.6):
        drops["not_sinhala"] += 1
        return None

    if not (kind == "detail" and generic) and not on_topic(topic, answer):
        drops["off_topic"] += 1
        return None
    if kind == "detail" and generic and chapter:
        prompt = pick("detail_ctx", sec_id).format(chapter=chapter, topic=heading)
    else:
        prompt = pick(kind, sec_id).format(topic=topic)
    return row(sec_id, kind, topic, book, prompt, answer)


def book_rows(path: Path, book: dict, seen_blocks: set, seen_terms: set, drops: Counter) -> list[dict]:
    rows = []
    for i, sec in enumerate(iter_sections(path.read_text(encoding="utf-8"), seen_blocks, drops)):
        sec_id = f"{book['id']}:{i}"
        heading = sec["heading"]
        if heading and SKIP_SECTION.search(heading):
            drops["exercise_section"] += 1
            continue
        if heading and SUMMARY.search(heading):
            ans = cap_blocks(sec["blocks"], DETAIL_CAP)
            if sec["chapter"] and len(ans) >= MIN_ANSWER:
                rows.append(row(f"{sec_id}:summary", "summary", sec["chapter"], book,
                                pick("summary", sec_id).format(chapter=sec["chapter"]), ans))
            continue
        if heading and EXAMPLE.match(heading):
            if (r := example_row(sec_id, sec, book)):
                rows.append(r)
            else:
                drops["example_unusable"] += 1
            continue
        rows += verse_rows(sec_id, sec, book)
        rows += definition_rows(sec_id, sec, book, seen_terms)
        if (r := section_row(sec_id, sec, book, drops)):
            rows.append(r)
    return rows


# ------------------------------------------------------------------ main

ROW_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("source", pa.string()),
    pa.field("kind", pa.string()),
    pa.field("title", pa.string()),
    pa.field("subject", pa.string()),
    pa.field("grade", pa.int32()),
    pa.field("messages", pa.list_(pa.struct([
        pa.field("content", pa.string()),
        pa.field("role", pa.string()),
    ]))),
])


def load_books(md_root: Path, arts: set[str]) -> list[tuple[Path, dict]]:
    out = []
    for f in sorted(md_root.glob("*/grade_*.md")):
        m = re.match(r"grade_(\d+)(?:_part_(\d+))?", f.stem)
        out.append((f, {"id": f"{f.parent.name}/{f.stem}", "subject": f.parent.name,
                        "grade": int(m[1]), "kind": "arts" if f.parent.name in arts else "technical"}))
    return out


def load_tokenizer(cfg: dict, override: str | None):
    """The training model's tokenizer; any SinLlama checkpoint shares it."""
    from transformers import AutoTokenizer
    for p in [override, cfg.get("tokenizer"), "models/SinLlama_wiki_uc_gen_mix", "models/SinLlama_cpt"]:
        if p and resolve(p).exists():
            return AutoTokenizer.from_pretrained(resolve(p))
    raise SystemExit("no tokenizer found: pass --tokenizer PATH")


def replay_rows(path: str, seed: int, drops: Counter, n_rows: int | None = None,
                token_budget: int | None = None, count=None,
                exclude_prompts: set[str] = frozenset()) -> list[dict]:
    """Cleaned UltraChat dialogues in a seeded random order, until n_rows rows or
    token_budget templated tokens. Already cleaned, so prepare() only trims a
    trailing user turn and repairs empty turns."""
    table = pq.read_table(path, columns=["prompt_id", "messages"])
    order = list(range(table.num_rows))
    random.Random(seed).shuffle(order)
    ids, msgs = table.column("prompt_id"), table.column("messages")
    rows, tokens = [], 0
    for i in order:
        if (n_rows is not None and len(rows) >= n_rows) or \
                (token_budget is not None and tokens >= token_budget):
            break
        kept, reason = prepare(msgs[i].as_py(), False, Counter())
        if kept is None:
            drops[reason] += 1
            continue
        if norm(kept[0]["content"]) in exclude_prompts:
            drops["prompt_in_eval"] += 1
            continue
        if count is not None:
            tokens += count(kept)
        rows.append({"id": ids[i].as_py(), "source": "ultrachat", "kind": "dialogue",
                     "title": "", "messages": kept})
    return rows


def write(rows: list[dict], path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=ROW_SCHEMA), path, compression="zstd")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "textbook_sft" / "config.yaml"))
    ap.add_argument("--show", type=int, default=0, help="print N example rows per kind")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--tokenizer", default=None, help="tokenizer for the token budget")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))["textbook_mix"]
    seed = cfg.get("seed", 42)
    md_root = resolve(cfg["markdown_dir"])
    out_dir = resolve(cfg["out_dir"])
    arts = set(json.loads((md_root / "_report.json").read_text()).keys())
    arts = {k.split("/")[0] for k in arts}

    drops: Counter = Counter()
    seen_blocks: set[str] = set()
    seen_terms: set[str] = set()
    rows: list[dict] = []
    for path, book in load_books(md_root, arts):
        rows += book_rows(path, book, seen_blocks, seen_terms, drops)

    # identical answers (shared chapters across the Design & Technology books, ...)
    uniq, seen_answers = [], set()
    for r in rows:
        key = norm(r["messages"][1]["content"])[:300]
        if key in seen_answers:
            drops["answer_duplicate"] += 1
            continue
        seen_answers.add(key)
        uniq.append(r)
    rows = uniq
    # rows still carrying extraction damage are cheaper to drop than to repair
    rows = [r for r in rows if not (broken := broken_text(r)) or drops.update([broken])]
    # a prompt that occurs with several answers teaches nothing consistent: keep the first
    by_prompt: dict[str, int] = {}
    uniq = []
    for r in rows:
        q = r["messages"][0]["content"]
        if q in by_prompt:
            drops["prompt_duplicate"] += 1
            continue
        by_prompt[q] = 1
        uniq.append(r)
    rows = uniq
    print(f"textbooks: {len(rows):,} rows; dropped "
          + ", ".join(f"{k}={v:,}" for k, v in drops.most_common()))

    # eval split by chapter, so no lesson leaks into both sides
    eval_pct = cfg.get("eval_percent", 4)
    is_eval = lambda r: bucket(f"{r['subject']}/{r['grade']}/{r['title']}:{seed}", 100) < eval_pct
    tb_train = [r for r in rows if not is_eval(r)]
    tb_eval = [r for r in rows if is_eval(r)]
    # the same passage can appear in two grades' books: such eval rows are not held out
    train_grams = set().union(*(word_ngrams(r["messages"][1]["content"]) for r in tb_train))
    leaked = [r for r in tb_eval if (g := word_ngrams(r["messages"][1]["content"]))
              and len(g & train_grams) / len(g) >= 0.5]
    tb_eval = [r for r in tb_eval if r not in leaked]
    drops["eval_overlaps_train"] = len(leaked)

    template = ChatTemplate.from_config(yaml.safe_load(open(args.config))["template"])
    tok = load_tokenizer(cfg, args.tokenizer)
    count = lambda msgs: len(tok(template.render(msgs)[0], add_special_tokens=False).input_ids)
    tb_tokens = sum(count(r["messages"]) for r in tb_train)

    uc_drops: Counter = Counter()
    uc_eval = replay_rows(str(resolve(cfg["ultrachat_eval"])), seed, uc_drops,
                          n_rows=cfg["ultrachat_eval_rows"])
    eval_prompts = {norm(r["messages"][0]["content"]) for r in uc_eval}
    budget = round(tb_tokens * cfg["ultrachat_token_ratio"])
    uc = replay_rows(str(resolve(cfg["ultrachat_train"])), seed, uc_drops, token_budget=budget,
                     count=count, exclude_prompts=eval_prompts)
    uc_tokens = sum(count(r["messages"]) for r in uc)
    for r in uc + uc_eval:
        r.update(subject="", grade=0)
    print(f"tokens (templated): textbook {tb_tokens:,} / ultrachat {uc_tokens:,} "
          f"({len(uc):,} train dialogues, {len(uc_eval):,} eval); dropped "
          + (", ".join(f"{k}={v:,}" for k, v in uc_drops.most_common()) or "none"))

    train = tb_train + uc
    random.Random(seed + 2).shuffle(train)

    print()
    by_subject = Counter(r["subject"] for r in tb_train)
    manifest = {
        "seed": seed, "detail_cap": DETAIL_CAP, "short_max": SHORT_MAX, "prompts": PROMPTS,
        "textbook_dropped": dict(drops), "ultrachat_dropped": dict(uc_drops),
        "textbook_train": describe(tb_train, "textbook train"),
        "textbook_eval": describe(tb_eval, "textbook eval"),
        "textbook_train_by_subject": dict(by_subject.most_common()),
        "ultrachat_train_rows": len(uc), "ultrachat_eval_rows": len(uc_eval),
        "textbook_train_tokens": tb_tokens, "ultrachat_train_tokens": uc_tokens,
        "ultrachat_token_ratio": cfg["ultrachat_token_ratio"],
        "train_rows": len(train),
    }
    print(f"  train total: {len(train):,} rows ({100 * len(uc) / max(1, len(train)):.1f}% ultrachat rows, "
          f"{100 * uc_tokens / max(1, uc_tokens + tb_tokens):.1f}% ultrachat tokens)")

    for kind in PROMPTS:
        for r in [r for r in tb_train if r["kind"] == kind][: args.show]:
            print(f"\n--- {kind} [{r['subject']} g{r['grade']}]\n>>> {r['messages'][0]['content'][:400]}"
                  f"\n<<< {r['messages'][1]['content'][:600]}")

    write(train, out_dir / "train_mix.parquet", args.dry_run)
    write(tb_eval, out_dir / "eval_textbook.parquet", args.dry_run)
    write(uc_eval, out_dir / "eval_uc.parquet", args.dry_run)
    if not args.dry_run:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"\n-> {out_dir}: train_mix.parquet, eval_textbook.parquet, eval_uc.parquet, manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
