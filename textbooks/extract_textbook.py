"""Extract Sri Lankan government textbooks (PDF) to structured Unicode Markdown.

The textbooks are typeset in legacy FM fonts (FMAbhaya, FMAbabld, ...), which store
Sinhala as ASCII/Latin-1 codes, mixed with occasional real-Unicode spans (IskoolaPota).
Every word is converted according to its own font, then the page is rebuilt in
reading order: full-width bands (title, wide tables) split the page; inside a band
the left column is read before the right; tables and captions stay where they sit.

usage: python textbooks/extract_textbook.py IN.pdf [OUT.md]
"""
import re
import sys
from pathlib import Path

import pdfplumber

# ---------------------------------------------------------------- FM -> Unicode

# Multi-character sequences first (longest match wins), then single glyphs.
FM_MULTI = {
    "ft": "ඓ", "tA": "ඒ", "ta": "ඒ", "T`": "ඕ", "W!": "ඌ",
    "`.": "ඟ", "`È": "ඳි", "`â": "ඬ්", "`o": "ඳ", "`v": "ඬ", "`c": "ඦ",
    "re": "රු", "rE": "රූ", "DD": "ෲ",
}
FM_SINGLE = {
    # independent vowels
    "w": "අ", "b": "ඉ", "B": "ඊ", "W": "උ", "R": "ඍ", "t": "එ", "T": "ඔ",
    # consonants
    "l": "ක", "L": "ඛ", ".": "ග", ">": "ඝ", "X": "ඞ", "Õ": "ඟ",
    "p": "ච", "P": "ඡ", "c": "ජ", "[": "ඤ", "{": "ඥ",
    "g": "ට", "G": "ඨ", "v": "ඩ", "V": "ඪ", "K": "ණ", "~": "ඬ",
    ";": "ත", ":": "ථ", "o": "ද", "O": "ධ", "k": "න", "|": "ඳ",
    "m": "ප", "M": "ඵ", "n": "බ", "N": "භ", "u": "ම", "U": "ඹ",
    "h": "ය", "r": "ර", ",": "ල", "j": "ව", "Y": "ශ", "I": "ෂ",
    "i": "ස", "y": "හ", "<": "ළ", "*": "ෆ",
    # vowel signs and modifiers
    "d": "ා", "e": "ැ", "E": "ෑ", "s": "ි", "S": "ී", "q": "ු", "Q": "ූ",
    "=": "ු", "+": "ූ", "D": "ෘ", "a": "්", "A": "්", "x": "ං", "!": "ෟ",
    "H": "්‍ය", "%": "්‍ර", "¾": "ර්", "/": "රැ", "?": "රෑ",
    # pre-composed glyphs
    "ñ": "මි", "ó": "මී", "ï": "ම්", "ì": "බි", "î": "බී", "ú": "වි",
    "ù": "වී", "õ": "ව්", "ß": "රි", "Í": "රී", "á": "ටි", "Á": "ටී",
    "‚": "ණි", "Œ": "ණී", "È": "දි", "§": "දී", "ÿ": "දු", "¥": "දූ",
    "Þ": "දා", "÷": "ඳු", "ê": "ධි", "Ê": "ජ්", "Ô": "ජී", "É": "ච්",
    "þ": "ඡ්", "›": "ශ්‍රී", "í": "බ්", "Ü": "ට්", "Ñ": "චි", "Ö": "චී",
    "à": "ටී", "ä": "ඩි", "Ó": "ථි", "¢": "ඳි", "´": "ඕ",
    "ø": "ද්‍ර", "¨": "ලු", "ð": "ජි", "ë": "ධී", "±": "දැ", "Ë": "ක්ෂ",
    "¿": "ළු", "ö": "ඹු", "â": "ඩ්", "Ò": "ථි", "ô": "ඹි", "µ": "ද්‍ය", "Å": "ෂි",
    "ã": "ද", "Æ": "ලූ", "À": "ඨි", "£": "ඳී", "ç": "ඬි", "×": "ඥා",
    "z": "“", "Z": "”", "¬": "-", "—": "“", "˜": "”",
    # punctuation (FM reuses ' and " for full stop and comma)
    "'": ".", '"': ",", "&": ")", "^": "(", "@": "?", "\\": "ඳා",
    "(": ":", "¡": "–", "#": "ඃ", "æ": "!", "¦": ";", "²": "•", "³": "•",
    "ò": "ඹ්", "å": "ඬ්", "®": "ඣ", "Ø": "ඤ", "è": "ධ්", "Î": "ඪි", "Ä": "ඛ්",
    "Ù": "ඞ්", "Ç": "ජා", "ª": "ඳූ", "é": "ඬු", "¯": "ඣි", "ý": "ඡි", "œ": "ර්ය",
    "ü": "ඤූ", "C": "ක්", "Ì": "ඏ", "Ï": "ඐ", "«": "", "š": "", "$": "/",
    # Roman numerals (two FM styles: upper case for chapters, lower case for items)
    "¸": "I", "¹": "V", "º": "X", "»": "i", "¼": "v", "½": "x",
}
KOMBU, KOMBU2 = "", ""  # placeholders for prefix vowel signs f / ff
FM_RE = re.compile("|".join(re.escape(k) for k in sorted(
    [*FM_MULTI, *FM_SINGLE, "ff", "f"], key=len, reverse=True)))
CONS = "(?:ක්ෂ|[ක-ෆ])(?:්\u200d[රය])?"  # consonant (or ක්ෂ), optionally with rakaransaya/yansaya
KOMBU_RULES = [
    (re.compile(KOMBU2 + f"({CONS})"), r"\1ෛ"),
    (re.compile(KOMBU + f"({CONS})ා්"), r"\1ෝ"),
    (re.compile(KOMBU + f"({CONS})ා"), r"\1ො"),
    (re.compile(KOMBU + f"({CONS})ෟ"), r"\1ෞ"),
    (re.compile(KOMBU + f"({CONS})්"), r"\1ේ"),
    (re.compile(KOMBU + f"({CONS})"), r"\1ෙ"),
]
# Sequences FM typists build visually that Unicode spells differently.
FIXUPS = [(re.compile(a), b) for a, b in [
    ("අා", "ආ"), ("අැ", "ඇ"), ("අෑ", "ඈ"), ("එ්", "ඒ"), ("ඔ්", "ඕ"), ("උෟ", "ඌ"),
    ("([ක-ෆ])([ා-ෟ])්\u200dර", "\\1්\u200dර\\2"),  # ä% / YS% : vowel sign before rakaransaya
    (r"([ි-ූ])\1+", r"\1"), ("ීි|ිී", "ී"),  # doubled / typo'd vowel signs
    ("ඵෙතිහාසික", "ඓතිහාසික"),            # ඓ typed as f + M
    ("`ද", "ඳ"), ("`ඩ", "ඬ"), ("`ග", "ඟ"), ("`ජ", "ඦ"),  # nasal prefix before a ligature
]]
unknown_glyphs = {}
page_errors = []


def fm_to_unicode(s):
    def sub(m):
        k = m.group(0)
        if k == "ff":
            return KOMBU2
        if k == "f":
            return KOMBU
        return FM_MULTI.get(k) or FM_SINGLE[k]

    out = FM_RE.sub(sub, s)
    for pat, rep in KOMBU_RULES:
        out = pat.sub(rep, out)
    for pat, rep in FIXUPS:
        out = pat.sub(rep, out)
    for ch in out:  # anything still Latin-1 > 0x7F was not in the table
        if "\x80" <= ch <= "ɏ" or ch in (KOMBU, KOMBU2):
            unknown_glyphs[ch] = unknown_glyphs.get(ch, 0) + 1
    return out.replace(KOMBU, "ෙ").replace(KOMBU2, "ෛ")


# ------------------------------------------------------------------ font roles

# Tamil / Arabic passages and dingbat bullets are dropped; every FM-layout Sinhala
# font (FMAbhaya, SARAWIPULA, 0KDBOLIDDA, ...) is converted; anything else
# (Times, Arial, IskoolaPota Unicode, ...) passes through unchanged.
DROP_FONT = re.compile(r"TAM|Tharmini|Baamini|Kalaham|Arabic|Wingdings|Webdings|Dingbat|"
                       r"MonotypeSorts|ESRI", re.I)
FM_FONT = re.compile(r"FM|SARA|0KD|Basuru|AKELANI|DL-|Chamodi|Helabasa|Emanee|Amantha|"
                     r"AaAnita|ABILIC", re.I)


def font_role(fontname):
    name = re.sub(r"^[A-Z]{6}\+", "", fontname)
    if DROP_FONT.search(name):
        return "drop"
    return "fm" if FM_FONT.search(name) else "plain"


def is_bold(fontname):
    return bool(re.search(r"bld|Bold|BOLD|Black|Heavy", fontname))


# ------------------------------------------------------------- layout rebuild

TOP_MARGIN, BOTTOM_MARGIN = 0.045, 0.07   # running heads, page numbers, print notice
LIST_ITEM = re.compile(r"^((\d+|[a-zA-Z]|[ivx]+)[.)] |• ?|- )")
SENTENCE_END = re.compile(r"[.?!:;)”\"']$")
_FIG = r"(රූපය|රූප සටහන|ඡායාරූපය|වගුව|සිතියම|ප්‍රස්තාරය)"
CAPTION = re.compile(rf"^(\d+(\.\d+)+\s*{_FIG}|{_FIG}\s*(අංක)?\s*\d)")


def word_text(w):
    text = fm_to_unicode(w["text"]) if w["role"] == "fm" else w["text"]
    return f'{w["script"]}{{{text}}}' if w.get("script") else text


MATH_SCRIPTS = False  # technical books: mark superscripts ^{..} and subscripts _{..}


def mark_scripts(ws):
    """A word clearly smaller than the line's main text and shifted off its baseline
    is an exponent (raised) or a subscript / chemical index (lowered)."""
    main = max(w["size"] for w in ws)
    base = sorted(w["bottom"] for w in ws if w["size"] > 0.85 * main)[0:1]
    if not base:
        return
    base = sorted(w["bottom"] for w in ws if w["size"] > 0.85 * main)[
        len([w for w in ws if w["size"] > 0.85 * main]) // 2]
    for w in ws:
        if w["size"] < 0.8 * main:
            if w["bottom"] < base - 0.15 * main:
                w["script"] = "^"
            elif w["bottom"] > base + 0.08 * main:
                w["script"] = "_"


def words_to_lines(words):
    """Group words into lines. Full-size words form the lines (a gap > 20pt, a column
    gutter, starts a new one); smaller words (exponents, indices, list numbers in
    another font) then join the line they overlap most."""
    if not words:
        return []
    sizes = sorted(w["size"] for w in words)
    median = sizes[len(sizes) // 2]
    big = [w for w in words if w["size"] >= 0.8 * median]
    small = [w for w in words if w["size"] < 0.8 * median]
    lines = []

    def fits(ln, w):
        overlap = min(ln["bottom"], w["bottom"]) - max(ln["top0"], w["top"])
        height = min(ln["bottom"] - ln["top0"], w["bottom"] - w["top"])
        near = w["x0"] < ln["x1"] + 20 and w["x1"] > ln["x0"] - 20
        return overlap, height, near

    for w in sorted(big, key=lambda w: (round(w["bottom"]), w["x0"])):
        for ln in lines:
            overlap, height, near = fits(ln, w)
            if overlap > 0.6 * height and near:
                break
        else:
            ln = {"words": [], "x0": w["x0"], "x1": w["x1"], "top0": w["top"], "bottom": w["bottom"]}
            lines.append(ln)
        ln["words"].append(w)
        ln["x0"], ln["x1"] = min(ln["x0"], w["x0"]), max(ln["x1"], w["x1"])
    for w in sorted(small, key=lambda w: (round(w["bottom"]), w["x0"])):
        best = None
        for ln in lines:
            overlap, height, near = fits(ln, w)
            if near and overlap > 0.3 * height and (best is None or overlap > best[0]):
                best = (overlap, ln)
        if best is None:  # stand-alone small text (caption line, label)
            ln = {"words": [], "x0": w["x0"], "x1": w["x1"], "top0": w["top"], "bottom": w["bottom"]}
            lines.append(ln)
        else:
            ln = best[1]
        ln["words"].append(w)
        ln["x0"], ln["x1"] = min(ln["x0"], w["x0"]), max(ln["x1"], w["x1"])
    for ln in lines:
        ws = sorted(ln["words"], key=lambda w: w["x0"])
        if MATH_SCRIPTS:
            mark_scripts(ws)
        text = word_text(ws[0])
        for prev, w in zip(ws, ws[1:]):
            glued = ((w["fontname"] != prev["fontname"] or w.get("script") or prev.get("script"))
                     and w["x0"] - prev["x1"] < 1.5)
            text += ("" if glued else " ") + word_text(w)
        ln["text"] = re.sub(r" ([.,)])", r"\1", text).strip()
        n = sum(len(w["text"]) for w in ws)
        ln["size"] = sum(w["size"] * len(w["text"]) for w in ws) / n
        ln["bold"] = sum(len(w["text"]) for w in ws if is_bold(w["fontname"])) / n > 0.6
        ln["top"] = min(w["top"] for w in ws)
    return [ln for ln in lines if ln["text"]]


class Book:
    """Per-book state: body font size decides what counts as heading / small print."""

    def __init__(self, body_size, min_size=0.65):
        self.body = body_size
        self.min_size = min_size  # smaller text is dropped (map labels); lower for maths

    def kind(self, ln):
        if ln["size"] >= 1.6 * self.body:
            return "title"
        if len(ln["text"]) < 120 and (ln["size"] >= self.body + 1
                                      or (ln["bold"] and not LIST_ITEM.match(ln["text"]))):
            return "heading"
        return "body"

    def group_paragraphs(self, lines, width):
        """Top-sorted lines of one column -> (top, markdown) blocks. A vertical gap only ends a
        paragraph after a sentence end, so gaps left by figures are bridged."""
        if not lines:
            return []
        blocks, cur = [], []
        for ln in lines:
            kind = self.kind(ln)
            if cur:
                last = cur[-1]
                gap = ln["top"] - last["bottom"] > 0.6 * ln["size"]
                size_jump = abs(ln["size"] - last["size"]) > 1
                wrapped_heading = (kind == "heading" == self.kind(last) and not gap
                                   and not SENTENCE_END.search(last["text"]))
                wrapped_item = (LIST_ITEM.match(cur[0]["text"]) and not gap and not size_jump
                                and not LIST_ITEM.match(ln["text"]))
                if wrapped_heading or wrapped_item:
                    pass
                elif (kind != self.kind(cur[0]) or kind == "heading" or size_jump
                        or LIST_ITEM.match(ln["text"]) or CAPTION.match(ln["text"])
                        or (gap and (SENTENCE_END.search(last["text"])
                                     or LIST_ITEM.match(cur[0]["text"])))):
                    blocks.append(cur)
                    cur = []
            cur.append(ln)
        blocks.append(cur)
        return [(b[0]["top"], self.render_block(b, width)) for b in blocks]

    def render_block(self, lines, col_width):
        kind = self.kind(lines[0])
        texts = [l["text"] for l in lines]
        # verse: several short lines that are not a justified paragraph keep their breaks
        short = sum(l["x1"] - l["x0"] < 0.7 * col_width for l in lines[:-1])
        rights = [l["x1"] for l in lines[:-1]]
        justified = len(rights) >= 2 and max(rights) - min(rights) < 4  # prose beside a picture
        verse = (kind == "body" and len(lines) >= 2 and short >= 0.8 * (len(lines) - 1)
                 and not justified)
        text = "\n".join(texts) if verse else re.sub(r"\s+", " ", " ".join(texts))
        if kind == "heading" and lines[0]["size"] < self.body + 1 and len(text) > 100:
            kind = "body"  # a bold quotation / key passage, not a heading
        text = re.sub(r"^• ?", "- ", text, flags=re.M)
        if CAPTION.match(text) and kind != "title":
            text = f"*{text}*"
        elif kind == "title":
            text = "# " + text
        elif kind == "heading":
            text = "## " + text
        return text


def render_table(page, table):
    rows = []
    for row in table.rows:
        cells = []
        for bbox in row.cells:
            if bbox is None:
                cells.append("")
                continue
            x0, top, x1, bottom = bbox
            bbox = (max(x0, page.bbox[0]), max(top, page.bbox[1]),
                    min(x1, page.bbox[2]), min(bottom, page.bbox[3]))
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                cells.append("")
                continue
            ws = [w for w in page.crop(bbox).extract_words(extra_attrs=["fontname", "size"])
                  if font_role(w["fontname"]) != "drop"]
            for w in ws:
                w["role"] = font_role(w["fontname"])
            text = " ".join(l["text"] for l in sorted(words_to_lines(ws), key=lambda l: l["top"]))
            cells.append(text.replace("|", "\\|"))
        rows.append(cells)
    # decorative frames (a heading box, a shaded panel) are detected as tables too:
    # require two rows with 2+ filled cells and more than one distinct cell text
    filled = [c for r in rows for c in r if c.strip()]
    if sum(sum(bool(c.strip()) for c in r) >= 2 for r in rows) < 2 or len(set(filled)) < 3:
        return None
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    return "\n".join(md + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def page_words(page):
    """Words with their font role. Pages that carry explicit space glyphs are split
    on those, in content-stream (typing) order: positional sorting tears FM words
    apart at zero-width hal glyphs and letter-spacing in justified books."""
    chars = page.chars
    if chars and sum(c["text"] == " " for c in chars) > 0.06 * len(chars):
        words = stream_words(chars)
    else:
        words = page.extract_words(extra_attrs=["fontname", "size"])
    out = []
    for w in words:
        w["role"] = font_role(w["fontname"])
        if w["role"] != "drop":
            out.append(w)
    return out


def stream_words(chars):
    words, cur = [], []

    def flush():
        if cur:
            words.append({"text": "".join(c["text"] for c in cur),
                          "x0": min(c["x0"] for c in cur), "x1": max(c["x1"] for c in cur),
                          "top": min(c["top"] for c in cur),
                          "bottom": max(c["bottom"] for c in cur),
                          "fontname": cur[0]["fontname"], "size": cur[0]["size"]})
            cur.clear()

    for c in chars:
        if c["text"].isspace():
            flush()
            continue
        if cur:
            p = cur[-1]
            new_line = abs(c["bottom"] - p["bottom"]) > 0.5 * max(c["size"], 1)
            jump = c["x0"] < p["x0"] - 2 * c["size"] or c["x0"] > p["x1"] + 0.6 * c["size"]
            if (new_line or jump or c["fontname"] != p["fontname"]
                    or abs(c["size"] - p["size"]) > 0.5):
                flush()
        cur.append(c)
    flush()
    return words


def extract_page(page, book):
    page = page.dedupe_chars(tolerance=2)  # faux-bold headings are drawn twice
    H = page.height
    tables = []
    for t in page.find_tables():
        if len(t.rows) >= 2 and max(len(r.cells) for r in t.rows) >= 2:
            md = render_table(page, t)
            if md:
                tables.append((t, md))
    tboxes = [t.bbox for t, _ in tables]

    def in_table(w):
        return any(b[0] - 1 <= w["x0"] and w["x1"] <= b[2] + 1 and
                   b[1] - 1 <= w["top"] and w["bottom"] <= b[3] + 1 for b in tboxes)

    words = [w for w in page_words(page)
             if TOP_MARGIN * H < w["top"] and w["bottom"] < (1 - BOTTOM_MARGIN) * H
             and book.min_size * book.body <= w["size"] < 2.5 * book.body  # labels / badges
             and not in_table(w)]
    lines = words_to_lines(words)
    if is_contents_page(lines):
        return []
    lines = pair_verse_halves([l for l in lines if not re.fullmatch(r"\d{1,3}", l["text"])])
    items = [{"top": l["top"], "bottom": l["bottom"], "x0": l["x0"], "x1": l["x1"], "line": l}
             for l in lines]
    for t, md in tables:
        x0, top, x1, bottom = t.bbox
        items.append({"top": top, "bottom": bottom, "x0": x0, "x1": x1, "table": md})
    if not items:
        return []

    # Column split: lines crossing the middle span the page; others are left/right.
    left, right = min(it["x0"] for it in items), max(it["x1"] for it in items)
    mid = (left + right) / 2
    for it in items:
        it["col"] = ("span" if it["x0"] < mid - 15 and it["x1"] > mid + 15
                     else "L" if it["x1"] <= mid + 15 else "R")
    if sum(it["col"] == "R" for it in items) < 3:  # single-column page
        for it in items:
            it["col"] = "span"

    # Consecutive spanning items form a band; between bands read left then right.
    out, run = [], []
    for it in sorted(items, key=lambda it: it["top"]) + [None]:
        if it is not None and (not run or (it["col"] == "span") == (run[0]["col"] == "span")):
            run.append(it)
            continue
        if run and run[0]["col"] == "span":
            out += render_column(run, book, right - left)
        elif run:
            out += render_column([i for i in run if i["col"] == "L"], book, mid - left)
            out += render_column([i for i in run if i["col"] == "R"], book, right - mid)
        run = [it] if it is not None else []
    return out


def pair_verse_halves(lines):
    """Poems are often set as two half-lines either side of the page centre. Those
    halves share a baseline and are short, unlike justified two-column prose, so
    re-join them into one line per verse row (halves separated by two spaces)."""
    if len(lines) < 4:
        return lines
    left = min(l["x0"] for l in lines)
    mid = (left + max(l["x1"] for l in lines)) / 2
    half = mid - left
    L = [l for l in lines if l["x1"] <= mid + 15]
    R = [l for l in lines if l["x0"] >= mid - 15]
    pairs, taken = [], set()
    for r in R:  # pair each right half with the nearest left line on its baseline
        cands = [l for l in L if abs(r["bottom"] - l["bottom"]) < 3 and id(l) not in taken]
        if not cands:
            continue
        l = max(cands, key=lambda l: l["x1"])
        if (l["x1"] - l["x0"] < 0.75 * half and r["x1"] - r["x0"] < 0.75 * half
                and not LIST_ITEM.match(r["text"])):  # a grid of bullets is a list, not verse
            pairs.append((l, r))
            taken.add(id(l))
    if len(pairs) < 3:
        return lines
    used = set()
    for l, r in pairs:
        l["text"] = l["text"] + "  " + r["text"]
        l["x1"] = r["x1"]
        used.add(id(r))
    return [l for l in lines if id(l) not in used]


def is_contents_page(lines):
    """Table of contents: a 'පටුන' title, or many bare page-number lines."""
    if any(l["text"].strip(" .") in ("පටුන", "පටුව", "විෂය පටුන") for l in lines):
        return True
    if MATH_SCRIPTS:  # maths pages are full of bare numbers
        return False
    numeric = sum(bool(re.fullmatch(r"[\d\s.-]+", l["text"])) for l in lines)
    return len(lines) >= 8 and numeric >= 0.25 * len(lines)


def render_column(items, book, width):
    """Paragraphs of one column; a table goes before the first paragraph below it."""
    tables = sorted((it["top"], it["table"]) for it in items if "table" in it)
    out = []
    for top, block in book.group_paragraphs([it["line"] for it in items if "line" in it], width):
        while tables and tables[0][0] < top:
            out.append(tables.pop(0)[1])
        out.append(block)
    return out + [t[1] for t in tables]


def merge_breaks(blocks):
    """Re-join a paragraph that a column or page break cut mid-sentence."""
    merged = []
    for b in blocks:
        if (merged and is_prose(merged[-1]) and is_prose(b)
                and not SENTENCE_END.search(merged[-1]) and not LIST_ITEM.match(b)):
            merged[-1] += " " + b
        else:
            merged.append(b)
    return merged


def is_prose(block):
    return (not block.startswith(("#", "*", "|", "$", "[")) and "\n" not in block
            and not LIST_ITEM.match(block))


def body_size(pdfs):
    """Most common font size (by character count) of converted Sinhala text."""
    from collections import Counter
    sizes = Counter()
    for path in pdfs:
        with pdfplumber.open(path) as pdf:
            step = max(1, len(pdf.pages) // 25)
            for page in pdf.pages[::step]:
                for ch in page.chars:
                    if font_role(ch["fontname"]) == "fm":
                        sizes[round(ch["size"] * 2) / 2] += 1
    return sizes.most_common(1)[0][0] if sizes else 12


def extract_pages(pdfs, book):
    """Yield (pdf path, page number, page, blocks) for every non-front-matter page."""
    for path in pdfs:
        with pdfplumber.open(path) as pdf:
            in_front = True
            for page in pdf.pages:
                try:
                    page_blocks = extract_page(page, book)
                    if in_front:
                        if page.page_number > FRONT_PAGES or starts_lesson(page, page_blocks):
                            in_front = False
                        elif is_front_matter(page_blocks) or has_lesson_later(pdf, book):
                            continue
                    yield path, page.page_number, page, page_blocks
                except Exception as e:  # one malformed page must not sink the book
                    page_errors.append(f"{path.name} p{page.page_number}: {e}")
                page.flush_cache()


def assemble(blocks):
    return "\n\n".join(b for b in merge_breaks(blocks) if not is_garbled(b)) + "\n"


def extract_book(pdfs):
    book = Book(body_size(pdfs))
    return assemble([b for *_, blocks in extract_pages(pdfs, book) for b in blocks])


FRONT_PAGES = 15
FRONT_MATTER = re.compile(
    r"ජාතික ගීය|අපි වෙමු එක මවකගෙ|අධ්‍යාපන ප්‍රකාශන දෙපාර්තමේන්තුව|මණ්ඩල|පෙරවදන|නියාමනය හා අධීක්ෂණය|සංස්කාරක මණ්ඩලය|ලේඛක මණ්ඩලය|සෝදුපත්|"
    r"පිටකවර|පරිගණක අක්ෂර|ප්‍රකාශන කොමසාරිස්|සම්බන්ධීකරණය|මෙහෙයවීම|පණිවිඩය|"
    r"ජාතික ප්‍රතිඥාව|අධ්‍යාපන අමාත්‍ය|මුද්‍රණය හා ප්‍රකාශනය|ISBN|ශ්‍රේණිය$")


def starts_lesson(page, blocks):
    """First real chapter page: a chapter title, past the cover, no credits markers."""
    return (page.page_number >= 3 and any(b.startswith("# ") for b in blocks[:3])
            and not is_front_matter(blocks))


def has_lesson_later(pdf, book, _cache={}):
    """Does a chapter-title page exist within the front-matter window? If so, every
    page before it is front matter; if not, only marker-matched pages are dropped."""
    key = pdf.stream.name if hasattr(pdf.stream, "name") else id(pdf)
    if key not in _cache:
        _cache[key] = any(starts_lesson(p, extract_page(p, book))
                          for p in pdf.pages[2:FRONT_PAGES])
    return _cache[key]


def is_front_matter(blocks):
    """Cover, anthem, preface / minister's message and credits pages."""
    return not blocks or any(FRONT_MATTER.search(b) for b in blocks[:6])


def is_garbled(block):
    """Text from charts / rotated labels / visual-order Unicode comes out as runs of
    lone letters and orphaned vowel signs; such blocks are dropped."""
    tokens = re.findall(r"[^\s|#*-]+", block)
    if len(tokens) < 4:
        return False
    junk = sum(bool(re.fullmatch(r"[\u0d80-\u0dff]", t) or re.match(r"[\u0dca-\u0ddf]", t))
               for t in tokens)
    return junk > 0.25 * len(tokens)


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".md")
    dst.write_text(extract_book([src]), encoding="utf-8")
    print(f"wrote {dst}")
    if unknown_glyphs:
        print("unmapped glyphs:", unknown_glyphs, file=sys.stderr)


if __name__ == "__main__":
    main()


MATH_LINE = re.compile(r"\^\{|_\{|[=×÷√π∴≤≥≠±]|\d\s*[+−–-]\s*\d")


def fraction_bars(page):
    """Short horizontal rules with text right above and right below: stacked
    fractions (and long division), which only a layout-aware reader can rebuild."""
    segs = [(l["x0"], l["x1"], l["top"]) for l in page.lines if abs(l["top"] - l["bottom"]) < 1.5]
    segs += [(r["x0"], r["x1"], r["top"]) for r in page.rects
             if r["height"] < 1.5 and 3 < r["width"]]
    chars = [c for c in page.chars if not c["text"].isspace()]
    n = 0
    for x0, x1, y in segs:
        if not 3 < x1 - x0 < 90:
            continue
        mid = (x0 + x1) / 2
        above = any(c["x0"] - 2 <= mid <= c["x1"] + 8 and 0 < y - c["bottom"] < 9 for c in chars)
        below = any(c["x0"] - 2 <= mid <= c["x1"] + 8 and 0 < c["top"] - y < 9 for c in chars)
        n += above and below
    return n


def math_score(page, blocks):
    """(fraction bars, lines with math) on a page."""
    lines = [l for b in blocks for l in b.split("\n")]
    return fraction_bars(page), sum(bool(MATH_LINE.search(l)) for l in lines)
