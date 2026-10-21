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
    "`.": "ඟ", "`È": "ඳි", "`o": "ඳ", "`v": "ඬ", "`c": "ඦ",
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
    "Þ": "දා", "÷": "ඳු", "ê": "ධි", "Ê": "ජි", "Ô": "ජී", "É": "ච්",
    "þ": "ඡ්", "›": "ශ්‍රී", "í": "බ්", "Ü": "ට්", "Ñ": "චි", "Ö": "චී",
    "à": "ටී", "ä": "ඩි", "Ó": "ථි", "¢": "ඳි", "´": "ඕ",
    "ø": "ද්‍ර", "¨": "ලු", "ð": "ජි", "ë": "ධී", "±": "දැ", "Ë": "ක්ෂ",
    "¿": "ළු", "ö": "ඹු", "â": "ඩ්", "Ò": "ථි", "ô": "ඹි", "µ": "ද්‍ය", "Å": "ෂි",
    "ã": "ද", "Æ": "ලූ", "À": "ඨි", "£": "ඳී", "ç": "ඬි", "×": "ඥා",
    "z": "“", "Z": "”", "¬": "-",
    # punctuation (FM reuses ' and " for full stop and comma)
    "'": ".", '"': ",", "&": ")", "^": "(",
}
KOMBU, KOMBU2 = "", ""  # placeholders for prefix vowel signs f / ff
FM_RE = re.compile("|".join(re.escape(k) for k in sorted(
    [*FM_MULTI, *FM_SINGLE, "ff", "f"], key=len, reverse=True)))
CONS = "[ක-ෆ](?:්‍[රය])?"  # consonant, optionally with rakaransaya/yansaya
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
    ("([ක-ෆ])ි්\u200dර", "\\1්\u200dරි"),  # ä% = ඩි + rakaransaya -> ඩ්‍රි
    (r"([ි-ූ])\1+", r"\1"), ("ීි|ිී", "ී"),  # doubled / typo'd vowel signs
    ("ඵෙතිහාසික", "ඓතිහාසික"),            # ඓ typed as f + M
]]
unknown_glyphs = {}


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


def is_fm(font):
    return "FM" in font


# ------------------------------------------------------------- layout rebuild

BODY_SIZE = 12
FOOTER_Y = 725  # government-print notice + page number live below this


def word_text(w):
    return fm_to_unicode(w["text"]) if is_fm(w["fontname"]) else w["text"]


def words_to_lines(words, tol=3):
    """Group words into lines by baseline; join same-line pieces split by font."""
    lines = []
    for w in sorted(words, key=lambda w: (round(w["bottom"]), w["x0"])):
        for ln in lines:
            if abs(ln["bottom"] - w["bottom"]) <= tol and w["x0"] >= ln["x0"] - 1:
                if w["x0"] < ln["x1"] + 20:  # word gaps <= 15pt, gutter >= 24pt
                    ln["words"].append(w)
                    ln["x1"] = max(ln["x1"], w["x1"])
                    break
        else:
            lines.append({"words": [w], "x0": w["x0"], "x1": w["x1"],
                          "top": w["top"], "bottom": w["bottom"]})
    for ln in lines:
        ws = sorted(ln["words"], key=lambda w: w["x0"])
        text = word_text(ws[0])
        for prev, w in zip(ws, ws[1:]):
            glued = w["fontname"] != prev["fontname"] and w["x0"] - prev["x1"] < 1
            text += ("" if glued else " ") + word_text(w)
        text = re.sub(r" ([.,)])", r"\1", text)
        ln["text"] = text
        main = max(ws, key=lambda w: len(w["text"]))
        ln["size"] = round(main["size"])
        ln["bold"] = "Bold" in main["fontname"] or "bld" in main["fontname"]
        ln["top"] = min(w["top"] for w in ws)
    return lines


def line_kind(ln):
    if ln["size"] >= 20:
        return "title"
    if ln["size"] > BODY_SIZE or ln["bold"]:
        return "heading"
    if ln["size"] < BODY_SIZE:
        return "caption"
    return "body"


LIST_ITEM = re.compile(r"^\d+\.? ")
SENTENCE_END = re.compile(r"[.?!:)]$")


def group_paragraphs(lines):
    """Group one column's top-sorted lines into (kind, text, top, bottom) blocks.

    A vertical gap only ends a paragraph if the text before it ends a sentence:
    gaps left behind by tables/figures that interrupt a paragraph are bridged.
    """
    blocks, cur, last = [], [], None
    for ln in lines:
        kind = line_kind(ln)
        if cur:
            gap = ln["top"] - last["bottom"] > 0.6 * ln["size"]
            prev_list = LIST_ITEM.match(cur[0]["text"])
            if (kind != line_kind(cur[0]) or kind == "heading"
                    or LIST_ITEM.match(ln["text"])
                    or (gap and (SENTENCE_END.search(last["text"]) or prev_list))):
                blocks.append(cur)
                cur = []
        cur.append(ln)
        last = ln
    if cur:
        blocks.append(cur)
    return [(line_kind(b[0]), re.sub(r"\s+", " ", " ".join(l["text"] for l in b)).strip(),
             b[0]["top"], b[-1]["bottom"]) for b in blocks]


def to_markdown(kind, text):
    return {"title": "# {}", "heading": "## {}", "caption": "*{}*"}.get(kind, "{}").format(text)


def render_table(page, table):
    rows = []
    for row in table.rows:
        cells = []
        for bbox in row.cells:
            if bbox is None:
                cells.append("")
                continue
            ws = page.crop(bbox).extract_words(extra_attrs=["fontname", "size"])
            cells.append(" ".join(ln["text"] for ln in sorted(
                words_to_lines(ws), key=lambda l: l["top"])).replace("|", "\\|"))
        rows.append(cells)
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(md)


def extract_page(page):
    page = page.dedupe_chars()  # faux-bold headings are drawn twice
    mid = page.width / 2
    tables = page.find_tables()
    tboxes = [t.bbox for t in tables]

    def in_table(w):
        return any(b[0] - 1 <= w["x0"] and w["x1"] <= b[2] + 1 and
                   b[1] - 1 <= w["top"] and w["bottom"] <= b[3] + 1 for b in tboxes)

    words = [w for w in page.extract_words(extra_attrs=["fontname", "size"])
             if w["top"] < FOOTER_Y and not in_table(w)
             and not (w["size"] >= 40)]  # big chapter-number badge
    items = []  # (top, x0, x1, bottom, kind, payload)
    for ln in words_to_lines(words):
        items.append({"top": ln["top"], "bottom": ln["bottom"], "x0": ln["x0"],
                      "x1": ln["x1"], "line": ln})
    for t in tables:
        x0, top, x1, bottom = t.bbox
        items.append({"top": top, "bottom": bottom, "x0": x0, "x1": x1,
                      "table": render_table(page, t)})

    # Full-width elements (title, wide tables, their captions) split the page
    # into horizontal bands; each band is read left column, then right column.
    def full_width(it):
        return it["x1"] - it["x0"] > 0.7 * (page.width - 100)

    seps = sorted((it for it in items if full_width(it)), key=lambda it: it["top"])
    out, y = [], -1
    for sep in seps + [None]:
        y_end = sep["top"] if sep else 1e9
        band = [it for it in items if not full_width(it) and y <= it["top"] < y_end]
        for col in (lambda it: it["x0"] < mid, lambda it: it["x0"] >= mid):
            out += render_items(sorted(filter(col, band), key=lambda it: it["top"]))
        if sep:
            out += render_items([sep])
            y = sep["bottom"]
    return merge_column_breaks(out)


def merge_column_breaks(blocks):
    """Re-join a paragraph that the column (or band) break cut mid-sentence."""
    merged = []
    for b in blocks:
        b = re.sub(r" ([.,)])", r"\1", b)
        if (merged and is_body(merged[-1]) and is_body(b)
                and not SENTENCE_END.search(merged[-1]) and not LIST_ITEM.match(b)):
            merged[-1] += " " + b
        else:
            merged.append(b)
    return merged


def is_body(block):
    return not block.startswith(("#", "*", "|")) and not LIST_ITEM.match(block)


def render_items(items):
    """Render a column; floats (tables, captions) go after the paragraph they cut."""
    lines = [it["line"] for it in items if "line" in it and line_kind(it["line"]) != "caption"]
    floats = [it for it in items if "table" in it]
    floats += [{"top": l["top"], "bottom": l["bottom"], "line": l}
               for it in items if "line" in it for l in [it["line"]] if line_kind(l) == "caption"]
    float_blocks = []
    caps = group_paragraphs(sorted((f["line"] for f in floats if "line" in f), key=lambda l: l["top"]))
    float_blocks += [(top, bottom, to_markdown(k, t)) for k, t, top, bottom in caps]
    float_blocks += [(f["top"], f["bottom"], f["table"]) for f in floats if "table" in f]
    float_blocks.sort()
    out = []
    for kind, text, top, bottom in group_paragraphs(lines):
        while float_blocks and float_blocks[0][0] < top:
            out.append(float_blocks.pop(0)[2])
        out.append(to_markdown(kind, text))
        while float_blocks and float_blocks[0][0] < bottom:
            out.append(float_blocks.pop(0)[2])
    return out + [b[2] for b in float_blocks]


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".md")
    with pdfplumber.open(src) as pdf:
        pages = [f"<!-- page {p.page_number} -->\n\n" + "\n\n".join(extract_page(p))
                 for p in pdf.pages]
    dst.write_text("\n\n".join(pages) + "\n", encoding="utf-8")
    print(f"wrote {dst}")
    if unknown_glyphs:
        print("unmapped glyphs:", unknown_glyphs, file=sys.stderr)


if __name__ == "__main__":
    main()
