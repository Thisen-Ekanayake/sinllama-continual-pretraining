#!/usr/bin/env python3
"""Chunk textbooks/markdown/<subject>/grade_N[_part_K].md into retrieval passages.

Paragraph-packing: blank-line paragraphs are packed into passages of about
TARGET characters (never above HARD; an over-long paragraph is split on
sentence ends). Each passage remembers the most recent real heading (the OCR
leaves numeric junk headings like "## 3] 4] 10]", which are skipped). Markdown
markup is stripped so both embedders and the LLM see plain text.

usage: python chunk_textbooks.py textbooks/markdown out/chunks.jsonl
"""
import json
import os
import re
import sys
import unicodedata

TARGET, HARD, MIN_SI = 700, 1000, 80   # chars; MIN_SI = min Sinhala letters kept
SI = re.compile(r"[඀-෿]")


def n_si(s):
    return len(SI.findall(s))


def plain(s):
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)          # images
    s = re.sub(r"^\s*#+\s*", "", s, flags=re.M)           # heading marks
    s = re.sub(r"^\s*[-*+]\s+", "", s, flags=re.M)        # bullets
    s = re.sub(r"^\|?\s*-{3,}.*$", "", s, flags=re.M)     # table rules
    s = s.replace("|", " | ").replace("*", "").replace("$", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def split_long(p):
    """Split one over-long paragraph on sentence ends (Sinhala '.' / '?')."""
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.?!])\s+", p):
        if cur and len(cur) + len(sent) + 1 > TARGET:
            parts.append(cur)
            cur = sent
        else:
            cur = (cur + " " + sent).strip()
    if cur:
        parts.append(cur)
    out = []                                   # sentences longer than HARD
    for x in parts:
        while len(x) > HARD:
            cut = x.rfind(" ", 0, TARGET)
            cut = cut if cut > TARGET // 2 else TARGET
            out.append(x[:cut])
            x = x[cut:].strip()
        out.append(x)
    return out


def chunk_file(path, subject, book):
    heading = ""
    paras = []                                 # (heading, text)
    for block in re.split(r"\n\s*\n", open(path, encoding="utf-8").read()):
        b = block.strip()
        if not b:
            continue
        if re.match(r"^#{1,6}\s", b) and "\n" not in b:
            h = plain(b)
            if n_si(h) >= 3 and len(h) <= 120:  # a real heading, not OCR junk
                heading = h
                continue
        t = plain(b)
        if not t:
            continue
        for piece in (split_long(t) if len(t) > HARD else [t]):
            paras.append((heading, piece))

    chunks, cur, cur_h = [], [], None
    def flush():
        if cur:
            text = "\n".join(cur)
            if n_si(text) >= MIN_SI:
                chunks.append(dict(subject=subject, book=book, heading=cur_h or "",
                                   text=text))
    for h, t in paras:
        size = sum(len(x) + 1 for x in cur)
        if cur and (size + len(t) > TARGET or h != cur_h and size > TARGET // 2):
            flush()
            cur = []
        if not cur:
            cur_h = h
        cur.append(t)
    flush()
    return chunks


def main(root, out):
    seen, rows = set(), []
    for subject in sorted(os.listdir(root)):
        d = os.path.join(root, subject)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".md"):
                continue
            book = fn[:-3]
            for c in chunk_file(os.path.join(d, fn), subject, book):
                key = c["text"]
                if key in seen:                 # repeated running headers etc.
                    continue
                seen.add(key)
                c["id"] = len(rows)
                rows.append(c)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    L = sorted(len(r["text"]) for r in rows)
    print(f"{len(rows)} chunks from {len({(r['subject'], r['book']) for r in rows})} books; "
          f"chars p50 {L[len(L)//2]} p95 {L[int(len(L)*.95)]} max {L[-1]}; "
          f"total {sum(L)/1e6:.1f}M chars")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
