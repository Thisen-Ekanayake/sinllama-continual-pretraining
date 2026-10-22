"""Extract the technical textbooks (maths, science, ICT, technology ...) to
textbooks/markdown/<subject>/grade_<N>.md, with equations in LaTeX.

  rules  (CPU, parallel)  rule-based Markdown per page, superscripts ^{..} and
                          subscripts _{..} marked, plus a math score per page
  vlm    (GPU, resumable) pages with fraction bars or 3+ math lines are re-typeset
                          by the local vision model (vlm_page.py) from the page
                          image + the rule text; one cached JSON line per page
  write                   VLM output where it keeps the Sinhala text, rule output
                          otherwise; cross-book boilerplate removed

usage: python textbooks/extract_technical.py rules|vlm|write|all [--only mathematics]
"""
import argparse
import json
import time
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pdfplumber

import extract_textbook as et
from arts_books import OUT_ROOT, books

CACHE = Path(__file__).parent / "cache" / "technical"
MIN_COVERAGE = 0.85  # share of the page's Sinhala words the VLM output must keep
MAX_NEW = 0.15       # share of VLM Sinhala words allowed to be absent from the text layer


def key(grade, subject):
    return f"{subject}__grade_{grade}"


# ------------------------------------------------------------------ rules pass

def rules_book(job):
    grade, subject, pdfs = job
    et.MATH_SCRIPTS = True
    et.page_errors.clear()
    book = et.Book(et.body_size(pdfs), min_size=0.4)  # keep exponents and indices
    pages = []
    for path, number, page, blocks in et.extract_pages(pdfs, book):
        frac, math_lines = et.math_score(page, blocks)
        pages.append({"pdf": str(path), "page": number, "blocks": blocks,
                      "frac": frac, "math_lines": math_lines,
                      "vlm": frac > 0 or math_lines >= 3})
    out = {"grade": grade, "subject": subject, "pages": pages, "errors": list(et.page_errors)}
    (CACHE / f"{key(grade, subject)}.rules.json").write_text(json.dumps(out, ensure_ascii=False))
    return key(grade, subject), len(pages), sum(p["vlm"] for p in pages)


# -------------------------------------------------------------------- vlm pass

VLM_SUBJECTS = ("mathematics", "science")


def needs_vlm(subject, page):
    """Only stacked fractions in maths / science need the vision model: rules already
    keep exponents and plain equations, and 'fraction bars' in other subjects are
    ledger underlines and table rules."""
    return subject.split("_part_")[0] in VLM_SUBJECTS and page["frac"] > 0

def vlm_all(jobs):
    from vlm_page import vlm_page
    todo = []
    for grade, subject, _ in jobs:
        rules = json.loads((CACHE / f"{key(grade, subject)}.rules.json").read_text())
        done_path = CACHE / f"{key(grade, subject)}.vlm.jsonl"
        done = set()
        if done_path.exists():
            done = {(d["pdf"], d["page"]) for d in map(json.loads, done_path.open())}
        todo += [(done_path, p) for p in rules["pages"]
                 if needs_vlm(subject, p) and (p["pdf"], p["page"]) not in done]
    print(f"{len(todo)} pages to re-typeset", flush=True)
    for i, (done_path, p) in enumerate(todo, 1):
        md, finish = None, None
        for attempt in range(60):  # the server may be restarting after a crash
            try:
                md, finish = vlm_page(p["pdf"], p["page"], "\n\n".join(p["blocks"]))
                break
            except Exception as e:
                if getattr(e, "code", None) == 400:  # e.g. page too long for the context
                    md, finish = None, "rejected: 400"
                    print(f"  ! {p['pdf']} p{p['page']}: rejected (too long), using rule text", flush=True)
                    break
                print(f"  ! {p['pdf']} p{p['page']}: {e}; waiting for server", flush=True)
                time.sleep(30)
        if finish is None:
            continue  # leave the page unrecorded; a rerun retries it
        with done_path.open("a") as f:
            f.write(json.dumps({"pdf": p["pdf"], "page": p["page"], "md": md,
                                "finish": finish}, ensure_ascii=False) + "\n")
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}", flush=True)


# ------------------------------------------------------------------ write pass

SINHALA_WORD = re.compile(r"[඀-෿‍]+")


def sinhala_words(text):
    return Counter(SINHALA_WORD.findall(text))


def accept_vlm(rule_text, md, finish):
    """Keep the VLM page only if it reproduces the text layer's Sinhala and does not
    invent Sinhala of its own (small VLMs misread the script)."""
    if not md or finish != "stop":
        return False
    src, out = sinhala_words(rule_text), sinhala_words(md)
    if not src:
        return True
    kept = sum(min(n, out[w]) for w, n in src.items()) / sum(src.values())
    new = sum(n for w, n in out.items() if w not in src) / max(1, sum(out.values()))
    return kept >= MIN_COVERAGE and new <= MAX_NEW


def vlm_blocks(md):
    md = re.sub(r"^#{1,6} ", "## ", md, flags=re.M)  # page-level headings are sections
    return [b.strip() for b in re.split(r"\n\s*\n", md) if b.strip()]


def write_books(jobs):
    texts, stats = {}, {}
    for grade, subject, _ in jobs:
        k = key(grade, subject)
        rules = json.loads((CACHE / f"{k}.rules.json").read_text())
        vlm = {}
        vp = CACHE / f"{k}.vlm.jsonl"
        if vp.exists():
            vlm = {(d["pdf"], d["page"]): d for d in map(json.loads, vp.open())}
        blocks, used, rejected = [], 0, 0
        for p in rules["pages"]:
            d = vlm.get((p["pdf"], p["page"]))
            if d and accept_vlm("\n\n".join(p["blocks"]), d["md"], d["finish"]):
                blocks += vlm_blocks(d["md"])
                used += 1
            else:
                rejected += bool(d)
                blocks += p["blocks"]
        texts[(grade, subject)] = et.assemble(blocks)
        stats[(grade, subject)] = {"pages": len(rules["pages"]),
                                   "vlm_pages": sum(p["vlm"] for p in rules["pages"]),
                                   "vlm_used": used, "vlm_rejected": rejected,
                                   "page_errors": rules["errors"]}

    # paragraphs repeated in 3+ books are boilerplate (same rule as the arts books)
    norm = lambda b: re.sub(r"\W+", "", b)
    seen = Counter()
    for text in texts.values():
        for n in {norm(b) for b in text.split("\n\n") if len(norm(b)) > 15}:
            seen[n] += 1
    report_path = OUT_ROOT / "_report_technical.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for (grade, subject), text in sorted(texts.items()):
        kept = [b for b in text.split("\n\n") if seen[norm(b)] < 3 or len(norm(b)) <= 15]
        base, _, part = subject.partition("_part_")
        dst = OUT_ROOT / base / (f"grade_{grade}" + (f"_part_{part}" if part else "") + ".md")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n\n".join(kept).strip() + "\n", encoding="utf-8")
        report[str(dst.relative_to(OUT_ROOT))] = stats[(grade, subject)] | {
            "chars": len("\n\n".join(kept))}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True))
    print(f"wrote {len(texts)} books")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rules", "vlm", "write", "all"])
    ap.add_argument("--only", nargs="*", help="subject slugs (e.g. mathematics_part_1)")
    ap.add_argument("--jobs", type=int, default=14)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [b for b in books("technical") if not args.only or b[1] in args.only
            or b[1].split("_part_")[0] in args.only]
    if args.stage in ("rules", "all"):
        with Pool(args.jobs) as pool:
            for k, n, v in pool.imap_unordered(rules_book, jobs):
                print(f"rules {k}: {n} pages, {v} for VLM", flush=True)
    if args.stage in ("vlm", "all"):
        vlm_all(jobs)
    if args.stage in ("write", "all"):
        write_books(jobs)


if __name__ == "__main__":
    sys.exit(main())
