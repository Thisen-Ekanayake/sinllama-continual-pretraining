"""Extract every in-scope arts textbook to textbooks/markdown/<subject>/grade_<N>.md.

Books are processed in parallel. Afterwards, paragraphs that occur in 3+ different
books (national anthem, minister's message, preface, free-distribution notice,
editorial board) are removed as boilerplate. Per-book stats go to _report.json.

usage: python textbooks/extract_arts.py [--jobs 12] [--only history]
"""
import argparse
import json
import re
from collections import Counter
from multiprocessing import Pool

import extract_textbook as et
from arts_books import OUT_ROOT, books

BOILERPLATE_MIN_BOOKS = 3


def run(job):
    grade, subject, pdfs = job
    et.unknown_glyphs.clear()
    et.page_errors.clear()
    text = et.extract_book(pdfs)
    return grade, subject, [str(p) for p in pdfs], text, dict(et.unknown_glyphs), list(et.page_errors)


def norm(block):
    return re.sub(r"\W+", "", block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--only", nargs="*", help="subject slugs to (re)extract")
    args = ap.parse_args()
    jobs = [b for b in books() if not args.only or b[1] in args.only]
    jobs.sort(key=lambda b: -sum(p.stat().st_size for p in b[2]))  # big books first

    with Pool(args.jobs) as pool:
        results = []
        for r in pool.imap_unordered(run, jobs):
            print(f"done grade {r[0]:>2} {r[1]}", flush=True)
            results.append(r)

    # boilerplate = the same paragraph in several different books
    seen = Counter()
    for _, _, _, text, _, _ in results:
        for key in {norm(b) for b in text.split("\n\n") if len(norm(b)) > 15}:
            seen[key] += 1
    if args.only:  # partial run: also count paragraphs from books already on disk
        for md in OUT_ROOT.glob("*/grade_*.md"):
            if md.parent.name not in args.only:
                for key in {norm(b) for b in md.read_text().split("\n\n") if len(norm(b)) > 15}:
                    seen[key] += 1

    report_path = OUT_ROOT / "_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() and args.only else {}
    for grade, subject, pdfs, text, unknown, errors in sorted(results):
        blocks = text.split("\n\n")
        kept = [b for b in blocks if seen[norm(b)] < BOILERPLATE_MIN_BOOKS or len(norm(b)) <= 15]
        dst = OUT_ROOT / subject / f"grade_{grade}.md"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n\n".join(kept).strip() + "\n", encoding="utf-8")
        report[f"{subject}/grade_{grade}"] = {
            "pdfs": pdfs, "chars": len("\n\n".join(kept)), "blocks": len(kept),
            "boilerplate_dropped": len(blocks) - len(kept), "unmapped_glyphs": unknown, "page_errors": errors}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True))
    print(f"wrote {len(results)} books under {OUT_ROOT}")


if __name__ == "__main__":
    main()
