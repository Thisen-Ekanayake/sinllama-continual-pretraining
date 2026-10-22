"""Politely download Sinhala-medium textbooks from edupub.gov.lk.

Site flow (all POST forms, no API):
  SelectSyllabuss.php  (BookLanguage, BookGrade)        -> list of books (bookId)
  SelectChapter.php    (bookId, BookGrade, ...)          -> links to the PDF(s)
  Administrator/Sinhala/<grade>/<book>/<file>.pdf        -> the PDF

Politeness: one request at a time, a fixed delay (+ jitter) between every request,
exponential backoff on errors / 429 / 5xx, resumable (finished files and the book
index are cached, so a rerun only fetches what is missing).

usage:
  python textbooks/download_textbooks.py --dry-run --grades 10   # list only
  python textbooks/download_textbooks.py                          # all grades
"""
import argparse
import html
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import quote, urljoin

import requests

BASE = "http://www.edupub.gov.lk/"
SINHALA = "2"
GRADES = {str(g): str(g) for g in range(1, 12)} | {"12-13": "17"}  # label -> form value
UA = "NLP"


class Polite:
    def __init__(self, delay):
        self.delay, self.last = delay, 0.0
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA

    def request(self, method, url, **kw):
        for attempt in range(6):
            wait = self.delay + random.uniform(0, self.delay / 2) - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                r = self.s.request(method, url, timeout=(30, 300), **kw)
                if r.status_code < 400:
                    return r
                if r.status_code not in (429, 500, 502, 503, 504):
                    r.raise_for_status()
                retry_after = int(r.headers.get("Retry-After", 0) or 0)
            except requests.ConnectionError:
                retry_after = 0
            except requests.Timeout:
                retry_after = 0
            backoff = max(retry_after, 30 * 2 ** attempt)
            print(f"  ! {method} {url} failed (attempt {attempt + 1}), sleeping {backoff}s")
            time.sleep(backoff)
        raise RuntimeError(f"giving up on {url}")

    def download(self, url, dst, attempts=6):
        """Stream url to dst; a stall mid-body is retried, resuming from the partial file."""
        tmp = dst.with_suffix(".part")
        dst.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(attempts):
            have = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            try:
                r = self.request("GET", url, stream=True, headers=headers)
                if have and r.status_code != 206:  # server ignored Range: start over
                    have = 0
                with open(tmp, "ab" if have else "wb") as f:
                    copy_with_watchdog(r, f, have)
                total = r.headers.get("Content-Range", "").rpartition("/")[2] \
                    or r.headers.get("Content-Length")
                if total and total.isdigit() and tmp.stat().st_size != int(total):
                    raise requests.ConnectionError("short read")
                tmp.rename(dst)  # only complete files count as done
                return
            except (requests.ConnectionError, requests.Timeout) as e:
                backoff = 30 * 2 ** attempt
                print(f"  ! download stalled ({e.__class__.__name__}), "
                      f"{tmp.stat().st_size if tmp.exists() else 0} bytes kept, sleeping {backoff}s")
                time.sleep(backoff)
        raise RuntimeError(f"giving up on {url}")


STALL_WINDOW = 120       # seconds
MIN_BYTES_PER_WINDOW = 100_000  # below this the server is trickling: drop and resume


def copy_with_watchdog(r, f, have):
    """Copy the body, showing progress; abort if throughput collapses.

    A socket read timeout never fires when the server sends a few bytes now and
    then, so the run would hang forever - check real throughput instead.
    """
    total = int(r.headers.get("Content-Length", 0) or 0) + have
    got, start = have, time.time()
    mark_t, mark_b, shown = start, got, 0.0
    for chunk in r.iter_content(8192):
        f.write(chunk)
        got += len(chunk)
        now = time.time()
        if now - shown > 2:
            pct = f"{100 * got / total:5.1f}%" if total else ""
            rate = (got - have) / max(now - start, 1e-6) / 1024
            print(f"\r     {got / 2**20:7.1f} MB {pct} {rate:7.1f} KB/s", end="", flush=True)
            shown = now
        if now - mark_t > STALL_WINDOW:
            if got - mark_b < MIN_BYTES_PER_WINDOW:
                print()
                raise requests.ConnectionError(f"only {got - mark_b} bytes in {STALL_WINDOW}s")
            mark_t, mark_b = now, got
    print()


def list_books(web, grade_value):
    r = web.request("POST", BASE + "SelectSyllabuss.php",
                    files={"BookLanguage": (None, SINHALA), "BookGrade": (None, grade_value)})
    r.encoding = "utf-8"
    return [(bid, html.unescape(name).strip()) for name, bid in re.findall(
        r'class="SelectSyllabuss" bookName="([^"]*)" bookId="(\d+)"', r.text)]


def list_pdfs(web, book_id, book_name, grade_value):
    r = web.request("POST", BASE + "SelectChapter.php", data={
        "bookId": book_id, "BookGrade": grade_value,
        "BookLanguageView": " Sinhala", "bookName": book_name})
    r.encoding = "utf-8"
    return [(html.unescape(href), html.unescape(text).strip()) for href, text in re.findall(
        r'href="([^"]+\.pdf)"[^>]*class="SelectChapter"[^>]*>([^<]*)', r.text, re.I)]


def safe(name):
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip() or "untitled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="textbooks/pdfs")
    ap.add_argument("--grades", nargs="*", default=list(GRADES), help="e.g. 6 10 12-13")
    ap.add_argument("--delay", type=float, default=5.0, help="seconds between requests")
    ap.add_argument("--dry-run", action="store_true", help="list PDFs, download nothing")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    web = Polite(args.delay)
    failed = []

    for grade in args.grades:
        gv = GRADES[grade]
        if grade not in index:
            print(f"listing grade {grade} ...", flush=True)  # cache the (slow to crawl) book/PDF listing per grade
            books = []
            for bid, name in list_books(web, gv):
                pdfs = list_pdfs(web, bid, name, gv)
                print(f"grade {grade} | {name} | {len(pdfs)} pdf(s)")
                books.append({"bookId": bid, "name": name,
                              "pdfs": [{"href": h, "title": t} for h, t in pdfs]})
            index[grade] = books
            index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1))
        if args.dry_run:
            n = sum(len(b["pdfs"]) for b in index[grade])
            print(f"grade {grade}: {len(index[grade])} books, {n} pdfs")
            continue
        for book in index[grade]:
            for p in book["pdfs"]:
                dst = out / f"grade_{grade}" / safe(book["name"]) / safe(Path(p["href"]).name)
                if dst.exists():
                    continue
                url = urljoin(BASE, quote(p["href"], safe="/:"))
                print(f"  -> {dst}", flush=True)
                try:
                    web.download(url, dst)
                except RuntimeError as e:  # skip this file, keep going; rerun retries it
                    print(f"  ! {e}")
                    failed.append(url)
    if failed:
        print(f"{len(failed)} file(s) failed; rerun to retry:", *failed, sep="\n  ")


if __name__ == "__main__":
    main()
