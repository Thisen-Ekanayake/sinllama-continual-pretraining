#!/usr/bin/env bash
# Provision the raw Sinhala Wikipedia corpus: download the dump, run
# WikiExtractor, and reduce it to the same short_articles.json /
# medium_articles.json pair that Wikipedia_Dataset/ already carries on this
# dev box. A fresh MI300X pod checkout has none of this (the dump, the
# extracted JSON and the article buckets are all *.xml / *.json and are
# gitignored -- see .gitignore), so wikipedia_sft/run_wiki_sft.sh calls this
# unconditionally and every stage below skips itself when its output already
# exists.
#
#   bash wikipedia_sft/fetch_wikipedia_dump.sh
#   FORCE=1 bash wikipedia_sft/fetch_wikipedia_dump.sh   # re-run every stage
#   WIKI_DIR=/other/path bash wikipedia_sft/fetch_wikipedia_dump.sh
#
# Pipeline (mirrors the one already baked into Wikipedia_Dataset/, per
# Wikipedia_Dataset/wikiextractor/extract.sh, categorize_articles.py and
# split_by_category.py):
#
#   1. download   siwiki-latest-pages-articles.xml.bz2 from dumps.wikimedia.org
#   2. decompress -> siwiki-latest-pages-articles.xml
#   3. extract    WikiExtractor --json               -> extracted/AA/wiki_NN (NDJSON)
#   4. convert    NDJSON -> indented JSON array       -> extracted_json/AA/wiki_NN.json
#   5. categorize word-count bucket (stub/short/medium/long) -> extracted_categorized/
#   6. bucket     collect the short + medium buckets  -> short_articles.json, medium_articles.json
#
# Only steps 5-6 are opinionated (the 150/500/2000-word boundaries are
# Wikipedia_Dataset/categorize_articles.py's, kept identical here so this
# script reproduces that folder rather than inventing a new convention).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"

WIKI_DIR="${WIKI_DIR:-${REPO_ROOT}/Wikipedia_Dataset}"
DUMP_URL="${DUMP_URL:-https://dumps.wikimedia.org/siwiki/latest/siwiki-latest-pages-articles.xml.bz2}"
DUMP_BZ2="${WIKI_DIR}/siwiki-latest-pages-articles.xml.bz2"
DUMP_XML="${WIKI_DIR}/siwiki-latest-pages-articles.xml"
EXTRACTED_DIR="${WIKI_DIR}/extracted"
EXTRACTED_JSON_DIR="${WIKI_DIR}/extracted_json"
CATEGORIZED_DIR="${WIKI_DIR}/extracted_categorized"
WIKIEXTRACTOR_DIR="${WIKI_DIR}/wikiextractor"
PROCESSES="${PROCESSES:-$(nproc)}"

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] ERROR: $*" >&2; exit 1; }

mkdir -p "${WIKI_DIR}"

# -- 1-2. download + decompress ----------------------------------------------

if [ -s "${DUMP_XML}" ] && [ -z "${FORCE:-}" ]; then
  log "dump already present: ${DUMP_XML} ($(du -h "${DUMP_XML}" | cut -f1))"
else
  if [ -s "${DUMP_BZ2}" ] && [ -z "${FORCE:-}" ]; then
    log "compressed dump already present: ${DUMP_BZ2}"
  else
    command -v curl >/dev/null 2>&1 || die "curl not found"
    log "downloading ${DUMP_URL} -> ${DUMP_BZ2}"
    curl -fSL --retry 3 -C - -o "${DUMP_BZ2}" "${DUMP_URL}"
  fi
  command -v bunzip2 >/dev/null 2>&1 || die "bunzip2 not found (install bzip2)"
  log "decompressing -> ${DUMP_XML}"
  bunzip2 -k -f "${DUMP_BZ2}"
fi
[ -s "${DUMP_XML}" ] || die "expected a dump at ${DUMP_XML} after download/decompress"

# -- 3. WikiExtractor ---------------------------------------------------------

if [ -n "$(find "${EXTRACTED_DIR}" -mindepth 2 -type f -print -quit 2>/dev/null)" ] && [ -z "${FORCE:-}" ]; then
  log "extracted/ already populated: ${EXTRACTED_DIR}"
else
  # Prefer the vendored copy already checked out under Wikipedia_Dataset/
  # (attardi/wikiextractor); fall back to the PyPI package on a fresh pod.
  if [ -f "${WIKIEXTRACTOR_DIR}/wikiextractor/WikiExtractor.py" ]; then
    log "using vendored wikiextractor at ${WIKIEXTRACTOR_DIR}"
    RUN_EXTRACTOR=("${PYTHON}" -m wikiextractor.WikiExtractor)
    export PYTHONPATH="${WIKIEXTRACTOR_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  else
    if ! "${PYTHON}" -c "import wikiextractor" >/dev/null 2>&1; then
      log "installing wikiextractor from PyPI"
      "${PYTHON}" -m pip install --quiet wikiextractor
    fi
    RUN_EXTRACTOR=("${PYTHON}" -m wikiextractor.WikiExtractor)
  fi

  rm -rf "${EXTRACTED_DIR}"
  log "running WikiExtractor (processes=${PROCESSES}) -> ${EXTRACTED_DIR}"
  "${RUN_EXTRACTOR[@]}" "${DUMP_XML}" \
      --json \
      --processes "${PROCESSES}" \
      --templates "${WIKI_DIR}/templates.txt" \
      --output "${EXTRACTED_DIR}" \
      --bytes 1M \
      --links \
      --sections \
      --lists \
      --keep_tables \
      --min_text_length 0 \
      --filter_disambig_pages
fi

# -- 4. NDJSON -> indented JSON array -----------------------------------------

if [ -n "$(find "${EXTRACTED_JSON_DIR}" -mindepth 2 -type f -print -quit 2>/dev/null)" ] && [ -z "${FORCE:-}" ]; then
  log "extracted_json/ already populated: ${EXTRACTED_JSON_DIR}"
else
  log "converting NDJSON -> JSON arrays -> ${EXTRACTED_JSON_DIR}"
  SRC="${EXTRACTED_DIR}" DST="${EXTRACTED_JSON_DIR}" "${PYTHON}" - <<'PY'
import json, os

src, dst = os.environ["SRC"], os.environ["DST"]
n_files = n_articles = 0
for root, _dirs, files in os.walk(src):
    rel = os.path.relpath(root, src)
    for fname in files:
        articles = [json.loads(line) for line in open(os.path.join(root, fname), encoding="utf-8") if line.strip()]
        dst_dir = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(dst_dir, exist_ok=True)
        with open(os.path.join(dst_dir, f"{fname}.json"), "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        n_files += 1
        n_articles += len(articles)
print(f"  {n_files} files, {n_articles} articles")
PY
fi

# -- 5. categorize by word-count bucket ---------------------------------------

if [ -n "$(find "${CATEGORIZED_DIR}" -mindepth 2 -type f -print -quit 2>/dev/null)" ] && [ -z "${FORCE:-}" ]; then
  log "extracted_categorized/ already populated: ${CATEGORIZED_DIR}"
else
  log "categorizing (stub<150<short<500<medium<2000<long words) -> ${CATEGORIZED_DIR}"
  SRC="${EXTRACTED_JSON_DIR}" DST="${CATEGORIZED_DIR}" "${PYTHON}" - <<'PY'
import json, os

src, dst = os.environ["SRC"], os.environ["DST"]
bounds = [(150, "stub"), (500, "short"), (2000, "medium")]

def categorize(n):
    for bound, label in bounds:
        if n < bound:
            return label
    return "long"

totals, n_files, n_articles = {}, 0, 0
for root, _dirs, files in os.walk(src):
    rel = os.path.relpath(root, src)
    for fname in files:
        if not fname.endswith(".json"):
            continue
        articles = json.load(open(os.path.join(root, fname), encoding="utf-8"))
        for a in articles:
            title_wc = len(a.get("title", "").split())
            text_wc = len(a.get("text", "").split())
            a["title_word_count"] = title_wc
            a["text_word_count"] = text_wc
            a["category"] = categorize(text_wc)
            totals[a["category"]] = totals.get(a["category"], 0) + 1
        dst_dir = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(dst_dir, exist_ok=True)
        with open(os.path.join(dst_dir, fname), "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        n_files += 1
        n_articles += len(articles)
print(f"  {n_files} files, {n_articles} articles")
for label, count in sorted(totals.items(), key=lambda kv: -kv[1]):
    print(f"    {label}: {count}")
PY
fi

# -- 6. bucket short + medium into the two flat files -------------------------

SHORT_JSON="${WIKI_DIR}/short_articles.json"
MEDIUM_JSON="${WIKI_DIR}/medium_articles.json"
if [ -s "${SHORT_JSON}" ] && [ -s "${MEDIUM_JSON}" ] && [ -z "${FORCE:-}" ]; then
  log "short_articles.json / medium_articles.json already present"
else
  log "collecting short + medium buckets -> ${SHORT_JSON}, ${MEDIUM_JSON}"
  SRC="${CATEGORIZED_DIR}" SHORT="${SHORT_JSON}" MEDIUM="${MEDIUM_JSON}" "${PYTHON}" - <<'PY'
import json, os

src = os.environ["SRC"]
out = {"short": os.environ["SHORT"], "medium": os.environ["MEDIUM"]}
buckets = {"short": [], "medium": []}
for root, _dirs, files in os.walk(src):
    for fname in files:
        if not fname.endswith(".json"):
            continue
        for a in json.load(open(os.path.join(root, fname), encoding="utf-8")):
            if a.get("category") in buckets:
                buckets[a["category"]].append(a)
for cat, path in out.items():
    with open(path, "w", encoding="utf-8") as f:
        json.dump(buckets[cat], f, ensure_ascii=False, indent=2)
    print(f"  {cat}: {len(buckets[cat])} articles -> {path}")
PY
fi

log "Done. Next: python wikipedia_sft/build_wiki_sft_dataset.py --config wikipedia_sft/config.yaml"
