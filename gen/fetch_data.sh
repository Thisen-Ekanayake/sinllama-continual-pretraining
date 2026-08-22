#!/usr/bin/env bash
# Provision the stage-2 `gen` corpora on the pod.
#
#   bash gen/fetch_data.sh                 # fetch whatever is missing
#   FORCE=1 bash gen/fetch_data.sh         # re-pull everything
#   bash gen/fetch_data.sh --english-only  # skip the GCS half
#
# Two sources, because the two halves live in different places:
#
#   Sinhala  gs://sinllama_cpt/UltraChat-Sinhala/final/{train,test}_gen.parquet
#            (782 MB; byte-identical to the copies committed under
#            UltraChat_Sinhala/, so this is a no-op in a full checkout)
#   English  HuggingFaceH4/ultrachat_200k, data/{train,test}_gen-*  (~1.4 GB)
#            train_gen ships as 3 shards and is NOT in the repo -- the Sinhala
#            corpus was translated from it, but only the Sinhala side was kept.
#
# Idempotent: every file already on disk is skipped, so run_gen_sft.sh can call
# this unconditionally.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
BUCKET="${BUCKET:-gs://sinllama_cpt}"
SI_DIR="${REPO_ROOT}/UltraChat_Sinhala"
EN_DIR="${SI_DIR}/english/hf"
HF_REPO="${HF_REPO:-HuggingFaceH4/ultrachat_200k}"

ENGLISH_ONLY=""
SINHALA_ONLY=""
for arg in "$@"; do
  case "$arg" in
    --english-only) ENGLISH_ONLY=1 ;;
    --sinhala-only) SINHALA_ONLY=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] ERROR: $*" >&2; exit 1; }

# gcloud's bundled gsutil is not on PATH in a non-interactive shell, and it has
# been in a different place on every pod so far.
find_gsutil() {
  if command -v gsutil >/dev/null 2>&1; then command -v gsutil; return; fi
  for c in "$HOME/google-cloud-sdk/bin/gsutil" /usr/lib/google-cloud-sdk/bin/gsutil \
           /snap/bin/gsutil /usr/local/bin/gsutil /opt/google-cloud-sdk/bin/gsutil; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
}

# -- Sinhala -----------------------------------------------------------------

if [ -z "${ENGLISH_ONLY}" ]; then
  need=()
  for f in train_gen.parquet test_gen.parquet; do
    if [ -s "${SI_DIR}/${f}" ] && [ -z "${FORCE:-}" ]; then
      log "sinhala ${f} already present ($(du -h "${SI_DIR}/${f}" | cut -f1))"
    else
      need+=("${f}")
    fi
  done
  if [ ${#need[@]} -gt 0 ]; then
    GSUTIL="$(find_gsutil)"
    [ -n "${GSUTIL}" ] || die "gsutil not found — install the gcloud SDK, or copy ${need[*]} into ${SI_DIR} by hand"
    mkdir -p "${SI_DIR}"
    for f in "${need[@]}"; do
      log "downloading ${BUCKET}/UltraChat-Sinhala/final/${f}"
      "${GSUTIL}" -m cp "${BUCKET}/UltraChat-Sinhala/final/${f}" "${SI_DIR}/${f}"
    done
  fi
fi

# -- English -----------------------------------------------------------------

if [ -z "${SINHALA_ONLY}" ]; then
  log "syncing ${HF_REPO} (data/train_gen-*, data/test_gen-*) -> ${EN_DIR}"
  FORCE="${FORCE:-}" HF_REPO="${HF_REPO}" EN_DIR="${EN_DIR}" "${PYTHON}" - <<'PY'
import os, glob, sys
from huggingface_hub import snapshot_download

en_dir, repo, force = os.environ["EN_DIR"], os.environ["HF_REPO"], os.environ["FORCE"]
patterns = ["data/train_gen-*", "data/test_gen-*"]

existing = glob.glob(os.path.join(en_dir, "data", "train_gen-*.parquet"))
if existing and not force:
    print(f"  english train_gen already present: {len(existing)} shard(s)")
    sys.exit(0)

path = snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    allow_patterns=patterns,
    local_dir=en_dir,
    max_workers=4,
)
got = sorted(glob.glob(os.path.join(path, "data", "*_gen-*.parquet")))
if not got:
    sys.exit(f"snapshot_download wrote nothing matching {patterns} into {path}")
for f in got:
    print(f"  {os.path.basename(f)}  {os.path.getsize(f) / 1e6:.0f} MB")
PY
fi

# -- Verify ------------------------------------------------------------------

log "verifying"
# Only check the halves this invocation was actually asked to fetch, so
# --sinhala-only does not fail on an English download it deliberately skipped.
"${PYTHON}" - "${SI_DIR}" "${EN_DIR}" "${ENGLISH_ONLY:-0}" "${SINHALA_ONLY:-0}" <<'PY'
import glob, sys
import pyarrow.parquet as pq

si_dir, en_dir, english_only, sinhala_only = sys.argv[1:5]
targets = []
if english_only != "1":
    targets += [
        ("sinhala train_gen", [f"{si_dir}/train_gen.parquet"]),
        ("sinhala test_gen", [f"{si_dir}/test_gen.parquet"]),
    ]
if sinhala_only != "1":
    targets += [
        ("english train_gen", sorted(glob.glob(f"{en_dir}/data/train_gen-*.parquet"))),
        ("english test_gen", sorted(glob.glob(f"{en_dir}/data/test_gen-*.parquet"))
                             or sorted(glob.glob(f"{si_dir}/english/test_gen-*.parquet"))),
    ]
missing = False
for name, files in targets:
    files = [f for f in files if glob.glob(f)]
    if not files:
        print(f"  MISSING  {name}")
        missing = True
        continue
    rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(f"  ok       {name}: {rows:,} rows in {len(files)} file(s)")
sys.exit(1 if missing else 0)
PY

log "Done. Next: python gen/build_mixed_gen.py --config gen/config.yaml"
