#!/usr/bin/env bash
# Publish the cleaned UltraChat corpus and the instruct model it trained to GCS.
#
#   bash sft/upload_gcs.sh dataset      # cleaned parquets  -> UltraChat_Sinhala_cleaned/
#   bash sft/upload_gcs.sh merge        # LoRA + v02        -> models/SinLlama_uc_instruct_cleaned
#   bash sft/upload_gcs.sh adapters     # LoRA adapter      -> SinLlama_uc_instruct_cleaned_adapters/
#   bash sft/upload_gcs.sh merged       # merged model      -> SinLlama_uc_instruct_cleaned/
#   bash sft/upload_gcs.sh post-train   # merge, then upload adapters + merged
#   bash sft/upload_gcs.sh all          # dataset + post-train
#
#   DRY_RUN=1 bash sft/upload_gcs.sh all          # print what would happen
#   BUCKET=gs://other bash sft/upload_gcs.sh ...  # override the destination
#   WITH_CHECKPOINTS=1 bash sft/upload_gcs.sh adapters   # include checkpoint-*/
#
# Paths come from sft/config.yaml, so this follows the run rather than
# hardcoding it. Provenance for what "cleaned" means: docs/ultrachat-cleaning.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config.yaml}"
BUCKET="${BUCKET:-gs://sinllama_cpt}"
PYTHON="${PYTHON:-python3}"

# Overridable so a later stage can reuse this script rather than fork it —
# gen/upload_gcs.sh sets all three plus CONFIG. Defaults are the stage-1 names.
DATASET_NAME="${DATASET_NAME:-UltraChat_Sinhala_cleaned}"
ADAPTER_NAME="${ADAPTER_NAME:-SinLlama_uc_instruct_cleaned_adapters}"
MERGED_NAME="${MERGED_NAME:-SinLlama_uc_instruct_cleaned}"

MERGED_DIR="${MERGED_DIR:-${REPO_ROOT}/models/${MERGED_NAME}}"

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] ERROR: $*" >&2; exit 1; }

# gcloud's bundled gsutil is not on PATH in a non-interactive shell.
find_gsutil() {
  if command -v gsutil >/dev/null 2>&1; then command -v gsutil; return; fi
  for c in "$HOME/google-cloud-sdk/bin/gsutil" /usr/lib/google-cloud-sdk/bin/gsutil \
           /snap/bin/gsutil /usr/local/bin/gsutil; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
  die "gsutil not found (looked on PATH and in ~/google-cloud-sdk/bin)"
}
GSUTIL="$(find_gsutil)"

cfg() {  # cfg <dotted.key> -- read a value out of config.yaml
  "${PYTHON}" - "${CONFIG}" "$1" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    d = d[k]
print(d)
PY
}

abs() { case "$1" in /*) echo "$1";; *) echo "${REPO_ROOT}/$1";; esac; }

human() { du -sh "$1" 2>/dev/null | cut -f1; }

run() {
  if [ -n "${DRY_RUN:-}" ]; then echo "  DRY_RUN: $*"; else "$@"; fi
}

# Compare local file count against what landed in the bucket.
verify() {
  local dest="$1" expected="$2"
  [ -n "${DRY_RUN:-}" ] && return 0
  local got
  got="$("${GSUTIL}" ls -r "${dest}" 2>/dev/null | grep -cv '/:\?$' || true)"
  if [ "${got}" -lt "${expected}" ]; then
    die "verification failed for ${dest}: expected >= ${expected} objects, found ${got}"
  fi
  log "  verified ${got} objects at ${dest}"
}

upload_dir() {  # upload_dir <local> <dest> [extra rsync args...]
  local src="$1" dest="$2"; shift 2
  [ -d "${src}" ] || die "not a directory: ${src}"
  local n; n="$(find "${src}" -maxdepth 1 -type f | wc -l)"
  [ "${n}" -gt 0 ] || die "no files directly under ${src} -- refusing to upload an empty dir"
  log "uploading ${src} ($(human "${src}")) -> ${dest}"
  run "${GSUTIL}" -m rsync -r "$@" "${src}" "${dest}"
  verify "${dest}" "${n}"
}

# --------------------------------------------------------------------------

do_dataset() {
  local train eval_f
  train="$(abs "$(cfg data.train_file)")"
  eval_f="$(abs "$(cfg data.eval_file)")"
  for f in "${train}" "${eval_f}"; do
    [ -s "${f}" ] || die "missing or empty: ${f} -- run sft/clean_ultrachat.py first"
  done
  case "${train}" in
    *_clean.parquet) ;;
    *) die "config data.train_file is '${train}', which is not a _clean parquet. \
Point sft/config.yaml at the cleaned corpus before publishing it as cleaned." ;;
  esac

  local dest="${BUCKET}/${DATASET_NAME}"
  log "uploading cleaned corpus -> ${dest}"
  log "  $(basename "${train}")  $(human "${train}")"
  log "  $(basename "${eval_f}") $(human "${eval_f}")"
  run "${GSUTIL}" -m cp "${train}" "${eval_f}" "${dest}/"
  # Ship the provenance with the data -- "cleaned" is meaningless without it.
  local doc="${REPO_ROOT}/docs/ultrachat-cleaning.md"
  [ -f "${doc}" ] && run "${GSUTIL}" cp "${doc}" "${dest}/README.md"
  verify "${dest}" 2
}

adapter_dir() {
  local out; out="$(abs "$(cfg train.output_dir)")"
  # trainer.save_model() writes the adapter into output_dir itself; older runs
  # used an early_stop/ subdir. Prefer whichever actually holds an adapter.
  if [ -f "${out}/adapter_model.safetensors" ]; then echo "${out}"
  elif [ -f "${out}/early_stop/adapter_model.safetensors" ]; then echo "${out}/early_stop"
  else die "no adapter_model.safetensors under ${out} (or ${out}/early_stop) -- has training finished?"
  fi
}

do_merge() {
  local adapter base
  adapter="$(adapter_dir)"
  base="$(abs "$(cfg model.path)")"
  [ -f "${base}/config.json" ] || die "base model not found: ${base}"

  if [ -f "${MERGED_DIR}/config.json" ]; then
    log "merged model already present at ${MERGED_DIR} -- skipping merge"
    return 0
  fi
  log "merging ${adapter} into ${base} -> ${MERGED_DIR}"
  run "${PYTHON}" "${REPO_ROOT}/cpt/merge_sinllama_lora_low_mem.py" \
      --base_model "${base}" \
      --lora_model "${adapter}" \
      --output_dir "${MERGED_DIR}"
  [ -n "${DRY_RUN:-}" ] || [ -f "${MERGED_DIR}/config.json" ] \
    || die "merge produced no config.json in ${MERGED_DIR}"
}

do_adapters() {
  local src; src="$(adapter_dir)"
  local args=()
  # Checkpoints are resume state, not the deliverable; ~660 MB each.
  [ -n "${WITH_CHECKPOINTS:-}" ] || args=(-x '^checkpoint-[0-9]+/')
  upload_dir "${src}" "${BUCKET}/${ADAPTER_NAME}" "${args[@]}"
}

do_merged() {
  [ -f "${MERGED_DIR}/config.json" ] || die "no merged model at ${MERGED_DIR} -- run '$0 merge' first"
  [ -f "${MERGED_DIR}/tokenizer_config.json" ] \
    || die "${MERGED_DIR} has no tokenizer_config.json -- the chat template would be lost"
  grep -q chat_template "${MERGED_DIR}/tokenizer_config.json" \
    || log "WARNING: no chat_template in ${MERGED_DIR}/tokenizer_config.json"
  upload_dir "${MERGED_DIR}" "${BUCKET}/${MERGED_NAME}"
}

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

case "${1:-}" in
  dataset)    do_dataset ;;
  merge)      do_merge ;;
  adapters)   do_adapters ;;
  merged)     do_merged ;;
  post-train) do_merge; do_adapters; do_merged ;;
  all)        do_dataset; do_merge; do_adapters; do_merged ;;
  *)          usage ;;
esac

log "done"
