#!/usr/bin/env bash
# Publish the stage-3 wiki-SFT artifacts to GCS.
#
#   bash wikipedia_sft/upload_gcs.sh dataset      # train/eval parquet + manifest -> Wikipedia_SFT_Sinhala/
#   bash wikipedia_sft/upload_gcs.sh merge        # LoRA + uc_gen_bilingual -> models/SinLlama_uc_wiki
#   bash wikipedia_sft/upload_gcs.sh adapters     # LoRA adapter  -> SinLlama_uc_wiki_adapters/
#   bash wikipedia_sft/upload_gcs.sh merged       # merged model  -> SinLlama_uc_wiki/
#   bash wikipedia_sft/upload_gcs.sh post-train   # merge, then upload adapters + merged
#   bash wikipedia_sft/upload_gcs.sh all          # dataset + post-train
#
#   DRY_RUN=1 bash wikipedia_sft/upload_gcs.sh all
#   BUCKET=gs://other bash wikipedia_sft/upload_gcs.sh ...
#
# merge/adapters/merged/post-train are delegated to sft/upload_gcs.sh with the
# artifact names overridden, same pattern as gen/upload_gcs.sh -- that script
# is already config-driven, so the only thing this stage needs of its own is
# `dataset`: the wiki mix ships a train/eval parquet pair plus manifest.json,
# not sft/'s *_clean.parquet pair.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config.yaml}"
BUCKET="${BUCKET:-gs://sinllama_cpt}"
PYTHON="${PYTHON:-python3}"

export CONFIG BUCKET PYTHON
export DATASET_NAME="${DATASET_NAME:-Wikipedia_SFT_Sinhala}"
export ADAPTER_NAME="${ADAPTER_NAME:-SinLlama_uc_wiki_adapters}"
export MERGED_NAME="${MERGED_NAME:-SinLlama_uc_wiki}"
export MERGED_DIR="${MERGED_DIR:-${REPO_ROOT}/models/${MERGED_NAME}}"

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] ERROR: $*" >&2; exit 1; }

find_gsutil() {
  if command -v gsutil >/dev/null 2>&1; then command -v gsutil; return; fi
  for c in "$HOME/google-cloud-sdk/bin/gsutil" /usr/lib/google-cloud-sdk/bin/gsutil \
           /snap/bin/gsutil /usr/local/bin/gsutil /opt/google-cloud-sdk/bin/gsutil; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
}

do_dataset() {
  local out_dir
  out_dir="$("${PYTHON}" - "${CONFIG}" <<'PY'
import sys, yaml, pathlib
p = yaml.safe_load(open(sys.argv[1]))["wiki_source"]["out_dir"]
print(p if pathlib.Path(p).is_absolute() else pathlib.Path(sys.argv[1]).resolve().parents[1] / p)
PY
)"
  local files=(train_wiki.parquet eval_wiki.parquet manifest.json)
  for f in "${files[@]}"; do
    [ -s "${out_dir}/${f}" ] || die "missing or empty: ${out_dir}/${f} -- run wikipedia_sft/build_wiki_sft_dataset.py first"
  done

  local gsutil dest="${BUCKET}/${DATASET_NAME}"
  gsutil="$(find_gsutil)"
  [ -n "${gsutil}" ] || die "gsutil not found"
  log "uploading the wiki SFT set -> ${dest}"
  for f in "${files[@]}"; do
    log "  ${f}  $(du -h "${out_dir}/${f}" | cut -f1)"
  done
  if [ -n "${DRY_RUN:-}" ]; then
    echo "  DRY_RUN: ${gsutil} -m cp ${files[*]} ${dest}/"
    return 0
  fi
  ( cd "${out_dir}" && "${gsutil}" -m cp "${files[@]}" "${dest}/" )
  log "  done"
}

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

case "${1:-}" in
  dataset)    do_dataset ;;
  all)        do_dataset; bash "${REPO_ROOT}/sft/upload_gcs.sh" post-train ;;
  merge|adapters|merged|post-train)
              bash "${REPO_ROOT}/sft/upload_gcs.sh" "$1" ;;
  *)          usage ;;
esac
