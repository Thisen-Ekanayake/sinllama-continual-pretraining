#!/usr/bin/env bash
# Pull the base model for the UltraChat SFT run out of GCS.
#
# Reads `model.path` and `model.gcs_uri` from sft/config.yaml and mirrors the
# bucket directory into the local path with gsutil (same tool and flags as
# benchmark/lib/gcs_upload.sh, opposite direction). No-op when the model is
# already on disk, so run_sft_uc.sh can call it unconditionally.
#
#   bash sft/fetch_model.sh [path/to/config.yaml]
#   FORCE=1 bash sft/fetch_model.sh      # re-sync even if it is already there
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/config.yaml}"
PYTHON="${PYTHON:-python3}"

read -r MODEL_PATH GCS_URI < <("${PYTHON}" - "${CONFIG}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))["model"]
print(cfg["path"], cfg.get("gcs_uri") or "-")
PY
)

# Config paths are relative to the repo root.
case "${MODEL_PATH}" in
  /*) DEST="${MODEL_PATH}" ;;
  *)  DEST="${REPO_ROOT}/${MODEL_PATH}" ;;
esac

if [ -f "${DEST}/config.json" ] && [ -z "${FORCE:-}" ]; then
  echo "[$(date '+%F %T')] model already present: ${DEST} (FORCE=1 to re-sync)"
  exit 0
fi

if [ "${GCS_URI}" = "-" ]; then
  echo "error: ${DEST} is missing and model.gcs_uri is not set in ${CONFIG}" >&2
  exit 1
fi

if ! command -v gsutil >/dev/null 2>&1; then
  echo "error: gsutil not found — install the gcloud SDK or copy the model to ${DEST} by hand" >&2
  exit 1
fi

echo "[$(date '+%F %T')] Downloading ${GCS_URI} -> ${DEST}"
mkdir -p "${DEST}"
gsutil -m rsync -r "${GCS_URI}" "${DEST}"

if [ ! -f "${DEST}/config.json" ]; then
  echo "error: ${DEST}/config.json missing after sync — is ${GCS_URI} the model directory?" >&2
  exit 1
fi
echo "[$(date '+%F %T')] Done: $(du -sh "${DEST}" | cut -f1) in ${DEST}"
