#!/usr/bin/env bash
# Upload the finished gen-stage adapter and merged model to GCS under the
# requested names, after training completes.
#
#   bash gen/upload_gcs_final.sh
#
# Merges the LoRA adapter into SinLlama_uc_instruct_cleaned (if not already
# merged), then uploads:
#   adapter -> gs://sinllama_cpt/SinLlama_UC_Gen_Adapters/
#   merged  -> gs://sinllama_cpt/SinLlama_UC_Gen_Model/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CONFIG="${SCRIPT_DIR}/config.yaml"
export BUCKET="${BUCKET:-gs://sinllama_cpt}"
export DATASET_NAME="UltraChat_gen_bilingual"
export ADAPTER_NAME="SinLlama_UC_Gen_Adapters"
export MERGED_NAME="SinLlama_UC_Gen_Model"
export MERGED_DIR="${REPO_ROOT}/models/${MERGED_NAME}"
export PYTHON="${REPO_ROOT}/venv-rocm7/bin/python"

export PATH="/root/google-cloud-sdk/bin:${PATH}"

log() { echo "[$(date '+%F %T')] $*"; }

log "merging adapter into base model"
bash "${REPO_ROOT}/sft/upload_gcs.sh" merge

log "uploading adapter -> ${BUCKET}/${ADAPTER_NAME}"
bash "${REPO_ROOT}/sft/upload_gcs.sh" adapters

log "uploading merged model -> ${BUCKET}/${MERGED_NAME}"
bash "${REPO_ROOT}/sft/upload_gcs.sh" merged

log "done. gs://sinllama_cpt/${ADAPTER_NAME}/ and gs://sinllama_cpt/${MERGED_NAME}/ are live."
