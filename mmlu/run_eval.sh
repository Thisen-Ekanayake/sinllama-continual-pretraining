#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate all SinLlama / Llama models on SinhalaMMLU (3-shot, bf16) and write
# the detailed markdown/JSON reports. When it finishes, optionally upload the
# markdown to GCS.
#
# The Bactrian-X instruct model is scored in the Alpaca
# "### Instruction / ### Input / ### Response" template it was SFT'd on
# (--alpaca-models); the base / CPT models use the raw SinhalaMMLU template.
#
# Target hardware: a single AMD MI300X (192 GB, ROCm) — hence the large batch
# size and no quantization.
#
# Usage:  bash run_eval.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve paths from this script's own location (<repo>/mmlu/run_eval.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$PROJECT_DIR/SinhalaMMLU"
OUT_DIR="$SCRIPT_DIR/results"

# Models to evaluate: base -> cpt -> instruct, plus the original Llama-3 ref.
MODELS=(
  "$PROJECT_DIR/llama-3-8b"
  "$PROJECT_DIR/SinLlama_v01"
  "$PROJECT_DIR/SinLlama_cpt_merged"
  "$PROJECT_DIR/SinLlama_Backtrianx_instruct"
)

# Model-name substring(s) to score in the native Alpaca SFT format.
ALPACA_MODELS="SinLlama_Backtrianx_instruct"

# GCS bucket for the markdown reports (leave empty to skip the upload).
BUCKET="gs://sinllama-cpt/mmlu_results_bf16"

mkdir -p "$OUT_DIR"

# MI300X (ROCm) allocator: reduce fragmentation on the long Sinhala sequences.
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date '+%F %T')] Evaluating ${#MODELS[@]} models on SinhalaMMLU (3-shot, bf16, MI300X)"
python -u "$SCRIPT_DIR/evaluate_sinhala_mmlu.py" \
  --data-root     "$DATA_ROOT" \
  --prompt-file   "$SCRIPT_DIR/prompt.txt" \
  --models        "${MODELS[@]}" \
  --alpaca-models "$ALPACA_MODELS" \
  --kshot 3 --batch-size 32 --max-len 4096 \
  --out-dir "$OUT_DIR"

echo "[$(date '+%F %T')] Evaluation finished. Reports:"
ls -1 "$OUT_DIR"/*.md

# Optional: upload the markdown reports to GCS if gsutil is configured.
if [[ -n "$BUCKET" ]] && command -v gsutil >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] Uploading markdown reports to $BUCKET/"
  gsutil -m cp "$OUT_DIR"/*.md "$BUCKET/"
  gsutil ls "$BUCKET/"
else
  echo "[$(date '+%F %T')] Skipping GCS upload (gsutil not found or BUCKET empty)."
fi

echo "[$(date '+%F %T')] ALL DONE"
