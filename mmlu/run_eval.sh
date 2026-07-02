#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate all SinLlama / Llama models on SinhalaMMLU (3-shot) and, when the
# evaluation finishes successfully, upload the markdown reports to GCS.
#
# Re-runnable: models that already have a <model>_metrics.json in the output
# directory are reused (--skip-existing), so only new models are evaluated.
#
# Usage:  bash run_eval.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR="/workspace/sinllama-continual-pretraining"
EVAL_DIR="$PROJECT_DIR/mmlu_eval"
DATA_ROOT="$PROJECT_DIR/SinhalaMMLU"
OUT_DIR="$EVAL_DIR/results"
BUCKET="gs://sinllama-cpt/mmlu_results_bf16"     # markdown reports land here

# Models to evaluate (paths relative to $PROJECT_DIR)
MODELS=(
  "$PROJECT_DIR/SinLlama_cpt_merged"
  "$PROJECT_DIR/SinLlama_Backtrianx_instruct"
  "$PROJECT_DIR/SinLlama_v01"
  "$PROJECT_DIR/llama-3-8b"
)

cd "$EVAL_DIR"
mkdir -p "$OUT_DIR"

# reduce CUDA fragmentation on the 16 GB GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date '+%F %T')] Evaluating ${#MODELS[@]} models on SinhalaMMLU (3-shot, 4-bit NF4)"
python -u evaluate_sinhala_mmlu.py \
  --data-root   "$DATA_ROOT" \
  --prompt-file "$EVAL_DIR/prompt.txt" \
  --models      "${MODELS[@]}" \
  --kshot 3 --batch-size 4 \
  --skip-existing \
  --out-dir "$OUT_DIR"

echo "[$(date '+%F %T')] Evaluation finished. Markdown reports:"
ls -1 "$OUT_DIR"/*.md

echo "[$(date '+%F %T')] Uploading markdown reports to $BUCKET/"
gsutil -m cp "$OUT_DIR"/*.md "$BUCKET/"

echo "[$(date '+%F %T')] Upload complete. Bucket contents:"
gsutil ls "$BUCKET/"
echo "[$(date '+%F %T')] ALL DONE"
