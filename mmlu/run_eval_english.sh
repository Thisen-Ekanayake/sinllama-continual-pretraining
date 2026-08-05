#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate all SinLlama / Llama models on English MMLU (5-shot, bf16) IN PARALLEL
# to check catastrophic forgetting. One process per model, each uploads to
# $BUCKET/<model_name>/ when done, final combine pass.
#
# Env toggles:
#   CANONICAL=1   use the standard Hendrycks header-once format for ALL models
#                 (leaderboard-comparable numbers) -> results_english_canonical/
#                 and gs://.../mmlu_results_english_canonical ; no Alpaca wrap.
#   NUM_GPUS (1)  MAX_PARALLEL (2)  BATCH_SIZE (16)  MAX_LEN (4096)
#
# Usage:  bash run_eval_english.sh                 # per-block format (Alpaca for instruct)
#         CANONICAL=1 bash run_eval_english.sh     # canonical Hendrycks format, all models
# ---------------------------------------------------------------------------
set -uo pipefail          # no -e: one model failing must not kill the rest

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$SCRIPT_DIR/english_mmlu"

# find the Google Cloud SDK if it's installed but not on PATH (enables uploads)
export PATH="$HOME/google-cloud-sdk/bin:$PATH"

MODELS=(
  "$PROJECT_DIR/models/llama-3-8b"
  "$PROJECT_DIR/models/SinLlama_v01"
  "$PROJECT_DIR/models/SinLlama_cpt"
  "$PROJECT_DIR/models/SinLlama_Bactrianx_Instruct"
  "$PROJECT_DIR/models/SinLlama_v02"
)
ALPACA_MODELS="SinLlama_Bactrianx_Instruct"

# --- format / scoring mode (compose CANONICAL + CALIBRATE) ---
STYLE_ARGS=()
TAG="english"
if [[ "${CANONICAL:-0}" == "1" ]]; then
  STYLE_ARGS+=(--canonical); TAG="english_canonical"; ALPACA_MODELS=""
  echo "[$(date '+%F %T')] MODE: canonical Hendrycks format (all models raw)"
else
  echo "[$(date '+%F %T')] MODE: per-block format (Alpaca for instruct)"
fi
if [[ "${CALIBRATE:-0}" == "1" ]]; then
  STYLE_ARGS+=(--calibrate); TAG="${TAG}_cal"
  echo "[$(date '+%F %T')] Contextual calibration: ON"
fi
OUT_DIR="$PROJECT_DIR/benchmark/EnglishMMLU_results_final"
BUCKET="gs://sinllama-cpt/mmlu_results_$TAG"

ALPACA_ARG=()
[[ -n "$ALPACA_MODELS" ]] && ALPACA_ARG=(--alpaca-models "$ALPACA_MODELS")

NUM_GPUS="${NUM_GPUS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LEN="${MAX_LEN:-4096}"

mkdir -p "$OUT_DIR"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date '+%F %T')] Launching ${#MODELS[@]} English-MMLU evals — up to $MAX_PARALLEL at once across $NUM_GPUS GPU(s)"

i=0
for MODEL in "${MODELS[@]}"; do
  name="$(basename "$MODEL")"
  gpu=$(( i % NUM_GPUS ))
  echo "[$(date '+%F %T')] -> launching '$name' on GPU $gpu (log: $OUT_DIR/${name}.log)"
  HIP_VISIBLE_DEVICES="$gpu" CUDA_VISIBLE_DEVICES="$gpu" \
  python -u "$SCRIPT_DIR/evaluate_english_mmlu.py" \
    --data-root "$DATA_ROOT" \
    --models    "$MODEL" \
    "${ALPACA_ARG[@]}" "${STYLE_ARGS[@]}" \
    --kshot 5 --batch-size "$BATCH_SIZE" --max-len "$MAX_LEN" \
    --out-dir "$OUT_DIR" --bucket "$BUCKET" --skip-existing \
    > "$OUT_DIR/${name}.log" 2>&1 &
  i=$(( i + 1 ))
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n || true; done
done

wait || true
echo "[$(date '+%F %T')] All per-model evals finished (per-model results in $BUCKET/<model>/)."

echo "[$(date '+%F %T')] Building combined results.md"
python -u "$SCRIPT_DIR/evaluate_english_mmlu.py" \
  --data-root "$DATA_ROOT" --models "${MODELS[@]}" \
  --out-dir "$OUT_DIR" --bucket "$BUCKET" --combine-only

echo "[$(date '+%F %T')] Reports:"
ls -1 "$OUT_DIR"/*.md
echo "[$(date '+%F %T')] ALL DONE"
