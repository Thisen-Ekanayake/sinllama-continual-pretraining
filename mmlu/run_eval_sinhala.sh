#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate all SinLlama / Llama models on SinhalaMMLU (3-shot, bf16) IN PARALLEL.
#
# Each model is evaluated in its own process; as soon as a model finishes, that
# process uploads its result files to  $BUCKET/<model_name>/  (see the eval
# script's --bucket). A final --combine-only pass builds and uploads the
# aggregate results.md once every model is done.
#
# The Bactrian-X instruct model is scored in the Alpaca
# "### Instruction / ### Input / ### Response" template it was SFT'd on
# (--alpaca-models); base / CPT models use the raw SinhalaMMLU template.
#
# Target hardware: AMD MI300X (192 GB, ROCm). Concurrency knobs (override via env):
#   NUM_GPUS      physical GPUs to spread jobs across            (default 1)
#   MAX_PARALLEL  concurrent evals — they share a GPU if > NUM_GPUS (default 2)
#   BATCH_SIZE    per-process batch size                         (default 32)
#   MAX_LEN       max prompt length                              (default 4096)
#
# Usage:  bash run_eval.sh
#         MAX_PARALLEL=4 NUM_GPUS=4 bash run_eval.sh     # multi-GPU node
# ---------------------------------------------------------------------------
set -uo pipefail          # NOTE: no -e — one model failing must not kill the rest

# Resolve paths from this script's own location (<repo>/mmlu/run_eval.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$SCRIPT_DIR/SinhalaMMLU"
OUT_DIR="$PROJECT_DIR/benchmark/SinhalaMMLU_results_final"

# Models to evaluate: base -> cpt -> instruct -> v02, plus the original Llama-3 ref.
MODELS=(
  "$PROJECT_DIR/models/llama-3-8b"
  "$PROJECT_DIR/models/SinLlama_v01"
  "$PROJECT_DIR/models/SinLlama_cpt"
  "$PROJECT_DIR/models/SinLlama_Bactrianx_Instruct"
  "$PROJECT_DIR/models/SinLlama_v02"
)

# Model-name substring(s) scored in the native Alpaca SFT format.
ALPACA_MODELS="SinLlama_Bactrianx_Instruct"

# GCS bucket (leave empty to skip all uploads). Per-model files land in
# $BUCKET/<model_name>/ ; the combined results.md lands in $BUCKET/ .
BUCKET="gs://sinllama-cpt/mmlu_results_bf16"

# Concurrency knobs
NUM_GPUS="${NUM_GPUS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LEN="${MAX_LEN:-4096}"

mkdir -p "$OUT_DIR"

# MI300X (ROCm) allocator: reduce fragmentation on the long Sinhala sequences.
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date '+%F %T')] Launching ${#MODELS[@]} model evals — up to $MAX_PARALLEL at once across $NUM_GPUS GPU(s)"

i=0
for MODEL in "${MODELS[@]}"; do
  name="$(basename "$MODEL")"
  gpu=$(( i % NUM_GPUS ))
  echo "[$(date '+%F %T')] -> launching '$name' on GPU $gpu (log: $OUT_DIR/${name}.log)"
  HIP_VISIBLE_DEVICES="$gpu" CUDA_VISIBLE_DEVICES="$gpu" \
  python -u "$SCRIPT_DIR/evaluate_sinhala_mmlu.py" \
    --data-root     "$DATA_ROOT" \
    --prompt-file   "$SCRIPT_DIR/prompt.txt" \
    --models        "$MODEL" \
    --alpaca-models "$ALPACA_MODELS" \
    --kshot 3 --batch-size "$BATCH_SIZE" --max-len "$MAX_LEN" \
    --out-dir "$OUT_DIR" --bucket "$BUCKET" --skip-existing \
    > "$OUT_DIR/${name}.log" 2>&1 &
  i=$(( i + 1 ))
  # throttle: wait for a slot to free up before launching the next model
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n || true; done
done

wait || true
echo "[$(date '+%F %T')] All per-model evals finished (per-model results in $BUCKET/<model>/)."

# Build + upload the combined results.md from the per-model metrics.
echo "[$(date '+%F %T')] Building combined results.md"
python -u "$SCRIPT_DIR/evaluate_sinhala_mmlu.py" \
  --data-root   "$DATA_ROOT" \
  --prompt-file "$SCRIPT_DIR/prompt.txt" \
  --models      "${MODELS[@]}" \
  --out-dir "$OUT_DIR" --bucket "$BUCKET" --combine-only

echo "[$(date '+%F %T')] Reports:"
ls -1 "$OUT_DIR"/*.md
echo "[$(date '+%F %T')] ALL DONE"
