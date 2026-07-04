#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate all SinLlama / Llama models on HellaSwag IN PARALLEL. One process per
# model, each uploads to $BUCKET/<model_name>/ when done, then a combine pass.
#
# Default is the ORIGINAL English set only — used as the catastrophic-forgetting
# probe. (The Sinhala-translated set is excluded by default: NLLB mangles the
# adversarial endings + activity_label, so every model sits near the random
# floor and the number is uninformative. Still reachable via DATASET=sinhala.)
#
# HellaSwag is scored the lm-eval way (per-ending continuation log-likelihood,
# headline metric acc_norm) — the same protocol behind Llama-3-8B's public
# HellaSwag number. Validation split (test labels are hidden). 0-shot default.
#
# Env toggles:
#   DATASET=english|sinhala|both   (default english)
#   KSHOT (0)  NUM_GPUS (1)  MAX_PARALLEL (2)  BATCH_SIZE (32)  MAX_LEN (1024)
#   EN_DATA / SI_DATA / SI_TRAIN   dataset path overrides
#
# Usage:  bash run_eval_hellaswag.sh                 # English only, 0-shot (default)
#         DATASET=sinhala bash run_eval_hellaswag.sh # translated Sinhala set
#         DATASET=both    bash run_eval_hellaswag.sh # both
#         KSHOT=10 bash run_eval_hellaswag.sh        # 10-shot (leaderboard-style)
# ---------------------------------------------------------------------------
set -uo pipefail          # no -e: one model failing must not kill the rest

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# find the Google Cloud SDK if installed but not on PATH (enables uploads)
export PATH="$HOME/google-cloud-sdk/bin:$PATH"

MODELS=(
  "$PROJECT_DIR/llama-3-8b"
  "$PROJECT_DIR/SinLlama_v01"
  "$PROJECT_DIR/SinLlama_cpt_merged"
  "$PROJECT_DIR/SinLlama_Backtrianx_instruct"
)

# dataset locations (the HellaSwag data lives in $HOME, not the repo)
EN_DATA="${EN_DATA:-$HOME/hellaswag}"                         # dir w/ validation-*.parquet
SI_DATA="${SI_DATA:-$HOME/HellaSwag/data/translated}"        # dir w/ validation.sinhala.jsonl
SI_TRAIN="${SI_TRAIN:-$SI_DATA}"

DATASET="${DATASET:-english}"
KSHOT="${KSHOT:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LEN="${MAX_LEN:-1024}"

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_one_dataset() {
  local ds="$1" data="$2" train="$3"
  local tag="hellaswag_${ds}"
  [[ "$KSHOT" != "0" ]] && tag="${tag}_${KSHOT}shot"
  local out_dir="$SCRIPT_DIR/results_${tag}"
  local bucket="gs://sinllama-cpt/${tag}"
  mkdir -p "$out_dir"

  echo "[$(date '+%F %T')] === HellaSwag ($ds) — ${#MODELS[@]} models, up to $MAX_PARALLEL at once across $NUM_GPUS GPU(s), ${KSHOT}-shot ==="
  local i=0
  for MODEL in "${MODELS[@]}"; do
    local name gpu
    name="$(basename "$MODEL")"
    gpu=$(( i % NUM_GPUS ))
    echo "[$(date '+%F %T')] -> launching '$name' on GPU $gpu (log: $out_dir/${name}.log)"
    HIP_VISIBLE_DEVICES="$gpu" CUDA_VISIBLE_DEVICES="$gpu" \
    python -u "$SCRIPT_DIR/evaluate_hellaswag.py" \
      --data-path "$data" --train-path "$train" \
      --dataset "$ds" --split validation \
      --models "$MODEL" \
      --kshot "$KSHOT" --batch-size "$BATCH_SIZE" --max-len "$MAX_LEN" \
      --out-dir "$out_dir" --bucket "$bucket" \
      > "$out_dir/${name}.log" 2>&1 &
    i=$(( i + 1 ))
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n || true; done
  done
  wait || true
  echo "[$(date '+%F %T')] ($ds) per-model evals finished; building combined results.md"
  python -u "$SCRIPT_DIR/evaluate_hellaswag.py" \
    --data-path "$data" --dataset "$ds" \
    --models "${MODELS[@]}" \
    --out-dir "$out_dir" --bucket "$bucket" --combine-only
  echo "[$(date '+%F %T')] ($ds) reports:"; ls -1 "$out_dir"/*.md
}

if [[ "$DATASET" == "english" || "$DATASET" == "both" ]]; then
  run_one_dataset english "$EN_DATA" "$EN_DATA"
fi
if [[ "$DATASET" == "sinhala" || "$DATASET" == "both" ]]; then
  run_one_dataset sinhala "$SI_DATA" "$SI_TRAIN"
fi

echo "[$(date '+%F %T')] ALL HELLASWAG EVALS DONE"
