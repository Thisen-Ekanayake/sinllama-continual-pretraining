#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate all models (../main.yml) on English MMLU (bf16) IN PARALLEL, to
# check catastrophic forgetting. Hyperparameters come from config.yml.
# Not meant to be run standalone — invoked by ../benchmark.sh.
#
# Env toggles:
#   CANONICAL=1   standard Hendrycks header-once format for ALL models
#                 (leaderboard-comparable numbers); no Alpaca wrap
#   NUM_GPUS, MAX_PARALLEL   concurrency knobs
# ---------------------------------------------------------------------------
set -uo pipefail   # no -e: one model failing must not kill the rest

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$BENCH_DIR")"
YAML_GET="$BENCH_DIR/lib/yaml_get.py"
MAIN_YML="$BENCH_DIR/main.yml"
CFG_YML="$SCRIPT_DIR/config.yml"
source "$BENCH_DIR/lib/gcs_upload.sh"

yaml() { python3 "$YAML_GET" "$MAIN_YML" "$CFG_YML" "$@"; }

# find the Google Cloud SDK if it's installed but not on PATH (enables uploads)
export PATH="$HOME/google-cloud-sdk/bin:$PATH"

mapfile -t MODEL_NAMES < <(yaml models --field name)
mapfile -t MODEL_PATHS < <(yaml models --field path)
mapfile -t ALPACA_MODELS < <(yaml alpaca_models)
mapfile -t CHAT_MODELS < <(yaml chat_models)

DATA_ROOT="$REPO_DIR/$(yaml datasets.mmlu.english)"
KSHOT="$(yaml english.kshot)"
BATCH_SIZE="$(yaml english.batch_size)"
MAX_LEN="$(yaml english.max_len)"
NUM_GPUS="${NUM_GPUS:-$(yaml concurrency.num_gpus)}"
MAX_PARALLEL="${MAX_PARALLEL:-$(yaml concurrency.max_parallel)}"

STYLE_ARGS=()
if [[ "${CANONICAL:-0}" == "1" ]]; then
  STYLE_ARGS+=(--canonical)
  ALPACA_MODELS=()
  OUT_DIR="$SCRIPT_DIR/$(yaml english.canonical_output_dir)"
  BUCKET="$(yaml gcs.bucket_root)/$(yaml english.canonical_bucket_prefix)"
  echo "[$(date '+%F %T')] MODE: canonical Hendrycks format (all models raw)"
else
  OUT_DIR="$SCRIPT_DIR/$(yaml english.output_dir)"
  BUCKET="$(yaml gcs.bucket_root)/$(yaml english.bucket_prefix)"
  echo "[$(date '+%F %T')] MODE: per-block format (Alpaca for instruct)"
fi
if [[ "${CALIBRATE:-0}" == "1" ]]; then
  STYLE_ARGS+=(--calibrate)
  echo "[$(date '+%F %T')] Contextual calibration: ON"
fi

ALPACA_ARG=()
[[ "${#ALPACA_MODELS[@]}" -gt 0 ]] && ALPACA_ARG=(--alpaca-models "${ALPACA_MODELS[@]}")
CHAT_ARG=()
[[ "${#CHAT_MODELS[@]}" -gt 0 ]] && CHAT_ARG=(--chat-models "${CHAT_MODELS[@]}")

mkdir -p "$OUT_DIR"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ALL_MODEL_PATHS=()
for p in "${MODEL_PATHS[@]}"; do ALL_MODEL_PATHS+=("$REPO_DIR/$p"); done

echo "[$(date '+%F %T')] English MMLU: ${#ALL_MODEL_PATHS[@]} models, up to $MAX_PARALLEL at once across $NUM_GPUS GPU(s)"

i=0
for idx in "${!ALL_MODEL_PATHS[@]}"; do
  MODEL="${ALL_MODEL_PATHS[$idx]}"
  name="${MODEL_NAMES[$idx]}"
  gpu=$(( i % NUM_GPUS ))
  echo "[$(date '+%F %T')] -> launching '$name' on GPU $gpu (log: $OUT_DIR/${name}.log)"
  HIP_VISIBLE_DEVICES="$gpu" CUDA_VISIBLE_DEVICES="$gpu" \
  python -u "$SCRIPT_DIR/evaluate_english_mmlu.py" \
    --data-root "$DATA_ROOT" \
    --models    "$MODEL" \
    "${ALPACA_ARG[@]}" "${CHAT_ARG[@]}" "${STYLE_ARGS[@]}" \
    --kshot "$KSHOT" --batch-size "$BATCH_SIZE" --max-len "$MAX_LEN" \
    --out-dir "$OUT_DIR" --bucket "$BUCKET" --skip-existing \
    > "$OUT_DIR/${name}.log" 2>&1 &
  i=$(( i + 1 ))
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n || true; done
done
wait || true
echo "[$(date '+%F %T')] All per-model evals finished."

echo "[$(date '+%F %T')] Building combined results.md"
python -u "$SCRIPT_DIR/evaluate_english_mmlu.py" \
  --data-root "$DATA_ROOT" --models "${ALL_MODEL_PATHS[@]}" \
  "${STYLE_ARGS[@]}" \
  --out-dir "$OUT_DIR" --bucket "$BUCKET" --combine-only

upload_to_gcs "$OUT_DIR" "$BUCKET"
echo "[$(date '+%F %T')] mmlu/english DONE"
