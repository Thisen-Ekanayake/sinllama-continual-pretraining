#!/usr/bin/env bash
# Internal helper, not a task entry point: run one (language, parallel/
# nonparallel) Global-PIQA eval across all configured models, in parallel,
# then combine + upload. Sourced-by-args from run_sinhala.sh / run_english.sh.
#
# Usage: _run_piqa_variant.sh <sinhala|english> <parallel|nonparallel>
set -uo pipefail

LANG_NAME="$1"     # sinhala | english
VARIANT="$2"        # parallel | nonparallel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$BENCH_DIR")"
YAML_GET="$BENCH_DIR/lib/yaml_get.py"
MAIN_YML="$BENCH_DIR/main.yml"
CFG_YML="$SCRIPT_DIR/config.yml"
source "$BENCH_DIR/lib/gcs_upload.sh"

yaml() { python3 "$YAML_GET" "$MAIN_YML" "$CFG_YML" "$@"; }

mapfile -t MODEL_NAMES < <(yaml models --field name)
mapfile -t MODEL_PATHS < <(yaml models --field path)
mapfile -t ALPACA_MODELS < <(yaml alpaca_models)
ALL_MODEL_PATHS=()
for p in "${MODEL_PATHS[@]}"; do ALL_MODEL_PATHS+=("$REPO_DIR/$p"); done

DATA="$REPO_DIR/$(yaml "datasets.piqa.${VARIANT}")"
KSHOT="$(yaml kshot)"
SEED="$(yaml seed)"
BATCH_SIZE="$(yaml batch_size)"
MAX_LEN="$(yaml max_len)"
QUANT="$(yaml quant)"
NUM_GPUS="${NUM_GPUS:-$(yaml concurrency.num_gpus)}"
MAX_PARALLEL="${MAX_PARALLEL:-$(yaml concurrency.max_parallel)}"

OUT_DIR="$SCRIPT_DIR/$(yaml "${VARIANT}_output_dir")_${LANG_NAME}"
BUCKET="$(yaml gcs.bucket_root)/$(yaml bucket_prefix)_${VARIANT}_${LANG_NAME}"
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_piqa_${VARIANT}_${LANG_NAME}.py"

mkdir -p "$OUT_DIR"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date '+%F %T')] PIQA ${VARIANT}/${LANG_NAME}: ${#ALL_MODEL_PATHS[@]} models, up to $MAX_PARALLEL at once across $NUM_GPUS GPU(s)"

i=0
for idx in "${!ALL_MODEL_PATHS[@]}"; do
  MODEL="${ALL_MODEL_PATHS[$idx]}"
  name="${MODEL_NAMES[$idx]}"
  gpu=$(( i % NUM_GPUS ))
  echo "[$(date '+%F %T')] -> launching '$name' on GPU $gpu (log: $OUT_DIR/${name}.log)"
  HIP_VISIBLE_DEVICES="$gpu" CUDA_VISIBLE_DEVICES="$gpu" \
  python -u "$EVAL_SCRIPT" \
    --data "$DATA" \
    --models "$MODEL" \
    --alpaca-models "${ALPACA_MODELS[@]}" \
    --kshot "$KSHOT" --seed "$SEED" --batch-size "$BATCH_SIZE" --max-len "$MAX_LEN" --quant "$QUANT" \
    --out-dir "$OUT_DIR" --bucket "$BUCKET" --skip-existing \
    > "$OUT_DIR/${name}.log" 2>&1 &
  i=$(( i + 1 ))
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n || true; done
done
wait || true
echo "[$(date '+%F %T')] All per-model evals finished."

echo "[$(date '+%F %T')] Building combined results.md"
python -u "$EVAL_SCRIPT" \
  --data "$DATA" --models "${ALL_MODEL_PATHS[@]}" \
  --out-dir "$OUT_DIR" --bucket "$BUCKET" --combine-only

upload_to_gcs "$OUT_DIR" "$BUCKET"
echo "[$(date '+%F %T')] piqa/${VARIANT}/${LANG_NAME} DONE"
