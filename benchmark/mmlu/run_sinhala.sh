#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate all models (../main.yml) on SinhalaMMLU (bf16) IN PARALLEL.
# Hyperparameters come from config.yml; model/dataset/bucket paths from
# ../main.yml. Not meant to be run standalone — invoked by ../benchmark.sh.
#
# Concurrency knobs (override via env): NUM_GPUS, MAX_PARALLEL
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

mapfile -t MODEL_NAMES < <(yaml models --field name)
mapfile -t MODEL_PATHS < <(yaml models --field path)
mapfile -t ALPACA_MODELS < <(yaml alpaca_models)

DATA_ROOT="$REPO_DIR/$(yaml datasets.mmlu.sinhala)"
PROMPT_FILE="$REPO_DIR/$(yaml sinhala.prompt_file)"
OUT_DIR="$SCRIPT_DIR/$(yaml sinhala.output_dir)"
BUCKET="$(yaml gcs.bucket_root)/$(yaml sinhala.bucket_prefix)"

KSHOT="$(yaml sinhala.kshot)"
BATCH_SIZE="$(yaml sinhala.batch_size)"
MAX_LEN="$(yaml sinhala.max_len)"
NUM_GPUS="${NUM_GPUS:-$(yaml concurrency.num_gpus)}"
MAX_PARALLEL="${MAX_PARALLEL:-$(yaml concurrency.max_parallel)}"

mkdir -p "$OUT_DIR"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ALL_MODEL_PATHS=()
for p in "${MODEL_PATHS[@]}"; do ALL_MODEL_PATHS+=("$REPO_DIR/$p"); done

echo "[$(date '+%F %T')] SinhalaMMLU: ${#ALL_MODEL_PATHS[@]} models, up to $MAX_PARALLEL at once across $NUM_GPUS GPU(s)"

i=0
for idx in "${!ALL_MODEL_PATHS[@]}"; do
  MODEL="${ALL_MODEL_PATHS[$idx]}"
  name="${MODEL_NAMES[$idx]}"
  gpu=$(( i % NUM_GPUS ))
  echo "[$(date '+%F %T')] -> launching '$name' on GPU $gpu (log: $OUT_DIR/${name}.log)"
  HIP_VISIBLE_DEVICES="$gpu" CUDA_VISIBLE_DEVICES="$gpu" \
  python -u "$SCRIPT_DIR/evaluate_sinhala_mmlu.py" \
    --data-root     "$DATA_ROOT" \
    --prompt-file   "$PROMPT_FILE" \
    --models        "$MODEL" \
    --alpaca-models "${ALPACA_MODELS[@]}" \
    --kshot "$KSHOT" --batch-size "$BATCH_SIZE" --max-len "$MAX_LEN" \
    --out-dir "$OUT_DIR" --bucket "$BUCKET" --skip-existing \
    > "$OUT_DIR/${name}.log" 2>&1 &
  i=$(( i + 1 ))
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n || true; done
done
wait || true
echo "[$(date '+%F %T')] All per-model evals finished."

echo "[$(date '+%F %T')] Building combined results.md"
python -u "$SCRIPT_DIR/evaluate_sinhala_mmlu.py" \
  --data-root   "$DATA_ROOT" \
  --prompt-file "$PROMPT_FILE" \
  --models      "${ALL_MODEL_PATHS[@]}" \
  --out-dir "$OUT_DIR" --bucket "$BUCKET" --combine-only

upload_to_gcs "$OUT_DIR" "$BUCKET"
echo "[$(date '+%F %T')] mmlu/sinhala DONE"
