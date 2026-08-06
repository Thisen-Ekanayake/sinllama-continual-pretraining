#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate all models (../main.yml) on HellaSwag IN PARALLEL (lm-eval style,
# acc_norm headline metric). Hyperparameters come from config.yml.
# Not meant to be run standalone — invoked by ../benchmark.sh.
#
# Env toggles: DATASET=english|sinhala|both, KSHOT, NUM_GPUS, MAX_PARALLEL,
#              EN_DATA / SI_DATA path overrides
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

export PATH="$HOME/google-cloud-sdk/bin:$PATH"

mapfile -t MODEL_NAMES < <(yaml models --field name)
mapfile -t MODEL_PATHS < <(yaml models --field path)
ALL_MODEL_PATHS=()
for p in "${MODEL_PATHS[@]}"; do ALL_MODEL_PATHS+=("$REPO_DIR/$p"); done

EN_DATA="${EN_DATA:-$(eval echo "$(yaml datasets.hellaswag.english)")}"
SI_DATA="${SI_DATA:-$(eval echo "$(yaml datasets.hellaswag.sinhala)")}"

DATASET="${DATASET:-$(yaml dataset)}"
KSHOT="${KSHOT:-$(yaml kshot)}"
SPLIT="$(yaml split)"
BATCH_SIZE="$(yaml batch_size)"
MAX_LEN="$(yaml max_len)"
NUM_GPUS="${NUM_GPUS:-$(yaml concurrency.num_gpus)}"
MAX_PARALLEL="${MAX_PARALLEL:-$(yaml concurrency.max_parallel)}"
BUCKET_ROOT="$(yaml gcs.bucket_root)"
BUCKET_PREFIX="$(yaml bucket_prefix)"

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_one_dataset() {
  local ds="$1" data="$2"
  local tag="${ds}"
  [[ "$KSHOT" != "0" ]] && tag="${tag}_${KSHOT}shot"
  local out_dir="$SCRIPT_DIR/$(yaml output_dir)_${tag}"
  local bucket="${BUCKET_ROOT}/${BUCKET_PREFIX}_${tag}"
  mkdir -p "$out_dir"

  echo "[$(date '+%F %T')] === HellaSwag ($ds) — ${#ALL_MODEL_PATHS[@]} models, up to $MAX_PARALLEL at once across $NUM_GPUS GPU(s), ${KSHOT}-shot ==="
  local i=0
  for idx in "${!ALL_MODEL_PATHS[@]}"; do
    local MODEL="${ALL_MODEL_PATHS[$idx]}" name="${MODEL_NAMES[$idx]}" gpu
    gpu=$(( i % NUM_GPUS ))
    echo "[$(date '+%F %T')] -> launching '$name' on GPU $gpu (log: $out_dir/${name}.log)"
    HIP_VISIBLE_DEVICES="$gpu" CUDA_VISIBLE_DEVICES="$gpu" \
    python -u "$SCRIPT_DIR/evaluate_hellaswag.py" \
      --data-path "$data" --train-path "$data" \
      --dataset "$ds" --split "$SPLIT" \
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
    --models "${ALL_MODEL_PATHS[@]}" \
    --out-dir "$out_dir" --bucket "$bucket" --combine-only

  upload_to_gcs "$out_dir" "$bucket"
}

if [[ "$DATASET" == "english" || "$DATASET" == "both" ]]; then
  run_one_dataset english "$EN_DATA"
fi
if [[ "$DATASET" == "sinhala" || "$DATASET" == "both" ]]; then
  run_one_dataset sinhala "$SI_DATA"
fi

echo "[$(date '+%F %T')] hellaswag DONE"
