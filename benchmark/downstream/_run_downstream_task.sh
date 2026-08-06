#!/usr/bin/env bash
# Internal helper, not a task entry point: LoRA-finetune every model in
# ../main.yml on one downstream classification sub-task, sequentially
# (continue on failure), then upload each model's run dir to GCS.
#
# Usage: _run_downstream_task.sh <news|sentiment|writing> <finetune_script.py> <dataset_key>
set -uo pipefail

TASK="$1"                 # news | sentiment | writing
FINETUNE_SCRIPT="$2"      # e.g. finetune_news_category.py
DATASET_KEY="$3"          # datasets.downstream.<key> in main.yml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$BENCH_DIR")"
YAML_GET="$BENCH_DIR/lib/yaml_get.py"
MAIN_YML="$BENCH_DIR/main.yml"
CFG_YML="$SCRIPT_DIR/config.yml"
source "$BENCH_DIR/lib/gcs_upload.sh"

yaml() { python3 "$YAML_GET" "$MAIN_YML" "$CFG_YML" "$@"; }

export WANDB_PROJECT="$(yaml wandb_project)"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mapfile -t MODEL_NAMES < <(yaml models --field name)
mapfile -t MODEL_PATHS < <(yaml models --field path)

DATA_DIR="$REPO_DIR/$(yaml "datasets.downstream.${DATASET_KEY}")"
FAST_LOAD="${FAST_LOAD:-$(yaml fast_load)}"
LOCAL_MODEL_DIR="$(yaml local_model_dir)"
LOAD_IN_4BIT="$(yaml load_in_4bit)"

TRAIN_FILE="$(yaml "${TASK}.train_file")"
VAL_FILE="$(yaml "${TASK}.val_file")"
TEST_FILE="$(yaml "${TASK}.test_file")"
RESULTS_FILE="$(yaml "${TASK}.results_file")"
MAX_SEQ_LEN="$(yaml "${TASK}.max_seq_length")"
LORA_RANK="$(yaml "${TASK}.lora_rank")"
LORA_ALPHA="$(yaml "${TASK}.lora_alpha")"
LORA_DROPOUT="$(yaml "${TASK}.lora_dropout")"
LORA_TARGETS="$(yaml "${TASK}.lora_target_modules")"
EPOCHS="$(yaml "${TASK}.epochs")"
LR="$(yaml "${TASK}.lr")"
TRAIN_BS="$(yaml "${TASK}.train_batch_size")"
EVAL_BS="$(yaml "${TASK}.eval_batch_size")"
GRAD_ACCUM="$(yaml "${TASK}.grad_accum")"
WARMUP_RATIO="$(yaml "${TASK}.warmup_ratio")"
WEIGHT_DECAY="$(yaml "${TASK}.weight_decay")"
LOGGING_STEPS="$(yaml "${TASK}.logging_steps")"
EVAL_STEPS="$(yaml "${TASK}.eval_steps")"
SAVE_STEPS="$(yaml "${TASK}.save_steps")"
SAVE_TOTAL_LIMIT="$(yaml "${TASK}.save_total_limit")"
SEED="$(yaml "${TASK}.seed")"
BUCKET="$(yaml gcs.bucket_root)/$(yaml "${TASK}.bucket_prefix")"

fourbit_flag() { [ "$LOAD_IN_4BIT" = "true" ] && echo "--load_in_4bit"; }

declare -a FAILED=()

for idx in "${!MODEL_PATHS[@]}"; do
  MODEL_PATH="$REPO_DIR/${MODEL_PATHS[$idx]}"
  MODEL_NAME="${MODEL_NAMES[$idx]}"

  RUN_MODEL="$MODEL_PATH"
  if [ "$FAST_LOAD" = "true" ]; then
    LOCAL_MODEL="$LOCAL_MODEL_DIR/$MODEL_NAME"
    if [ ! -f "$LOCAL_MODEL/config.json" ]; then
      echo "[$(date '+%F %T')] Copying $MODEL_NAME to $LOCAL_MODEL (one-time)..."
      rm -rf "$LOCAL_MODEL"
      cp -r "$MODEL_PATH" "$LOCAL_MODEL"
    fi
    RUN_MODEL="$LOCAL_MODEL"
  fi

  OUTPUT_DIR="$SCRIPT_DIR/runs/${TASK}_lora/${MODEL_NAME}"
  mkdir -p "$OUTPUT_DIR"

  echo "########## downstream/${TASK}: ${MODEL_NAME} ##########"
  python "${SCRIPT_DIR}/${FINETUNE_SCRIPT}" \
    --model_name_or_path "${RUN_MODEL}" \
    --data_dir "${DATA_DIR}" \
    --train_file "${TRAIN_FILE}" \
    --val_file "${VAL_FILE}" \
    --test_file "${TEST_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --results_file "${OUTPUT_DIR}/${RESULTS_FILE}" \
    --max_seq_length "${MAX_SEQ_LEN}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_target_modules "${LORA_TARGETS}" \
    --num_train_epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --per_device_train_batch_size "${TRAIN_BS}" \
    --per_device_eval_batch_size "${EVAL_BS}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --logging_steps "${LOGGING_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --seed "${SEED}" \
    --wandb_project "${WANDB_PROJECT}" \
    --run_name "${TASK}-lora-${MODEL_NAME}" \
    $(fourbit_flag) \
    || { echo "!!! downstream/${TASK}/${MODEL_NAME} FAILED"; FAILED+=("${MODEL_NAME}"); continue; }

  upload_to_gcs "$OUTPUT_DIR" "${BUCKET}/${MODEL_NAME}"
done

echo "============================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "downstream/${TASK}: all models completed."
else
  echo "downstream/${TASK}: completed with failures in: ${FAILED[*]}"
fi
