#!/bin/bash
# Downstream LoRA finetune: Sinhala Sentiment Analysis.
#   labels: POSITIVE, NEGATIVE, NEUTRAL      (train=7238)
#
# Writes results_sentiment.txt into the output_dir.
#
# Usage:
#   bash finetune/run_sentiment.sh
#   FAST_LOAD=0 bash finetune/run_sentiment.sh   # skip the /dev/shm copy

export WANDB_PROJECT=sinllama-finetune
export TOKENIZERS_PARALLELISM=false
# Reduce CUDA fragmentation OOMs (the large 139k-vocab logits are memory-heavy).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===================== shared / global =====================
# Path to the merged final model (output of the LoRA-merge step).
MERGED_MODEL=../SinLlama_cpt_merged

# /workspace is a slow MooseFS network volume (~46 MB/s). Copy the 16GB model once
# to RAM-backed /dev/shm so it loads in seconds instead of ~6 min. Set FAST_LOAD=0
# to skip. The copy is shared across the three per-task scripts.
FAST_LOAD=${FAST_LOAD:-1}
if [ "$FAST_LOAD" = "1" ]; then
    LOCAL_MODEL=/dev/shm/SinLlama_cpt_merged
    if [ ! -f "$LOCAL_MODEL/config.json" ]; then
        echo "Copying merged model to $LOCAL_MODEL (one-time, ~6 min over network)..."
        rm -rf "$LOCAL_MODEL"
        cp -r "$MERGED_MODEL" "$LOCAL_MODEL"
    fi
    MERGED_MODEL="$LOCAL_MODEL"
fi

DATA_DIR=../data/sentiment
OUTPUT_DIR=runs/sentiment_lora

# Set to 1 to use 4-bit QLoRA (lower VRAM); 0 for bf16 LoRA.
LOAD_IN_4BIT=0

PY=python   # change to your interpreter if needed (e.g. python3)
fourbit_flag() { [ "$LOAD_IN_4BIT" = "1" ] && echo "--load_in_4bit"; }

# ===================== task hyperparameters =====================
TRAIN_FILE=sentiment_train.csv
VAL_FILE=sentiment_val.csv
TEST_FILE=sentiment_test.csv
MAX_SEQ_LEN=512
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
EPOCHS=1
LR=2e-4
TRAIN_BS=8
EVAL_BS=16
GRAD_ACCUM=2
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.0
LOGGING_STEPS=20
EVAL_STEPS=50
SAVE_STEPS=50
SAVE_TOTAL_LIMIT=2
SEED=42
RUN_NAME=sentiment-lora

# ===================== run =====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "########## sentiment ##########"
$PY "${SCRIPT_DIR}/finetune_sentiment.py" \
    --model_name_or_path "${MERGED_MODEL}" \
    --data_dir "${DATA_DIR}" \
    --train_file "${TRAIN_FILE}" \
    --val_file "${VAL_FILE}" \
    --test_file "${TEST_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --results_file "${OUTPUT_DIR}/results_sentiment.txt" \
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
    --run_name "${RUN_NAME}" \
    $(fourbit_flag) \
    || { echo "!!! sentiment FAILED"; exit 1; }

echo "Done. Results -> ${OUTPUT_DIR}/results_sentiment.txt"
