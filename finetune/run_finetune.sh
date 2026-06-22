#!/bin/bash
# Run the three downstream LoRA finetunes (writing-style, sentiment, news-category)
# one after another on the merged SinLlama model.
#
# Each task writes results_<task>.txt into its own output_dir.
# If a task fails, the script logs it and continues to the next one.
#
# Usage:
#   bash finetune/run_finetune.sh
# Edit the variables below to change paths / hyperparameters.

export WANDB_PROJECT=sinllama-finetune

# ===================== shared / global =====================
# Path to the merged final model (output of the LoRA-merge step).
MERGED_MODEL=../SinLlama_cpt_merged

# The three dataset directories.
WRITING_DATA_DIR=../data/Writing
SENT_DATA_DIR=../data/sentiment
NEWS_DATA_DIR=../data/news

# Where the run artifacts (adapters + results_<task>.txt) are written.
WRITING_OUTPUT_DIR=runs/writing_style_lora
SENT_OUTPUT_DIR=runs/sentiment_lora
NEWS_OUTPUT_DIR=runs/news_category_lora

# Set to 1 to use 4-bit QLoRA (lower VRAM); 0 for bf16 LoRA.
LOAD_IN_4BIT=0

PY=python   # change to your interpreter if needed (e.g. python3)

# Helper: append "--load_in_4bit" only when requested.
fourbit_flag() { [ "$LOAD_IN_4BIT" = "1" ] && echo "--load_in_4bit"; }

# ============================================================
# TASK 1: Writing-style classification
#   labels: ACADEMIC, CREATIVE, NEWS, BLOG   (train=10010)
# ============================================================
W_TRAIN_FILE=writing_style_train.csv
W_VAL_FILE=writing_style_val.csv
W_TEST_FILE=writing_style_test.csv
W_MAX_SEQ_LEN=512
W_LORA_RANK=16
W_LORA_ALPHA=32
W_LORA_DROPOUT=0.05
W_LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
W_EPOCHS=3
W_LR=2e-4
W_TRAIN_BS=8
W_EVAL_BS=16
W_GRAD_ACCUM=2
W_WARMUP_RATIO=0.05
W_WEIGHT_DECAY=0.0
W_LOGGING_STEPS=20
W_EVAL_STEPS=200
W_SAVE_STEPS=200
W_SAVE_TOTAL_LIMIT=2
W_SEED=42
W_RUN_NAME=writing-style-lora

# ============================================================
# TASK 2: Sentiment analysis
#   labels: POSITIVE, NEGATIVE, NEUTRAL      (train=7238)
# ============================================================
S_TRAIN_FILE=sentiment_train.csv
S_VAL_FILE=sentiment_val.csv
S_TEST_FILE=sentiment_test.csv
S_MAX_SEQ_LEN=512
S_LORA_RANK=16
S_LORA_ALPHA=32
S_LORA_DROPOUT=0.05
S_LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
S_EPOCHS=3
S_LR=2e-4
S_TRAIN_BS=8
S_EVAL_BS=16
S_GRAD_ACCUM=2
S_WARMUP_RATIO=0.05
S_WEIGHT_DECAY=0.0
S_LOGGING_STEPS=20
S_EVAL_STEPS=200
S_SAVE_STEPS=200
S_SAVE_TOTAL_LIMIT=2
S_SEED=42
S_RUN_NAME=sentiment-lora

# ============================================================
# TASK 3: News-category classification
#   labels: 0..4                              (train=2596)
# ============================================================
N_TRAIN_FILE=news_train.csv
N_VAL_FILE=news_val.csv
N_TEST_FILE=news_test.csv
N_MAX_SEQ_LEN=512
N_LORA_RANK=16
N_LORA_ALPHA=32
N_LORA_DROPOUT=0.05
N_LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
N_EPOCHS=3
N_LR=2e-4
N_TRAIN_BS=8
N_EVAL_BS=16
N_GRAD_ACCUM=2
N_WARMUP_RATIO=0.05
N_WEIGHT_DECAY=0.0
N_LOGGING_STEPS=20
N_EVAL_STEPS=100
N_SAVE_STEPS=100
N_SAVE_TOTAL_LIMIT=2
N_SEED=42
N_RUN_NAME=news-category-lora

# ============================================================
# Run them one by one (continue on failure).
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
declare -a FAILED=()

echo "########## TASK 1/3: writing-style ##########"
$PY "${SCRIPT_DIR}/finetune_writing_style.py" \
    --model_name_or_path "${MERGED_MODEL}" \
    --data_dir "${WRITING_DATA_DIR}" \
    --train_file "${W_TRAIN_FILE}" \
    --val_file "${W_VAL_FILE}" \
    --test_file "${W_TEST_FILE}" \
    --output_dir "${WRITING_OUTPUT_DIR}" \
    --results_file "${WRITING_OUTPUT_DIR}/results_writing.txt" \
    --max_seq_length "${W_MAX_SEQ_LEN}" \
    --lora_rank "${W_LORA_RANK}" \
    --lora_alpha "${W_LORA_ALPHA}" \
    --lora_dropout "${W_LORA_DROPOUT}" \
    --lora_target_modules "${W_LORA_TARGETS}" \
    --num_train_epochs "${W_EPOCHS}" \
    --learning_rate "${W_LR}" \
    --per_device_train_batch_size "${W_TRAIN_BS}" \
    --per_device_eval_batch_size "${W_EVAL_BS}" \
    --gradient_accumulation_steps "${W_GRAD_ACCUM}" \
    --warmup_ratio "${W_WARMUP_RATIO}" \
    --weight_decay "${W_WEIGHT_DECAY}" \
    --logging_steps "${W_LOGGING_STEPS}" \
    --eval_steps "${W_EVAL_STEPS}" \
    --save_steps "${W_SAVE_STEPS}" \
    --save_total_limit "${W_SAVE_TOTAL_LIMIT}" \
    --seed "${W_SEED}" \
    --wandb_project "${WANDB_PROJECT}" \
    --run_name "${W_RUN_NAME}" \
    $(fourbit_flag) \
    || { echo "!!! writing-style FAILED"; FAILED+=("writing-style"); }

echo "########## TASK 2/3: sentiment ##########"
$PY "${SCRIPT_DIR}/finetune_sentiment.py" \
    --model_name_or_path "${MERGED_MODEL}" \
    --data_dir "${SENT_DATA_DIR}" \
    --train_file "${S_TRAIN_FILE}" \
    --val_file "${S_VAL_FILE}" \
    --test_file "${S_TEST_FILE}" \
    --output_dir "${SENT_OUTPUT_DIR}" \
    --results_file "${SENT_OUTPUT_DIR}/results_sentiment.txt" \
    --max_seq_length "${S_MAX_SEQ_LEN}" \
    --lora_rank "${S_LORA_RANK}" \
    --lora_alpha "${S_LORA_ALPHA}" \
    --lora_dropout "${S_LORA_DROPOUT}" \
    --lora_target_modules "${S_LORA_TARGETS}" \
    --num_train_epochs "${S_EPOCHS}" \
    --learning_rate "${S_LR}" \
    --per_device_train_batch_size "${S_TRAIN_BS}" \
    --per_device_eval_batch_size "${S_EVAL_BS}" \
    --gradient_accumulation_steps "${S_GRAD_ACCUM}" \
    --warmup_ratio "${S_WARMUP_RATIO}" \
    --weight_decay "${S_WEIGHT_DECAY}" \
    --logging_steps "${S_LOGGING_STEPS}" \
    --eval_steps "${S_EVAL_STEPS}" \
    --save_steps "${S_SAVE_STEPS}" \
    --save_total_limit "${S_SAVE_TOTAL_LIMIT}" \
    --seed "${S_SEED}" \
    --wandb_project "${WANDB_PROJECT}" \
    --run_name "${S_RUN_NAME}" \
    $(fourbit_flag) \
    || { echo "!!! sentiment FAILED"; FAILED+=("sentiment"); }

echo "########## TASK 3/3: news-category ##########"
$PY "${SCRIPT_DIR}/finetune_news_category.py" \
    --model_name_or_path "${MERGED_MODEL}" \
    --data_dir "${NEWS_DATA_DIR}" \
    --train_file "${N_TRAIN_FILE}" \
    --val_file "${N_VAL_FILE}" \
    --test_file "${N_TEST_FILE}" \
    --output_dir "${NEWS_OUTPUT_DIR}" \
    --results_file "${NEWS_OUTPUT_DIR}/results_news.txt" \
    --max_seq_length "${N_MAX_SEQ_LEN}" \
    --lora_rank "${N_LORA_RANK}" \
    --lora_alpha "${N_LORA_ALPHA}" \
    --lora_dropout "${N_LORA_DROPOUT}" \
    --lora_target_modules "${N_LORA_TARGETS}" \
    --num_train_epochs "${N_EPOCHS}" \
    --learning_rate "${N_LR}" \
    --per_device_train_batch_size "${N_TRAIN_BS}" \
    --per_device_eval_batch_size "${N_EVAL_BS}" \
    --gradient_accumulation_steps "${N_GRAD_ACCUM}" \
    --warmup_ratio "${N_WARMUP_RATIO}" \
    --weight_decay "${N_WEIGHT_DECAY}" \
    --logging_steps "${N_LOGGING_STEPS}" \
    --eval_steps "${N_EVAL_STEPS}" \
    --save_steps "${N_SAVE_STEPS}" \
    --save_total_limit "${N_SAVE_TOTAL_LIMIT}" \
    --seed "${N_SEED}" \
    --wandb_project "${WANDB_PROJECT}" \
    --run_name "${N_RUN_NAME}" \
    $(fourbit_flag) \
    || { echo "!!! news-category FAILED"; FAILED+=("news-category"); }

echo "============================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All 3 finetune tasks completed."
else
    echo "Completed with failures in: ${FAILED[*]}"
fi
echo "Results files:"
echo "  ${WRITING_OUTPUT_DIR}/results_writing.txt"
echo "  ${SENT_OUTPUT_DIR}/results_sentiment.txt"
echo "  ${NEWS_OUTPUT_DIR}/results_news.txt"
