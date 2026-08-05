#!/bin/bash
# LoRA fine-tune SinLlama_v01 on the cleaned MMLU-Sinhala auxiliary_train set
# (prompt/completion shaped; see sft/build_mmlu_sft.py + sft/build_mmlu_dataset.py).
# Produces the adapter merged into SinLlama_v02.
#
# Dataset: data/mmlu_sft/train/aux_train_clean.jsonl (97,185 items) -> dataset_dir
#          data/mmlu_sft/aux_val_proxy.jsonl (~1,117 items, translated-validation
#          proxy, moral_scenarios excluded) -> validation_file, monitoring only.
#
# Target hardware: a single AMD MI300X (192 GB, ROCm).
#   * bf16 LoRA, NO bitsandbytes.
#   * sdpa attention (flash-attn-2 left off for ROCm portability).
#   * no gradient checkpointing (fits comfortably; faster without).
set -euo pipefail

export WANDB_PROJECT=sinllama-sft-mmlu
export TOKENIZERS_PARALLELISM=false
# ROCm allocator: reduce fragmentation OOMs (analogous to PYTORCH_CUDA_ALLOC_CONF).
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# ---- LoRA (same recipe as sft/run_sft.sh: r=64, alpha=128, all 7 projections) ----
lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
lora_dropout=0.05
# Keep token embeddings / lm_head FROZEN.
modules_to_save=""

# ---- paths (override via env) ----
pretrained_model=${PRETRAINED_MODEL:-../models/SinLlama_v01}
tokenizer_name_or_path=${pretrained_model}
# dataset_dir must contain ONLY the training jsonl(s): the harness globs every
# *.json*/*.jsonl in it as train data. The val file therefore lives one level up.
dataset_dir=${DATASET_DIR:-../data/mmlu_sft/train}                 # holds aux_train_clean.jsonl
validation_file=${VALIDATION_FILE:-../data/mmlu_sft/aux_val_proxy.jsonl}
output_dir=${OUTPUT_DIR:-runs/sft_mmlu_lora}

# ---- batch / schedule ----
per_device_train_batch_size=16
per_device_eval_batch_size=16
gradient_accumulation_steps=4            # effective batch = 64
max_seq_length=1024                      # covers 99.81% of the cleaned training set
num_train_epochs=5                       # upper cap; early stopping expected to fire sooner

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Only pass --modules_to_save when it is non-empty.
mts_flag=()
[ -n "${modules_to_save}" ] && mts_flag=(--modules_to_save "${modules_to_save}")

torchrun --nnodes 1 --nproc_per_node 1 "${SCRIPT_DIR}/run_clm_sft_mmlu_with_peft.py" \
    --model_name_or_path "${pretrained_model}" \
    --tokenizer_name_or_path "${tokenizer_name_or_path}" \
    --dataset_dir "${dataset_dir}" \
    --validation_file "${validation_file}" \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --low_cpu_mem_usage \
    --seed 42 \
    --bf16 \
    --num_train_epochs ${num_train_epochs} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --logging_strategy steps \
    --logging_steps 20 \
    --logging_first_step True \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --eval_strategy steps \
    --eval_steps 100 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
    --greater_is_better True \
    --early_stopping_patience 5 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 4 \
    --max_seq_length ${max_seq_length} \
    --output_dir "${output_dir}" \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --group_by_length True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    "${mts_flag[@]}" \
    --torch_dtype bfloat16 \
    --load_in_kbits 16 \
    --report_to wandb \
    --run_name sinllama-sft-mmlu-lora-r64 \
    --ddp_find_unused_parameters False
