#!/bin/bash
# Instruction SFT of the CPT-merged SinLlama model on UltraChat-Sinhala (sft split).
#
# Dataset: multi-turn `messages` JSONL translated from HuggingFaceH4/ultrachat_200k.
#   train_sft.sinhala.jsonl  (207,831 dialogues)  -> dataset_dir
#   test_sft.sinhala.jsonl   ( 23,106 dialogues)  -> subsample for validation_file
# Pull these from the VM first with `bash sft/prepare_data.sh`.
#
# Target hardware: a single AMD MI300X (192 GB, ROCm). Notes baked in below:
#   * bf16 LoRA, NO bitsandbytes (4/8-bit is unreliable on ROCm and unnecessary at 192 GB).
#   * sdpa attention (flash-attn-2 left off for ROCm portability).
#   * no gradient checkpointing (fits comfortably; faster without).
set -euo pipefail

export WANDB_PROJECT=sinllama-sft
export TOKENIZERS_PARALLELISM=false
# ROCm allocator: reduce fragmentation OOMs (analogous to PYTORCH_CUDA_ALLOC_CONF).
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# ---- LoRA (matches the CPT recipe: r=64, alpha=128, all 7 projections) ----
lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
lora_dropout=0.05
# Keep token embeddings / lm_head FROZEN (consistent with CPT). To also adapt the
# Sinhala embeddings during SFT, set: modules_to_save="embed_tokens,lm_head"
modules_to_save=""

# ---- paths (override via env) ----
pretrained_model=${PRETRAINED_MODEL:-../SinLlama_cpt_merged}
tokenizer_name_or_path=${pretrained_model}
# dataset_dir must contain ONLY the training jsonl(s): the harness globs every
# *.jsonl in it as train data. The val file therefore lives one level up.
dataset_dir=${DATASET_DIR:-../data/sft_uc/train}                 # holds train_sft.sinhala.jsonl
validation_file=${VALIDATION_FILE:-../data/sft_uc/val_sft_2k.jsonl}
output_dir=${OUTPUT_DIR:-runs/sft_uc_lora}

# ---- batch / schedule ----
# 192 GB leaves lots of headroom at seq 2048; push per_device higher if you want.
per_device_train_batch_size=16
per_device_eval_batch_size=16
gradient_accumulation_steps=4            # effective batch = 64
max_seq_length=2048
num_train_epochs=1                       # UltraChat SFT is canonically 1 epoch over ~200k

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Only pass --modules_to_save when it is non-empty.
mts_flag=()
[ -n "${modules_to_save}" ] && mts_flag=(--modules_to_save "${modules_to_save}")

torchrun --nnodes 1 --nproc_per_node 1 "${SCRIPT_DIR}/run_clm_sft_with_peft.py" \
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
    --save_steps 500 \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 500 \
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
    --run_name sinllama-sft-uc-lora-r64 \
    --ddp_find_unused_parameters False
