#!/usr/bin/env bash
# Instruction SFT of SinLlama_v02 on UltraChat-Sinhala (sft split, multi-turn).
#
# Every hyperparameter and path lives in sft/config.yaml; this script only sets
# the environment and launches the trainer.
#
#   bash sft/run_sft_uc.sh                          # full run
#   bash sft/run_sft_uc.sh --set train.learning_rate=5e-5
#   CONFIG=sft/config.yaml bash sft/run_sft_uc.sh
#   SKIP_FETCH=1 bash sft/run_sft_uc.sh             # model already on disk
#
# Target hardware: a single AMD MI300X (192 GB, ROCm) — bf16 LoRA, no
# bitsandbytes, sdpa attention, no gradient checkpointing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config.yaml}"
PYTHON="${PYTHON:-python3}"

export TOKENIZERS_PARALLELISM=false
# ROCm allocator. NOT expandable_segments: this ROCm build prints
#   "expandable_segments not supported on this platform"
# and ignores it, so freed blocks are never coalesced and reserved memory
# ratchets upward as batch shapes vary (dynamic padding + group_by_length).
# garbage_collection_threshold makes the allocator release cached blocks once
# reserved passes 80% of VRAM. Add max_split_size_mb:512 if OOMs persist.
export PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-garbage_collection_threshold:0.8}
# hipBLASLt beats rocBLAS on MI300X for the shapes this model runs; AMD's
# workload-tuning guide recommends forcing it rather than leaving it to the
# heuristic.
export TORCH_BLAS_PREFER_HIPBLASLT=${TORCH_BLAS_PREFER_HIPBLASLT:-1}
# TunableOp searches rocBLAS/hipBLASLt candidates per GEMM shape and pins the
# winner. AMD measures 6-8% on MI300X. The tuning pass makes the FIRST run
# slower, then results are cached in the csv and reused. Opt in with TUNE=1.
if [ -n "${TUNE:-}" ]; then
  export PYTORCH_TUNABLEOP_ENABLED=1
  export PYTORCH_TUNABLEOP_FILENAME="${PYTORCH_TUNABLEOP_FILENAME:-${SCRIPT_DIR}/runs/tunableop_results.csv}"
  mkdir -p "$(dirname "${PYTORCH_TUNABLEOP_FILENAME}")"
fi
export WANDB_PROJECT="${WANDB_PROJECT:-$("${PYTHON}" - "${CONFIG}" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["train"].get("wandb_project", "sinllama-sft-uc"))
PY
)}"

[ -n "${SKIP_FETCH:-}" ] || bash "${SCRIPT_DIR}/fetch_model.sh" "${CONFIG}"

torchrun --nnodes 1 --nproc_per_node "${NPROC:-1}" "${SCRIPT_DIR}/run_sft_uc.py" \
    --config "${CONFIG}" \
    "$@"
