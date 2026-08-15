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
# ROCm allocator: reduce fragmentation OOMs (analogous to PYTORCH_CUDA_ALLOC_CONF).
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-$("${PYTHON}" - "${CONFIG}" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["train"].get("wandb_project", "sinllama-sft-uc"))
PY
)}"

[ -n "${SKIP_FETCH:-}" ] || bash "${SCRIPT_DIR}/fetch_model.sh" "${CONFIG}"

torchrun --nnodes 1 --nproc_per_node "${NPROC:-1}" "${SCRIPT_DIR}/run_sft_uc.py" \
    --config "${CONFIG}" \
    "$@"
