#!/usr/bin/env bash
# Stage-2 SFT: SinLlama_uc_instruct_cleaned on the bilingual UltraChat `gen` mix.
#
#   bash gen/run_gen_sft.sh                          # fetch, build, train
#   bash gen/run_gen_sft.sh --set train.learning_rate=3e-5
#   SKIP_FETCH=1 bash gen/run_gen_sft.sh             # model + data already there
#   SKIP_BUILD=1 bash gen/run_gen_sft.sh             # mix already built
#   SMOKE=1 bash gen/run_gen_sft.sh                  # 2k dialogues, eval every 20 steps
#
# Everything else lives in gen/config.yaml; this only sets the environment and
# sequences fetch -> build -> train. The trainer itself is sft/run_sft_uc.py,
# unforked: it is config-driven and dataset-agnostic, so this stage needs no
# copy of it. See gen/config.yaml for why this stage exists.
#
# Target hardware: a single AMD MI300X (192 GB, ROCm) — bf16 LoRA, no
# bitsandbytes, sdpa attention, no gradient checkpointing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config.yaml}"
PYTHON="${PYTHON:-python3}"

export TOKENIZERS_PARALLELISM=false
# ROCm allocator. NOT expandable_segments: this ROCm build prints
# "expandable_segments not supported on this platform" and ignores it, so freed
# blocks are never coalesced and reserved memory ratchets upward as batch shapes
# vary (dynamic padding + group_by_length). garbage_collection_threshold makes
# the allocator release cached blocks once reserved passes 80% of VRAM.
export PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-garbage_collection_threshold:0.8}
# hipBLASLt beats rocBLAS on MI300X for this model's shapes.
export TORCH_BLAS_PREFER_HIPBLASLT=${TORCH_BLAS_PREFER_HIPBLASLT:-1}
# TunableOp pins the best kernel per GEMM shape; AMD measures 6-8% on MI300X.
# The tuning pass makes the FIRST run slower, then the csv is reused. Opt in.
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

if [ -z "${SKIP_FETCH:-}" ]; then
  bash "${REPO_ROOT}/sft/fetch_model.sh" "${CONFIG}"
  bash "${SCRIPT_DIR}/fetch_data.sh"
fi

if [ -z "${SKIP_BUILD:-}" ]; then
  "${PYTHON}" "${SCRIPT_DIR}/build_mixed_gen.py" --config "${CONFIG}"
fi

SMOKE_ARGS=()
if [ -n "${SMOKE:-}" ]; then
  # End-to-end shakeout on the real GPU path in a few minutes: bf16 + sdpa,
  # loss decreasing, BOTH eval_si_loss and eval_en_loss firing, checkpoints
  # written, wandb live.
  SMOKE_ARGS=(--set data.max_train_samples=2000
              --set train.eval_steps=20
              --set train.save_steps=20
              --set train.num_train_epochs=1
              --set train.output_dir=gen/runs/smoke)
  echo "[$(date '+%F %T')] SMOKE mode: ${SMOKE_ARGS[*]}"
fi

torchrun --nnodes 1 --nproc_per_node "${NPROC:-1}" "${REPO_ROOT}/sft/run_sft_uc.py" \
    --config "${CONFIG}" \
    "${SMOKE_ARGS[@]}" \
    "$@"
