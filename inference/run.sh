#!/usr/bin/env bash
# Ad-hoc generation-quality inference.
#
#   bash inference/run.sh --model SinLlama_uc_instruct_cleaned
#   bash inference/run.sh --model SinLlama_v02 --prompt "..." --preset creative
#   bash inference/run.sh --model SinLlama_Bactrianx_Instruct --out results.txt
#
#   --out <path.txt>   write generations to this file instead of the terminal
#                       (default: print to the terminal)
#
# Everything else (which prompts, which model, which generation preset) is a
# flag on inference/run_inference.py -- run with --help for the full list.
# Bf16 + sdpa only, no quantization. Target hardware: a single AMD MI300X.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"

export TOKENIZERS_PARALLELISM=false
# Same ROCm allocator/BLAS tuning as gen/run_gen_sft.sh -- harmless for a
# single generate() call, but keeps every launcher in the repo consistent.
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-garbage_collection_threshold:0.8}"
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-1}"

cd "${REPO_ROOT}"
"${PYTHON}" inference/run_inference.py "$@"
