#!/usr/bin/env bash
# Instruct-mix QLoRA run on the lab box (~/sinllama-continual-pretraining/instruct_sft).
#   smoke:  OUT=runs/smoke_r64 ./run_instruct.sh --smoke
#   full:   setsid nohup ./run_instruct.sh > runs/r64.run.log 2>&1 < /dev/null &
# Up to 3 attempts; each attempt resumes from the last checkpoint in $OUT/ckpt and the same W&B run.
# Markers: $OUT/TRAIN_DONE or $OUT/FAILED. Extra arguments go to train_instruct.py.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/tts_env/bin/python}
OUT=${OUT:-runs/r64}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# torch 2.6 defaults torch.load to weights_only=True, which rejects the numpy RNG state in our own checkpoints on resume
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$OUT"; rm -f "$OUT/FAILED"
[ -e "$OUT/TRAIN_DONE" ] && { echo "already done"; exit 0; }
for attempt in 1 2 3; do
  echo "$(date) attempt $attempt"
  if $PY train_instruct.py --out "$OUT" "$@" >> "$OUT/train.log" 2>&1; then
    touch "$OUT/TRAIN_DONE"; echo "$(date) done"; exit 0
  fi
  echo "$(date) attempt $attempt failed (tail of train.log):"; tail -n 5 "$OUT/train.log"
  case "$*" in *--smoke*) break ;; esac
  sleep 60
done
touch "$OUT/FAILED"; exit 1
