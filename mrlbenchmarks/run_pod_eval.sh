#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run all 4 Global-PIQA few-shot eval scripts on a GPU cloud pod (bf16, sdpa)
# — Sinhala first (both datasets), then English (both datasets) — then scp each
# results_* directory back to this local mrlbenchmarks/.
#
# Each script holds out KSHOT (default 8) balanced few-shot exemplars and
# excludes them from the test set. Set KSHOT=0 for a zero-shot run.
#
# Connection is parameterized via env vars — set these for YOUR pod first:
#   POD_HOST   pod IP/hostname                (required)
#   POD_PORT   ssh port                       (default 22)
#   POD_KEY    ssh private key                (default ~/.ssh/id_ed25519)
#   POD_DIR    mrlbenchmarks path on the pod  (default below)
#   POD_VENV   venv activate path on the pod  (default ../venv/bin/activate)
# Eval knobs (optional): BATCH_SIZE, MAX_LEN, KSHOT.
#
# Usage:  POD_HOST=1.2.3.4 POD_PORT=22594 bash run_pod_eval.sh
# ---------------------------------------------------------------------------
set -uo pipefail          # no -e — one script failing must not kill the rest

POD_HOST="${POD_HOST:?set POD_HOST=<pod ip/hostname> (see header)}"
POD_PORT="${POD_PORT:-22}"
POD_KEY="${POD_KEY:-$HOME/.ssh/id_ed25519}"
POD_DIR="${POD_DIR:-/root/sinllama-continual-pretraining/mrlbenchmarks}"
POD_VENV="${POD_VENV:-../venv/bin/activate}"

BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LEN="${MAX_LEN:-2048}"
KSHOT="${KSHOT:-8}"
EVAL_ARGS="--quant bf16 --kshot $KSHOT --batch-size $BATCH_SIZE --max-len $MAX_LEN"

# -tt forces a pseudo-terminal on the remote end, so the remote python sees an
# interactive tty and its tqdm progress bars (model-shard loading, eval loop)
# stream live instead of block-buffering and dumping only at the end.
SSH=(ssh -tt -p "$POD_PORT" -i "$POD_KEY" "root@$POD_HOST")
SCP=(scp -P "$POD_PORT" -i "$POD_KEY")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/pod_logs"
mkdir -p "$LOG_DIR"

# (script, out-dir) pairs — Sinhala pair first, then the English pair.
RUNS=(
  "evaluate_piqa_parallel_sinhala.py results_parallel_sinhala"
  "evaluate_piqa_nonparallel_sinhala.py results_nonparallel_sinhala"
  "evaluate_piqa_parallel_english.py results_parallel_english"
  "evaluate_piqa_nonparallel_english.py results_nonparallel_english"
)
TOTAL=${#RUNS[@]}

fmt_secs() { printf '%dm%02ds' $(( $1 / 60 )) $(( $1 % 60 )); }

overall_start=$SECONDS
echo "[$(date '+%F %T')] Running $TOTAL few-shot eval scripts on $POD_HOST (one at a time, bf16 GPU, sdpa)"
echo "[$(date '+%F %T')] eval args: $EVAL_ARGS"

i=0
for run in "${RUNS[@]}"; do
  i=$(( i + 1 ))
  script="${run%% *}"
  outdir="${run#* }"
  echo
  echo "==================================================================="
  echo "[$(date '+%F %T')] [$i/$TOTAL] $script  ->  $outdir/  (log: $LOG_DIR/${outdir}.log)"
  echo "==================================================================="
  step_start=$SECONDS
  "${SSH[@]}" "cd $POD_DIR && source $POD_VENV && python -u $script $EVAL_ARGS --out-dir $outdir" \
    2>&1 | tee "$LOG_DIR/${outdir}.log"
  status="${PIPESTATUS[0]}"
  elapsed=$(( SECONDS - step_start ))
  if [ "$status" -ne 0 ]; then
    echo "[$(date '+%F %T')] WARNING: $script exited with status $status after $(fmt_secs "$elapsed") — continuing"
  else
    echo "[$(date '+%F %T')] [$i/$TOTAL] done in $(fmt_secs "$elapsed")"
  fi
done

echo
echo "[$(date '+%F %T')] All remote runs finished in $(fmt_secs $(( SECONDS - overall_start ))). Downloading results via scp..."

i=0
for run in "${RUNS[@]}"; do
  i=$(( i + 1 ))
  outdir="${run#* }"
  echo "[$i/$TOTAL] scp $outdir/ ..."
  "${SCP[@]}" -r "root@$POD_HOST:$POD_DIR/$outdir" "$SCRIPT_DIR/"
done

echo
echo "[$(date '+%F %T')] ALL DONE in $(fmt_secs $(( SECONDS - overall_start ))). Results downloaded to:"
for run in "${RUNS[@]}"; do
  outdir="${run#* }"
  echo "  $SCRIPT_DIR/$outdir/"
done
