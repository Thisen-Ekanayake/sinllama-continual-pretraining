#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run all 4 Global-PIQA few-shot eval scripts LOCALLY on CPU, one at a time —
# Sinhala first (both datasets), then English (both datasets). No pod, no scp.
#
# Each script holds out 8 balanced few-shot exemplars (2/2/2/2 for the 4-way
# parallel sets, 4/4 for the 2-way non-parallel sets), excludes them from the
# test set, and loads each model in float32 on CPU.
#
# NOTE: CPU inference on 8B models is slow — the Sinhala sets in particular
# (base Llama byte-falls on Sinhala → long prompts). Expect this to run for a
# while. Override knobs via env: BATCH_SIZE, MAX_LEN, KSHOT, MODELS.
#
# Usage: bash run_local_eval.sh
# ---------------------------------------------------------------------------
set -uo pipefail          # no -e — one script failing must not kill the rest

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/local_logs"
mkdir -p "$LOG_DIR"

# Prefer the project venv's python if present, else whatever python3 is on PATH.
PY="python3"
if [ -x "$SCRIPT_DIR/../.venv/bin/python" ]; then
  PY="$SCRIPT_DIR/../.venv/bin/python"
fi

BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LEN="${MAX_LEN:-2048}"
KSHOT="${KSHOT:-8}"

COMMON_ARGS=(--device cpu --kshot "$KSHOT" --batch-size "$BATCH_SIZE" --max-len "$MAX_LEN")

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
echo "[$(date '+%F %T')] Running $TOTAL few-shot eval scripts locally on CPU ($PY)"
echo "[$(date '+%F %T')] args: ${COMMON_ARGS[*]}"

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
  "$PY" -u "$script" "${COMMON_ARGS[@]}" --out-dir "$outdir" \
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
echo "[$(date '+%F %T')] Building combined results.md across all 4 benchmarks"
"$PY" -u combine_all_results.py 2>&1 | tee "$LOG_DIR/combine.log" || true

echo
echo "[$(date '+%F %T')] ALL DONE in $(fmt_secs $(( SECONDS - overall_start ))). Results:"
for run in "${RUNS[@]}"; do
  outdir="${run#* }"
  echo "  $SCRIPT_DIR/$outdir/"
done
echo "  $SCRIPT_DIR/results.md   (combined)"
