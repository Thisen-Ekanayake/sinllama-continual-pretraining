#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Unified entry point for every benchmark task. Dispatches to the per-task,
# per-sub-task shell scripts, which read their hyperparameters from
# main.yml + <task>/config.yml and upload their own results to GCS.
#
# Usage:
#   ./benchmark.sh --task <task> [--sub-task <sub-task>]
#   ./benchmark.sh --task all
#
# Tasks and their sub-tasks:
#   downstream   news | sentiment | writing   (default: all three)
#   mmlu         sinhala | english             (default: both)
#   piqa         sinhala | english             (runs parallel + non-parallel each; default: both)
#   hellaswag    (no sub-tasks)
#
# Examples:
#   ./benchmark.sh --task mmlu --sub-task sinhala
#   ./benchmark.sh --task downstream --sub-task news
#   ./benchmark.sh --task piqa
#   ./benchmark.sh --task all
# ---------------------------------------------------------------------------
set -uo pipefail   # no -e: one task failing must not kill the rest

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: benchmark.sh --task <downstream|hellaswag|mmlu|piqa|all> [--sub-task <name>]

  downstream   sub-tasks: news, sentiment, writing   (omit --sub-task to run all three)
  mmlu         sub-tasks: sinhala, english            (omit --sub-task to run both)
  piqa         sub-tasks: sinhala, english            (omit --sub-task to run both; each runs parallel + non-parallel)
  hellaswag    no sub-tasks
  all          run every task and every sub-task

  -h, --help   show this help and exit
EOF
}

TASK=""
SUB_TASK=""

while [ $# -gt 0 ]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --sub-task) SUB_TASK="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[ -n "$TASK" ] || { echo "error: --task is required" >&2; usage; exit 1; }

declare -a FAILED=()

run_step() {
  # run_step <label> <script> [args...]
  local label="$1"; shift
  local script="$1"; shift
  echo
  echo "==================================================================="
  echo "[$(date '+%F %T')] >>> $label"
  echo "==================================================================="
  if ! bash "$script" "$@"; then
    echo "[$(date '+%F %T')] !!! $label FAILED"
    FAILED+=("$label")
  fi
}

run_downstream() {
  local sub="$1"
  case "$sub" in
    news)      run_step "downstream/news"      "$SCRIPT_DIR/downstream/run_news.sh" ;;
    sentiment) run_step "downstream/sentiment" "$SCRIPT_DIR/downstream/run_sentiment.sh" ;;
    writing)   run_step "downstream/writing"   "$SCRIPT_DIR/downstream/run_writing.sh" ;;
    all|"")
      run_downstream news
      run_downstream sentiment
      run_downstream writing
      ;;
    *) echo "error: unknown downstream sub-task '$sub' (expected news|sentiment|writing)" >&2; exit 1 ;;
  esac
}

run_mmlu() {
  local sub="$1"
  case "$sub" in
    sinhala) run_step "mmlu/sinhala" "$SCRIPT_DIR/mmlu/run_sinhala.sh" ;;
    english) run_step "mmlu/english" "$SCRIPT_DIR/mmlu/run_english.sh" ;;
    all|"")
      run_mmlu sinhala
      run_mmlu english
      ;;
    *) echo "error: unknown mmlu sub-task '$sub' (expected sinhala|english)" >&2; exit 1 ;;
  esac
}

run_piqa() {
  local sub="$1"
  case "$sub" in
    sinhala) run_step "piqa/sinhala" "$SCRIPT_DIR/mrlbenchmarks/run_sinhala.sh" ;;
    english) run_step "piqa/english" "$SCRIPT_DIR/mrlbenchmarks/run_english.sh" ;;
    all|"")
      run_piqa sinhala
      run_piqa english
      ;;
    *) echo "error: unknown piqa sub-task '$sub' (expected sinhala|english)" >&2; exit 1 ;;
  esac
}

run_hellaswag() {
  [ -z "$1" ] || { echo "error: hellaswag has no sub-tasks" >&2; exit 1; }
  run_step "hellaswag" "$SCRIPT_DIR/hellaswag/run_hellaswag.sh"
}

case "$TASK" in
  downstream) run_downstream "$SUB_TASK" ;;
  mmlu)       run_mmlu "$SUB_TASK" ;;
  piqa)       run_piqa "$SUB_TASK" ;;
  hellaswag)  run_hellaswag "$SUB_TASK" ;;
  all)
    run_downstream all
    run_mmlu all
    run_piqa all
    run_hellaswag ""
    ;;
  *) echo "error: unknown task '$TASK'" >&2; usage; exit 1 ;;
esac

echo
echo "==================================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "[$(date '+%F %T')] ALL REQUESTED BENCHMARKS COMPLETED"
else
  echo "[$(date '+%F %T')] COMPLETED WITH FAILURES: ${FAILED[*]}"
  exit 1
fi
