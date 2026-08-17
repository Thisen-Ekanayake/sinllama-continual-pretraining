#!/usr/bin/env bash
# benchmark/mmlu/run_debias.sh
#
# Position-bias study for the SinLlama line: is the SinhalaMMLU gap between
# SinLlama_v02 and SinLlama_uc_instruct_cleaned real knowledge loss, or a shift
# in which option slot the model prefers?
#
# Three arms per (language, model):
#
#   raw         standard argmax over option logits. NOT run separately -- the
#               permutation arm's rotation-0 pass is bit-identical to it, so it
#               comes free and, importantly, is measured on THIS pod with THIS
#               stack. That matters: the SinLlama_cpt row in the last report
#               differs by 6-7pp between two runs of the same model because of
#               an sdpa/eager attention bug, so a same-run baseline is the only
#               trustworthy comparison point.
#
#   calibrated  contextual calibration (Zhao et al. 2021): subtract the model's
#               positional prior, measured from an all-"N/A" content-free prompt.
#               Cost: ~1 extra forward pass. Assumes the prior is independent of
#               question content.
#
#   permuted    full cyclic permutation: score each item n times with the options
#               rotated so every answer text occupies every slot exactly once,
#               then average. Position preference cancels by construction and no
#               independence assumption is needed. Cost: n x inference.
#
# Agreement between `calibrated` and `permuted` is the evidence that the
# debiasing is sound; disagreement means the prior is content-dependent and only
# `permuted` should be trusted.
#
# Usage:
#   bash benchmark/mmlu/run_debias.sh                 # both languages
#   LANGS=sinhala bash benchmark/mmlu/run_debias.sh   # one language
#   DRY_RUN=1 bash benchmark/mmlu/run_debias.sh       # print commands only
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# --- what to run ------------------------------------------------------------ #
# The two models the question is actually about. Add models/SinLlama_v01 to also
# cover the v01->v02 (MMLU-Sinhala SFT) step.
MODELS="${MODELS:-models/SinLlama_v02 models/SinLlama_uc_instruct_cleaned}"
LANGS="${LANGS:-sinhala english}"
ARMS="${ARMS:-permuted calibrated}"
OUT_ROOT="${OUT_ROOT:-benchmark/mmlu}"
BUCKET="${BUCKET:-gs://sinllama_cpt/debias_$(date +%Y%m%d)}"
DRY_RUN="${DRY_RUN:-0}"

# Scored RAW on purpose -- uc_instruct_cleaned's chat template was already shown
# to change which items are right without changing how many (11.5% churn, net
# -0.39pp, p=0.18), and raw is what makes it comparable to v02. Set
# CHAT_MODELS=SinLlama_uc_instruct to score the chat arm instead.
CHAT_MODELS="${CHAT_MODELS:-}"
ALPACA_MODELS="${ALPACA_MODELS:-}"

SI_BS="${SI_BS:-16}"
EN_BS="${EN_BS:-16}"

PY="${PY:-python}"

# same search as setup_debias_pod.sh -- the SDK is never on a non-interactive
# PATH and has moved between pods
find_gsutil() {
  local c
  c="$(command -v gsutil 2>/dev/null)" && [[ -x "$c" ]] && { echo "$c"; return; }
  for c in "$HOME/google-cloud-sdk/bin/gsutil" \
           "$REPO_DIR/google-cloud-sdk/bin/gsutil" \
           "/usr/lib/google-cloud-sdk/bin/gsutil" \
           "/snap/bin/gsutil"; do
    [[ -x "$c" ]] && { echo "$c"; return; }
  done
  c="$(find "$HOME" -maxdepth 4 -name gsutil -type f -perm -u+x 2>/dev/null | head -1)"
  [[ -n "$c" ]] && echo "$c"
}

log() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
run() {
  if [[ "$DRY_RUN" == "1" ]]; then printf '  DRY: %s\n' "$*"; return 0; fi
  "$@"
}

# --- preflight -------------------------------------------------------------- #
fail=0
for m in $MODELS; do
  if [[ ! -f "$m/config.json" ]]; then
    echo "MISSING MODEL: $m (run benchmark/mmlu/setup_debias_pod.sh first)" >&2
    fail=1
  fi
done
for l in $LANGS; do
  case "$l" in
    sinhala) d="benchmark/mmlu/SinhalaMMLU";;
    english) d="benchmark/mmlu/english_mmlu";;
  esac
  if [[ ! -d "$d" ]]; then
    echo "MISSING DATA: $d (run benchmark/mmlu/setup_debias_pod.sh first)" >&2
    fail=1
  fi
done
[[ "$fail" == "1" && "$DRY_RUN" != "1" ]] && exit 1

opt_list() {  # emit "--flag a b" only when non-empty, so we never pass a bare flag
  local flag="$1"; shift
  [[ -n "${*// /}" ]] && printf '%s %s' "$flag" "$*"
}

rc_overall=0

for lang in $LANGS; do
  case "$lang" in
    sinhala)
      DATA="benchmark/mmlu/SinhalaMMLU"; BS="$SI_BS"
      EVAL="benchmark/mmlu/evaluate_sinhala_mmlu.py"
      EXTRA="--prompt-file benchmark/prompts/mmlu_sinhala.txt"
      TAG="SinhalaMMLU";;
    english)
      DATA="benchmark/mmlu/english_mmlu"; BS="$EN_BS"
      EVAL="benchmark/mmlu/evaluate_english_mmlu.py"
      EXTRA=""
      TAG="EnglishMMLU";;
    *) echo "unknown lang: $lang" >&2; continue;;
  esac

  for arm in $ARMS; do
    OUT="$OUT_ROOT/${TAG}_results_${arm}"
    log "$lang / $arm  ->  $OUT"
    mkdir -p "$OUT"

    if [[ "$arm" == "permuted" ]]; then
      # shellcheck disable=SC2046
      run "$PY" benchmark/mmlu/permute_eval.py \
        --lang "$lang" \
        --models $MODELS \
        --data-root "$DATA" \
        --out-dir "$OUT" \
        --batch-size "$BS" \
        --rotations 0 \
        $EXTRA \
        $(opt_list --chat-models $CHAT_MODELS) \
        $(opt_list --alpaca-models $ALPACA_MODELS)
    else
      # shellcheck disable=SC2046
      run "$PY" "$EVAL" \
        --models $MODELS \
        --data-root "$DATA" \
        --out-dir "$OUT" \
        --batch-size "$BS" \
        --calibrate \
        $EXTRA \
        $(opt_list --chat-models $CHAT_MODELS) \
        $(opt_list --alpaca-models $ALPACA_MODELS)
    fi
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "FAILED: $lang/$arm (exit $rc)" >&2
      rc_overall=1
      continue
    fi

    if [[ -n "$BUCKET" && "$DRY_RUN" != "1" ]]; then
      GSUTIL="$(find_gsutil)"
      if [[ -n "$GSUTIL" && -x "$GSUTIL" ]]; then
        "$GSUTIL" -m -q cp -r "$OUT" "$BUCKET/" \
          && echo "  uploaded -> $BUCKET/$(basename "$OUT")" \
          || echo "  WARNING: upload failed for $OUT" >&2
      else
        echo "  WARNING: gsutil not found; skipped upload of $OUT" >&2
      fi
    fi
  done
done

log "comparing arms"
run "$PY" benchmark/mmlu/compare_debias.py --root "$OUT_ROOT" \
  | tee "$OUT_ROOT/debias_summary.txt"

if [[ $rc_overall -ne 0 ]]; then
  echo
  echo "*** ONE OR MORE ARMS FAILED -- see errors above. Do not read the"
  echo "*** summary as complete." >&2
  exit 1
fi
log "done. summary: $OUT_ROOT/debias_summary.txt"
