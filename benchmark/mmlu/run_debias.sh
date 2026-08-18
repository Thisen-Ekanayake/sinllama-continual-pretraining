#!/usr/bin/env bash
# benchmark/mmlu/run_debias.sh
#
# Answer-position bias study across the whole SinLlama line. See
# docs/position-bias-study.md.
#
# Three arms per (language, model):
#
#   raw         standard argmax over option logits. NOT run separately -- the
#               permutation arm's rotation-0 pass is bit-identical to it, so it
#               comes free and is measured on THIS pod with THIS stack. That
#               matters: SinLlama_cpt differs by 6-7pp between two runs of the
#               same model because of an sdpa/eager attention bug, so a same-run
#               baseline is the only trustworthy comparison point.
#
#   calibrated  contextual calibration (Zhao et al. 2021): subtract the model's
#               positional prior, measured from an all-"N/A" content-free prompt.
#               Cheap, but assumes the prior is content-independent. On the first
#               two models this assumption FAILED badly -- kept only as a control.
#
#   permuted    full cyclic permutation: score each item n times with the options
#               rotated so every answer text occupies every slot exactly once,
#               then average. Position preference cancels by construction.
#               Cost: n x inference. These are the numbers to report.
#
# Runs ONE MODEL PER INVOCATION so that (a) a crash costs one model rather than a
# whole arm -- the Sinhala permutation arm died once with an uncaught SIGABRT --
# and (b) progress is reported at a useful granularity. Completed units are
# skipped on re-run, so this script is resumable: just run it again.
#
# Usage:
#   bash benchmark/mmlu/run_debias.sh
#   MODELS="models/llama-3-8b" LANGS=sinhala bash benchmark/mmlu/run_debias.sh
#   DRY_RUN=1 bash benchmark/mmlu/run_debias.sh
#   FORCE=1 bash benchmark/mmlu/run_debias.sh     # re-run completed units
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

MODELS="${MODELS:-models/llama-3-8b models/SinLlama_v01 models/SinLlama_cpt models/SinLlama_Bactrianx_Instruct models/SinLlama_v02 models/SinLlama_uc_instruct_cleaned}"
LANGS="${LANGS:-sinhala english}"
ARMS="${ARMS:-permuted calibrated}"
OUT_ROOT="${OUT_ROOT:-benchmark/mmlu}"
BUCKET="${BUCKET:-gs://sinllama_cpt/debias_20260818}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

# Bactrian-X was SFT'd on the Alpaca template and is scored in it everywhere else
# in this repo; scoring it raw would make its numbers incomparable to the
# published tables. uc_instruct_cleaned stays RAW on purpose -- raw is what makes
# it comparable to v02, and its chat arm was already shown to be a null.
ALPACA_MODELS="${ALPACA_MODELS:-SinLlama_Bactrianx_Instruct SinLlama_Backtrianx_instruct}"
CHAT_MODELS="${CHAT_MODELS:-}"

SI_BS="${SI_BS:-16}"
EN_BS="${EN_BS:-16}"
PY="${PY:-python}"

STATUS_FILE="$OUT_ROOT/debias_progress.txt"
START_EPOCH=$(date +%s)

log() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }

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
GSUTIL="$(find_gsutil)"

tag_for() { [[ "$1" == "sinhala" ]] && echo SinhalaMMLU || echo EnglishMMLU; }
outdir_for() { echo "$OUT_ROOT/$(tag_for "$1")_results_$2"; }

# ---------------------------------------------------------------------------- #
# Build the unit list up front so "how much is left" is a real number, not a
# guess. Order: cheap arms first within a language, so partial runs still yield a
# complete cheap arm.
# ---------------------------------------------------------------------------- #
UNITS=()
for lang in $LANGS; do
  for arm in $ARMS; do
    for m in $MODELS; do
      UNITS+=("$lang|$arm|$m")
    done
  done
done
TOTAL=${#UNITS[@]}

declare -A UNIT_STATE UNIT_NOTE
for u in "${UNITS[@]}"; do UNIT_STATE["$u"]=pending; done

unit_done() {   # already has a metrics file?
  local lang="$1" arm="$2" model="$3"
  [[ -s "$(outdir_for "$lang" "$arm")/$(basename "$model")_metrics.json" ]]
}

write_progress() {
  local done=0 failed=0 skipped=0 pending=0
  for u in "${UNITS[@]}"; do
    case "${UNIT_STATE[$u]}" in
      done) done=$((done+1));; failed) failed=$((failed+1));;
      skipped) skipped=$((skipped+1));; *) pending=$((pending+1));;
    esac
  done
  local finished=$((done+failed+skipped))
  local elapsed=$(( $(date +%s) - START_EPOCH ))
  local eta="unknown"
  if (( done > 0 && pending > 0 )); then
    eta="~$(( elapsed / done * pending / 60 )) min"
  elif (( pending == 0 )); then
    eta="-"
  fi
  {
    echo "SinLlama MMLU position-bias study -- progress"
    echo "updated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo
    echo "finished  : $finished / $TOTAL   (done $done, reused $skipped, FAILED $failed)"
    echo "remaining : $pending"
    echo "elapsed   : $(( elapsed / 60 )) min"
    echo "est. left : $eta"
    echo
    printf '%-9s %-11s %-34s %s\n' STATE ARM MODEL NOTE
    printf '%s\n' "----------------------------------------------------------------------------------"
    for u in "${UNITS[@]}"; do
      IFS='|' read -r l a m <<< "$u"
      printf '%-9s %-11s %-34s %s\n' "${UNIT_STATE[$u]}" "$l/$a" "$(basename "$m")" "${UNIT_NOTE[$u]:-}"
    done
  } > "$STATUS_FILE"

  if [[ -n "$BUCKET" && "$DRY_RUN" != "1" && -n "$GSUTIL" && -x "$GSUTIL" ]]; then
    "$GSUTIL" -q cp "$STATUS_FILE" "$BUCKET/" 2>/dev/null \
      || echo "  WARNING: progress upload failed" >&2
  fi
}

upload_unit() {   # push just this model's files, as soon as it finishes
  local outdir="$1" name="$2"
  [[ -z "$BUCKET" || "$DRY_RUN" == "1" ]] && return 0
  if [[ -z "$GSUTIL" || ! -x "$GSUTIL" ]]; then
    echo "  WARNING: gsutil not found; skipped upload" >&2; return 0
  fi
  local dest="$BUCKET/$(basename "$outdir")/"
  local n=0
  for f in "$outdir/$name"_*; do
    [[ -e "$f" ]] || continue
    "$GSUTIL" -q cp "$f" "$dest" 2>/dev/null && n=$((n+1))
  done
  echo "  uploaded $n files -> $dest"
}

# ---------------------------------------------------------------------------- #
# Preflight
# ---------------------------------------------------------------------------- #
fail=0
for m in $MODELS; do
  [[ -f "$m/config.json" ]] || { echo "MISSING MODEL: $m" >&2; fail=1; }
done
for l in $LANGS; do
  d="benchmark/mmlu/$( [[ $l == sinhala ]] && echo SinhalaMMLU || echo english_mmlu )"
  [[ -d "$d" ]] || { echo "MISSING DATA: $d" >&2; fail=1; }
done
if [[ "$fail" == "1" ]]; then
  echo "Run benchmark/mmlu/setup_debias_pod.sh first." >&2
  [[ "$DRY_RUN" != "1" ]] && exit 1
fi

opt_list() { local flag="$1"; shift; [[ -n "${*// /}" ]] && printf '%s %s' "$flag" "$*"; }

read -r -a _MODEL_ARR <<< "$MODELS"
log "$TOTAL units: ${#_MODEL_ARR[@]} models x langs [$LANGS] x arms [$ARMS]"
write_progress

rc_overall=0
IDX=0
for u in "${UNITS[@]}"; do
  IFS='|' read -r lang arm model <<< "$u"
  name="$(basename "$model")"
  OUT="$(outdir_for "$lang" "$arm")"
  mkdir -p "$OUT"

  if [[ "$FORCE" != "1" ]] && unit_done "$lang" "$arm" "$model"; then
    UNIT_STATE["$u"]=skipped
    UNIT_NOTE["$u"]="already present"
    echo "  skip (done): $lang/$arm/$name"
    write_progress
    continue
  fi

  case "$lang" in
    sinhala) DATA=benchmark/mmlu/SinhalaMMLU; BS="$SI_BS"
             EVAL=benchmark/mmlu/evaluate_sinhala_mmlu.py
             EXTRA="--prompt-file benchmark/prompts/mmlu_sinhala.txt";;
    english) DATA=benchmark/mmlu/english_mmlu; BS="$EN_BS"
             EVAL=benchmark/mmlu/evaluate_english_mmlu.py
             EXTRA="";;
  esac

  IDX=$((IDX+1))
  log "[$IDX/$TOTAL] $lang / $arm / $name"
  t0=$(date +%s)

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  DRY: $arm $lang $model"
    UNIT_STATE["$u"]=done; UNIT_NOTE["$u"]="dry"
    write_progress; continue
  fi

  if [[ "$arm" == "permuted" ]]; then
    # shellcheck disable=SC2046
    "$PY" benchmark/mmlu/permute_eval.py --lang "$lang" --models "$model" \
      --data-root "$DATA" --out-dir "$OUT" --batch-size "$BS" --rotations 0 \
      $EXTRA $(opt_list --chat-models $CHAT_MODELS) \
      $(opt_list --alpaca-models $ALPACA_MODELS)
  else
    # shellcheck disable=SC2046
    "$PY" "$EVAL" --models "$model" --data-root "$DATA" --out-dir "$OUT" \
      --batch-size "$BS" --calibrate \
      $EXTRA $(opt_list --chat-models $CHAT_MODELS) \
      $(opt_list --alpaca-models $ALPACA_MODELS)
  fi
  rc=$?
  dt=$(( $(date +%s) - t0 ))

  if [[ $rc -ne 0 ]] || ! unit_done "$lang" "$arm" "$model"; then
    UNIT_STATE["$u"]=failed
    UNIT_NOTE["$u"]="exit $rc after ${dt}s"
    echo "FAILED: $lang/$arm/$name (exit $rc)" >&2
    rc_overall=1
  else
    acc=$("$PY" - "$OUT/${name}_metrics.json" <<'PYEOF' 2>/dev/null
import json,sys
d=json.load(open(sys.argv[1])); o=d["overall"]
raw=d.get("raw_arm",{}).get("accuracy")
print(f"{o['accuracy']:.2f}%" + (f" (raw {raw:.2f}%)" if raw is not None else ""))
PYEOF
)
    UNIT_STATE["$u"]=done
    UNIT_NOTE["$u"]="${acc:-ok} in ${dt}s"
    echo "  ${acc:-ok} in ${dt}s"
    upload_unit "$OUT" "$name"
  fi
  write_progress
done

log "comparing arms"
"$PY" benchmark/mmlu/compare_debias.py --root "$OUT_ROOT" \
  | tee "$OUT_ROOT/debias_summary.txt"
if [[ -n "$BUCKET" && "$DRY_RUN" != "1" && -n "$GSUTIL" && -x "$GSUTIL" ]]; then
  "$GSUTIL" -q cp "$OUT_ROOT/debias_summary.txt" "$BUCKET/" 2>/dev/null
fi
write_progress

echo
cat "$STATUS_FILE"
if [[ $rc_overall -ne 0 ]]; then
  echo
  echo "*** SOME UNITS FAILED -- re-run this script to retry only those." >&2
  exit 1
fi
log "done. summary: $OUT_ROOT/debias_summary.txt"
