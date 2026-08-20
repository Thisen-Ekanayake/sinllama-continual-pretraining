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

# Parallel dispatch: run up to MAX_PARALLEL units concurrently on one GPU,
# gated on LIVE free VRAM rather than a static per-job guess (job footprint
# varies a lot by model -- llama-3-8b byte-falls-back to much longer Sinhala
# prompts than the SinLlama line, so a fixed estimate would be wrong for one
# or the other). Default 1 preserves the old fully-serial behaviour.
MAX_PARALLEL="${MAX_PARALLEL:-1}"
VRAM_RESERVE_MB="${VRAM_RESERVE_MB:-51200}"     # keep this much free, always
VRAM_LAUNCH_MARGIN_MB="${VRAM_LAUNCH_MARGIN_MB:-20000}"  # + this much slack
                                                  # before greenlighting a NEW
                                                  # launch, since a job's memory
                                                  # keeps growing for ~30-60s
                                                  # after the process starts
                                                  # (model load, then first batch)
VRAM_POLL_SECS="${VRAM_POLL_SECS:-15}"

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

# ---------------------------------------------------------------------------- #
# Live VRAM query, for the parallel dispatcher below. rocm-smi reports total
# GPU usage across ALL processes (this study's own jobs and anything else on
# the box), which is exactly what we want to gate on -- we don't try to
# attribute usage to a specific job, we just never let free memory drop below
# VRAM_RESERVE_MB, full stop.
# ---------------------------------------------------------------------------- #
find_rocm_smi() {
  command -v rocm-smi 2>/dev/null && return
  for c in /opt/rocm/bin/rocm-smi; do [[ -x "$c" ]] && { echo "$c"; return; }; done
}
ROCM_SMI="$(find_rocm_smi)"

vram_free_mb() {
  if [[ -z "$ROCM_SMI" ]]; then echo -1; return; fi   # unknown -> caller must decide
  local out total used
  out="$("$ROCM_SMI" --showmeminfo vram 2>/dev/null)"
  total="$(grep -oE 'Total Memory \(B\): [0-9]+' <<<"$out" | head -1 | grep -oE '[0-9]+$')"
  used="$(grep -oE 'Total Used Memory \(B\): [0-9]+' <<<"$out" | head -1 | grep -oE '[0-9]+$')"
  if [[ -z "$total" || -z "$used" ]]; then echo -1; return; fi
  echo $(( (total - used) / 1024 / 1024 ))
}


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
log "$TOTAL units: ${#_MODEL_ARR[@]} models x langs [$LANGS] x arms [$ARMS]  (max_parallel=$MAX_PARALLEL)"
write_progress

# ---------------------------------------------------------------------------- #
# Per-unit job launcher, shared by the serial and parallel paths.
# ---------------------------------------------------------------------------- #
build_cmd() {
  local lang="$1" arm="$2" model="$3" out="$4"
  local data bs eval_py extra
  case "$lang" in
    sinhala) data=benchmark/mmlu/SinhalaMMLU; bs="$SI_BS"
             eval_py=benchmark/mmlu/evaluate_sinhala_mmlu.py
             extra="--prompt-file benchmark/prompts/mmlu_sinhala.txt";;
    english) data=benchmark/mmlu/english_mmlu; bs="$EN_BS"
             eval_py=benchmark/mmlu/evaluate_english_mmlu.py
             extra="";;
  esac
  if [[ "$arm" == "permuted" ]]; then
    # shellcheck disable=SC2046
    echo "$PY" benchmark/mmlu/permute_eval.py --lang "$lang" --models "$model" \
      --data-root "$data" --out-dir "$out" --batch-size "$bs" --rotations 0 \
      $extra $(opt_list --chat-models $CHAT_MODELS) \
      $(opt_list --alpaca-models $ALPACA_MODELS)
  else
    # shellcheck disable=SC2046
    echo "$PY" "$eval_py" --models "$model" --data-root "$data" --out-dir "$out" \
      --batch-size "$bs" --calibrate \
      $extra $(opt_list --chat-models $CHAT_MODELS) \
      $(opt_list --alpaca-models $ALPACA_MODELS)
  fi
}

read_acc() {
  "$PY" - "$1" <<'PYRD'
import json,sys
d=json.load(open(sys.argv[1])); o=d["overall"]
raw=d.get("raw_arm",{}).get("accuracy")
print(f"{o['accuracy']:.2f}%" + (f" (raw {raw:.2f}%)" if raw is not None else ""))
PYRD
}

finish_unit() {   # called once a unit's process has exited
  local u="$1" rc="$2" t0="$3" name="$4" lang="$5" arm="$6" model="$7" out="$8"
  local dt=$(( $(date +%s) - t0 ))
  if [[ $rc -ne 0 ]] || ! unit_done "$lang" "$arm" "$model"; then
    UNIT_STATE["$u"]=failed
    UNIT_NOTE["$u"]="exit $rc after ${dt}s"
    echo "FAILED: $lang/$arm/$name (exit $rc)" >&2
    rc_overall=1
  else
    local acc; acc=$(read_acc "$out/${name}_metrics.json")
    UNIT_STATE["$u"]=done
    UNIT_NOTE["$u"]="${acc:-ok} in ${dt}s"
    echo "  [$lang/$arm/$name] ${acc:-ok} in ${dt}s"
    upload_unit "$out" "$name"
  fi
  write_progress
}

rc_overall=0
IDX=0

# ---- collect the units that actually need running -------------------------- #
RUN_UNITS=()
for u in "${UNITS[@]}"; do
  IFS='|' read -r lang arm model <<< "$u"
  OUT="$(outdir_for "$lang" "$arm")"
  mkdir -p "$OUT"
  if [[ "$FORCE" != "1" ]] && unit_done "$lang" "$arm" "$model"; then
    UNIT_STATE["$u"]=skipped
    UNIT_NOTE["$u"]="already present"
    echo "  skip (done): $lang/$arm/$(basename "$model")"
  else
    RUN_UNITS+=("$u")
  fi
done
write_progress

if [[ "$MAX_PARALLEL" -le 1 ]]; then
  # -------------------------------------------------------------------------- #
  # Serial path (default). Same behaviour as before parallel support existed.
  # -------------------------------------------------------------------------- #
  for u in "${RUN_UNITS[@]}"; do
    IFS='|' read -r lang arm model <<< "$u"
    name="$(basename "$model")"
    OUT="$(outdir_for "$lang" "$arm")"
    IDX=$((IDX+1))
    log "[$IDX/$TOTAL] $lang / $arm / $name"
    t0=$(date +%s)
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "  DRY: $arm $lang $model"
      UNIT_STATE["$u"]=done; UNIT_NOTE["$u"]="dry"; write_progress; continue
    fi
    cmd="$(build_cmd "$lang" "$arm" "$model" "$OUT")"
    eval "$cmd"; rc=$?
    finish_unit "$u" "$rc" "$t0" "$name" "$lang" "$arm" "$model" "$OUT"
  done
else
  # -------------------------------------------------------------------------- #
  # Parallel path: launch units in the background, gated on LIVE free VRAM.
  # A unit is only launched when free - VRAM_RESERVE_MB >= VRAM_LAUNCH_MARGIN_MB;
  # VRAM_RESERVE_MB is a hard floor, MAX_PARALLEL just caps on top of it. This
  # reads GLOBAL GPU usage, so it coexists safely with a job started outside
  # this script (e.g. a resumed run already in flight).
  # -------------------------------------------------------------------------- #
  declare -A JOB_PID JOB_T0 JOB_NAME JOB_LANG JOB_ARM JOB_MODEL JOB_OUT JOB_LOG
  running=0
  qi=0
  total_run=${#RUN_UNITS[@]}

  reap_one() {   # wait until at least one running job has exited, then finish it
    while :; do
      for u in "${!JOB_PID[@]}"; do
        pid="${JOB_PID[$u]}"
        if ! kill -0 "$pid" 2>/dev/null; then
          wait "$pid"; local rc=$?
          echo "  --- log tail [$u] ---"; tail -3 "${JOB_LOG[$u]}" 2>/dev/null
          finish_unit "$u" "$rc" "${JOB_T0[$u]}" "${JOB_NAME[$u]}" \
            "${JOB_LANG[$u]}" "${JOB_ARM[$u]}" "${JOB_MODEL[$u]}" "${JOB_OUT[$u]}"
          unset 'JOB_PID[$u]' 'JOB_T0[$u]' 'JOB_NAME[$u]' 'JOB_LANG[$u]' \
                'JOB_ARM[$u]' 'JOB_MODEL[$u]' 'JOB_OUT[$u]' 'JOB_LOG[$u]'
          running=$((running-1))
          return
        fi
      done
      sleep "$VRAM_POLL_SECS"
    done
  }

  while (( qi < total_run || running > 0 )); do
    if (( qi < total_run && running < MAX_PARALLEL )); then
      free_mb=$(vram_free_mb)
      if [[ "$free_mb" == "-1" ]]; then
        echo "  WARNING: rocm-smi not found, cannot verify VRAM -- launching without a guard" >&2
        free_mb=999999
      fi
      if (( free_mb - VRAM_RESERVE_MB >= VRAM_LAUNCH_MARGIN_MB )); then
        u="${RUN_UNITS[$qi]}"; qi=$((qi+1))
        IFS='|' read -r lang arm model <<< "$u"
        name="$(basename "$model")"
        OUT="$(outdir_for "$lang" "$arm")"
        IDX=$((IDX+1))
        log "[$IDX/$TOTAL] launching $lang / $arm / $name  (free ${free_mb} MB, reserve $VRAM_RESERVE_MB MB)"
        if [[ "$DRY_RUN" == "1" ]]; then
          echo "  DRY: $arm $lang $model"
          UNIT_STATE["$u"]=done; UNIT_NOTE["$u"]="dry"; write_progress; continue
        fi
        logf="$OUT/${name}.launch.log"
        cmd="$(build_cmd "$lang" "$arm" "$model" "$OUT")"
        eval "$cmd" > "$logf" 2>&1 &
        pid=$!
        JOB_PID["$u"]=$pid; JOB_T0["$u"]=$(date +%s); JOB_NAME["$u"]="$name"
        JOB_LANG["$u"]="$lang"; JOB_ARM["$u"]="$arm"; JOB_MODEL["$u"]="$model"
        JOB_OUT["$u"]="$OUT"; JOB_LOG["$u"]="$logf"
        running=$((running+1))
        # give the new process time to load its model and take a first batch
        # before re-checking VRAM, so two loads don't stack and both overshoot
        # the reserve before either has stabilised
        sleep 60
        continue
      else
        echo "  waiting for VRAM: free ${free_mb} MB, need >= $((VRAM_RESERVE_MB+VRAM_LAUNCH_MARGIN_MB)) MB"
      fi
    fi
    if (( running > 0 )); then
      reap_one
    else
      sleep "$VRAM_POLL_SECS"
    fi
  done
fi

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
