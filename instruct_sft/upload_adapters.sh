#!/usr/bin/env bash
# Uploads the instruct-mix adapters to Google Drive (lab box ~/bin/rclone) while/after run_instruct.sh trains.
#   OUT=runs/r64 setsid nohup ./upload_adapters.sh > runs/r64.upload.log 2>&1 < /dev/null &
# - early stopping hit mid-epoch ($OUT/early_stop/READY): upload it right away -> $DEST/early_stop_step<S>_best<B>,
#   then at TRAIN_DONE upload the 1-epoch adapter $OUT/final -> $DEST/epoch1_final_step<N>
#   (and $OUT/best -> $DEST/epoch1_best_step<B> if the best moved after the early-stop point).
# - never hit: at TRAIN_DONE upload only the best adapter $OUT/best -> $DEST/best_step<B> (no early stop).
# Idempotent: markers $OUT/UPLOADED_* ; finishes with $OUT/UPLOAD_DONE or $OUT/UPLOAD_FAILED.
set -uo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-runs/r64}
DEST=${DEST:-gdrive:sinllama_cpt/vqa-early-stopping}
PY=${PY:-$HOME/tts_env/bin/python}
RC=~/bin/rclone
rm -f "$OUT/UPLOAD_FAILED"

up() {  # up <local dir> <remote subdir> <marker>
  [ -e "$OUT/UPLOADED_$3" ] && return 0
  for i in 1 2 3 4 5; do
    echo "$(date) uploading $1 -> $DEST/$2 (try $i)"
    if $RC copy "$1" "$DEST/$2" --retries 5 --low-level-retries 20 --stats-one-line --stats 60s \
       && $RC check "$1" "$DEST/$2" --one-way; then
      touch "$OUT/UPLOADED_$3"; echo "$(date) uploaded $2"; return 0
    fi
    sleep 120
  done
  echo "$(date) upload of $1 FAILED"; touch "$OUT/UPLOAD_FAILED"; return 1
}
step_of() { $PY -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]].rsplit('-',1)[1])" "$@"; }

while :; do
  if [ -e "$OUT/early_stop/READY" ] && [ ! -e "$OUT/UPLOADED_early_stop" ]; then
    S=$($PY -c "import json;print(json.load(open('$OUT/early_stop/early_stop.json'))['early_stop_at_step'])")
    B=$($PY -c "import json;print(json.load(open('$OUT/early_stop/early_stop.json'))['best_step'])")
    up "$OUT/early_stop" "early_stop_step${S}_best${B}" early_stop
  fi
  [ -e "$OUT/FAILED" ] && { echo "$(date) training FAILED; uploader exits"; exit 1; }
  if [ -e "$OUT/TRAIN_DONE" ]; then
    cp "$OUT/train_state.json" "$OUT/best/" 2>/dev/null
    B=$(step_of "$OUT/train_state.json" best_checkpoint)
    N=$($PY -c "import json;print(json.load(open('$OUT/train_state.json'))['global_step'])")
    if [ -e "$OUT/early_stop/READY" ]; then
      cp "$OUT/train_state.json" "$OUT/final/"
      up "$OUT/final" "epoch1_final_step$N" final || exit 1
      ES_B=$($PY -c "import json;print(json.load(open('$OUT/early_stop/early_stop.json'))['best_step'])")
      [ "$B" != "$ES_B" ] && { up "$OUT/best" "epoch1_best_step$B" best || exit 1; }
    else
      up "$OUT/best" "best_step${B}_no_early_stop" best || exit 1
    fi
    touch "$OUT/UPLOAD_DONE"; echo "$(date) UPLOAD DONE"; $RC lsl "$DEST"; exit 0
  fi
  sleep 300
done
