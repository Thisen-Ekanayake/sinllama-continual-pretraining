#!/usr/bin/env bash
# COCO train2017 VQA (English, Qwen3.5-9B) + Sinhala (NLLB-200 3.3B), in chunks, unattended on the lab box.
#
# Per chunk of CHUNK images (sorted file order), sequentially because Qwen and NLLB do not fit on the
# 16 GB GPU together:
#   1. llama-server up -> generate_vqa.py (same prompts as the val2017 VQA set, COCO captions as
#      grounding) -> en_XX.json; up to 4 passes, then the server is stopped
#   2. translate_nllb.py --nopunct-first --logprob   (NLLB, GPU)  -> si_XX.raw.cache.jsonl
#   3. repair_nllb.py                                (NLLB, GPU)  -> si_XX.raw.cache.repair.jsonl
#   4. postprocess_si.py --fix-split-hal             (CPU)        -> si_XX.json
# A chunk is finished when chunks/XX.done exists; rerunning skips finished chunks and every step
# resumes from its own cache. At the end all chunks are merged into vqa_train2017_{english,sinhala}.json.
#
# Launch:  setsid nohup ./run_train2017.sh > run.log 2>&1 < /dev/null &
# Smoke:   N_IMAGES=20 CHUNK=10 WORK=smoke ./run_train2017.sh
set -uo pipefail
cd "$(dirname "$0")"
HERE=$PWD
WORK=${WORK:-chunks}
CHUNK=${CHUNK:-10000}
IMAGES=${IMAGES:-$HOME/coco/train2017}
CAPTIONS=${CAPTIONS:-$HOME/coco/annotations/captions_train2017.json}
NLLB=${NLLB:-$HOME/nllb_fp16}
PY=${PY:-$HOME/tts_env/bin/python}
LX="--lexicon $HERE/sinhala_lexicon.pkl --vocab $HERE/unigram_32000_0.9995.vocab"
export LOG=$HERE/server.log
mkdir -p "$WORK"
rm -f ALL_DONE FAILED

total=$(ls "$IMAGES" | grep -c '\.jpg$')
N_IMAGES=${N_IMAGES:-$total}
nchunks=$(( (N_IMAGES + CHUNK - 1) / CHUNK ))
echo "start $(date): $N_IMAGES images in $nchunks chunks of $CHUNK -> $WORK/"

for (( c=0; c<nchunks; c++ )); do
  X=$(printf "%02d" $c)
  start=$(( c * CHUNK )); lim=$(( N_IMAGES - start < CHUNK ? N_IMAGES - start : CHUNK ))
  [ -f "$WORK/$X.done" ] && { echo "== chunk $X done already"; continue; }
  echo "== chunk $X: images $start..$((start + lim - 1)) $(date)"

  # 1. English generation
  for pass in 1 2 3 4; do
    ./start_server.sh || { pkill -x llama-server; sleep 5; ./start_server.sh || continue; }
    python3 generate_vqa.py --images-dir "$IMAGES" --captions "$CAPTIONS" --start $start --limit $lim \
      --out "$WORK/en_$X.json" 2>&1 | grep -E "^(grounding|[0-9]+ images|wrote|\[error\])|000/" | tail -n 3
    n=$(python3 -c "import json;print(len(json.load(open('$WORK/en_$X.json'))))" 2>/dev/null || echo 0)
    [ "$n" -ge "$lim" ] && break
    echo "   pass $pass: $n/$lim, restarting the server"; pkill -x llama-server; sleep 5
  done
  pkill -x llama-server; sleep 5
  echo "   english: $n/$lim records $(date)"
  [ "$n" -ge "$lim" ] || { echo "chunk $X: generation incomplete" > FAILED; exit 1; }

  # 2.-4. Sinhala
  ok=0
  # full step output goes to $WORK/si_XX.log; only the summary lines are echoed here
  SL="$WORK/si_$X.log"
  for pass in 1 2 3; do
    $PY translate_nllb.py --inp "$WORK/en_$X.json" --out "$WORK/si_$X.raw.json" --model "$NLLB" \
        --nopunct-first --logprob $LX >> "$SL" 2>&1 \
      && $PY repair_nllb.py --cache "$WORK/si_$X.raw.cache.jsonl" --model "$NLLB" >> "$SL" 2>&1 \
      && $PY postprocess_si.py --en "$WORK/en_$X.json" --out "$WORK/si_$X.json" --fix-split-hal $LX \
        --cache "$WORK/si_$X.raw.cache.jsonl" "$WORK/si_$X.raw.cache.repair.jsonl" >> "$SL" 2>&1 \
      && [ -s "$WORK/si_$X.json" ] && { ok=1; break; }
    echo "   translation pass $pass failed $(date), see $SL"; sleep 30
  done
  grep -E "^(wrote|repaired|ZWJ merges|rare letters|latin gaps)" "$SL" | tail -n 6 | sed 's/^/   /'
  [ $ok = 1 ] || { echo "chunk $X: translation failed" > FAILED; exit 1; }
  touch "$WORK/$X.done"
  echo "   chunk $X finished $(date)"
done

# merge
python3 - "$WORK" <<'EOF'
import glob, json, sys
w = sys.argv[1]
for lang, pat in (("english", "en_[0-9][0-9].json"), ("sinhala", "si_[0-9][0-9].json")):
    recs = [r for f in sorted(glob.glob(f"{w}/{pat}")) for r in json.load(open(f, encoding="utf-8"))]
    json.dump(recs, open(f"{w}/vqa_train2017_{lang}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"{lang}: {len(recs)} records -> {w}/vqa_train2017_{lang}.json")
EOF
echo "$(date)" > ALL_DONE
echo "end $(date)"
