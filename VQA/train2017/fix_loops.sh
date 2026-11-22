#!/usr/bin/env bash
# COCO train2017 loop repair, chunks 00-09 (lab box ~/coco_train). See fix_loops.py.
# Launch: setsid nohup ./fix_loops.sh > fix/fix.log 2>&1 < /dev/null &   -> fix/DONE or fix/FAILED
set -uo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PY=${PY:-$HOME/tts_env/bin/python}
NLLB=${NLLB:-$HOME/nllb_fp16}
LX="--lexicon $HERE/sinhala_lexicon.pkl --vocab $HERE/unigram_32000_0.9995.vocab"
export LOG=$HERE/fix/server.log
mkdir -p fix; rm -f fix/DONE fix/FAILED
fail() { echo "$(date) FAILED: $*"; pkill -x llama-server; echo "$*" > fix/FAILED; exit 1; }

echo "$(date) scan"
python3 scan_loops.py --samples 0 --out loops_100k.jsonl | tail -n 1
python3 fix_loops.py select || fail select

echo "$(date) regenerate Qwen-loop records"
./start_server.sh || fail "server start"
python3 fix_loops.py regen || fail regen
pkill -x llama-server; sleep 5
python3 fix_loops.py en || fail en

echo "$(date) translate"
$PY translate_nllb.py --inp fix/en_fix.json --out fix/si_fix.raw.json --model "$NLLB" \
    --nopunct-first --logprob $LX > fix/si.log 2>&1 || fail translate
$PY repair_nllb.py --cache fix/si_fix.raw.cache.jsonl --model "$NLLB" --long-loops >> fix/si.log 2>&1 || fail repair
$PY postprocess_si.py --en fix/en_fix.json --out fix/si_fix.json --fix-split-hal $LX \
    --cache fix/si_fix.raw.cache.jsonl fix/si_fix.raw.cache.repair.jsonl >> fix/si.log 2>&1 || fail postprocess
grep -E "^(wrote|repaired|ZWJ merges|rare letters|latin gaps|[0-9]+ cached)" fix/si.log | tail -n 8

python3 fix_loops.py merge || fail merge
date > fix/DONE
echo "$(date) done"
