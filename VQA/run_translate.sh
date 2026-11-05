#!/usr/bin/env bash
# Unattended translation of the full VQA set on the lab box; writes DONE or FAILED.
# Launch detached:  setsid nohup ./run_translate.sh > run_translate.log 2>&1 < /dev/null &
# Rerunning resumes from vqa_sinhala.cache.jsonl.
set -o pipefail
cd "$(dirname "$0")"
rm -f TRANSLATE_DONE TRANSLATE_FAILED
echo "start $(date)"
for pass in 1 2 3; do
  ~/tts_env/bin/python translate_nllb.py --inp vqa_coco_val2017.json --out vqa_sinhala.json \
    2>&1 | grep -v "Loading checkpoint" && [ -s vqa_sinhala.json ] && break
  echo "pass $pass failed $(date)"; sleep 30
done
n=$(python3 -c "import json;print(len(json.load(open('vqa_sinhala.json'))))" 2>/dev/null || echo 0)
if [ "$n" -ge 5000 ]; then echo "$n records $(date)" > TRANSLATE_DONE; else echo "$n/5000 $(date)" > TRANSLATE_FAILED; fi
echo "end $(date): $n records"
