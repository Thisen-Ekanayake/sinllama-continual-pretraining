#!/usr/bin/env bash
# Unattended full run on the lab box: start the server, generate all 5000 images,
# rerun up to 3 times to retry any failed images (generate_vqa.py resumes from
# its .jsonl), then stop the server and write DONE (or FAILED).
# Launch detached:  setsid nohup ./run_full.sh > run_full.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
OUT=${OUT:-vqa_coco_val2017.json}
rm -f DONE FAILED
echo "start $(date)"

for pass in 1 2 3 4; do
  if ! ./start_server.sh; then
    pkill -f llama-server; sleep 5; ./start_server.sh || { echo "server failed"; continue; }
  fi
  echo "== pass $pass $(date)"
  python3 generate_vqa.py --out "$OUT" 2>&1 | tee -a generate.log | grep -E "^(grounding|[0-9]+ images|wrote|\[error\])|00/"
  # finished when every image is in the final JSON
  n=$(python3 -c "import json;print(len(json.load(open('$OUT'))))" 2>/dev/null || echo 0)
  echo "records: $n"
  [ "$n" -ge 5000 ] && break
  pkill -f llama-server; sleep 5   # restart the server before retrying
done

pkill -f llama-server
n=$(python3 -c "import json;print(len(json.load(open('$OUT'))))" 2>/dev/null || echo 0)
if [ "$n" -ge 5000 ]; then echo "$n records $(date)" > DONE; else echo "$n/5000 records $(date)" > FAILED; fi
echo "end $(date): $n records"
