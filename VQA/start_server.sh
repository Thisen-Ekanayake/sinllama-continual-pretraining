#!/usr/bin/env bash
# Start llama-server (Qwen3.5-9B Q4_K_M + F16 vision projector) detached on the lab box
# and wait until it is healthy. Stop with: pkill -f llama-server
#   -np 4                   : 4 parallel slots (generate_vqa.py --workers 4)
#   -c 32768                : shared KV (unified), plenty for ~1.3k-token prompts x 4
#   --image-min-tokens 1024 : more vision tokens per image than the default
#                             (llama.cpp warns the default hurts accuracy on Qwen-VL)
set -euo pipefail
LLAMA=${LLAMA:-$HOME/llama.cpp}
QWEN=${QWEN:-$HOME/qwen}
LOG=${LOG:-$HOME/vqa/server.log}
PORT=${PORT:-8080}
mkdir -p "$(dirname "$LOG")"

if curl -sf "localhost:$PORT/health" >/dev/null; then
  echo "server already running on :$PORT"; exit 0
fi

export LD_LIBRARY_PATH="$LLAMA/bin:${LD_LIBRARY_PATH:-}"
setsid nohup "$LLAMA/bin/llama-server" \
  -m "$QWEN/Qwen3.5-9B-Q4_K_M.gguf" \
  --mmproj "$QWEN/mmproj-F16.gguf" \
  -ngl 99 -c 32768 -np 4 -fa on --jinja \
  --image-min-tokens 1024 \
  --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 < /dev/null &

for i in $(seq 120); do
  if curl -sf "localhost:$PORT/health" >/dev/null; then echo "server up (log $LOG)"; exit 0; fi
  sleep 2
done
echo "server did not come up, see $LOG" >&2; exit 1
