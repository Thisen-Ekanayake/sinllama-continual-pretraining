#!/bin/bash
# After run_gen.sh: NLLB si->en -> download Qwen3.5-9B GGUF -> llama-server
# -> judge both arms. Markers: TRANSLATE_DONE, ALL_DONE / REST_FAILED (reason in rest.log).
set -uo pipefail
cd ~/sinllama-continual-pretraining/VQA/sinllama_eval
PY=~/tts_env/bin/python
MODEL=~/sinllama-continual-pretraining/models/SinLlama_wiki_uc_gen_mix
QWEN=~/qwen
LLAMA=~/llama.cpp
fail() { echo "FAILED: $*"; pkill -f "[l]lama-server" ; touch REST_FAILED; exit 1; }
rm -f REST_FAILED ALL_DONE

echo "$(date) waiting for generation"
until [ -e GEN_DONE ]; do [ -e GEN_FAILED ] && fail "generation"; sleep 60; done

if [ ! -e TRANSLATE_DONE ]; then
  echo "$(date) NLLB si->en"
  $PY translate_si_en.py --inp sinllama_si_generated.json --out sinllama_si_translated_en.json \
    > translate.log 2>&1 || fail "translation"
  touch TRANSLATE_DONE
fi

$PY - <<'EOF' || fail "output check"
import json
for f in ["sinllama_en_generated.json", "sinllama_si_generated.json", "sinllama_si_translated_en.json"]:
    n = len(json.load(open(f))["records"]); print(f, n); assert n == 5000, f
EOF

# SinLlama model folder ($MODEL) is left in place for the user to delete; Qwen Q4_K_M fits beside it
df -h ~ | tail -1

if [ ! -s "$QWEN/Qwen3.5-9B-Q4_K_M.gguf" ]; then
  echo "$(date) downloading Qwen3.5-9B Q4_K_M"
  mkdir -p "$QWEN" && cd "$QWEN"
  ~/.local/bin/hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir . || fail "download"
  cd - > /dev/null
fi
ls -la "$QWEN"

if ! curl -sf localhost:8080/health > /dev/null; then
  export LD_LIBRARY_PATH="$LLAMA/bin:${LD_LIBRARY_PATH:-}"
  setsid nohup "$LLAMA/bin/llama-server" -m "$QWEN/Qwen3.5-9B-Q4_K_M.gguf" \
    -ngl 99 -c 16384 -np 4 -fa on --jinja --host 127.0.0.1 --port 8080 > server.log 2>&1 < /dev/null &
  for i in $(seq 120); do curl -sf localhost:8080/health > /dev/null && break; sleep 2; done
  curl -sf localhost:8080/health > /dev/null || fail "server start (see server.log)"
fi
echo "$(date) server up"

$PY judge.py --dump-criteria judge_criteria.json || fail "criteria dump"
$PY judge.py --arm en --out judge_smoke_en.json --limit 8 > judge_smoke.log 2>&1 \
  && $PY judge.py --arm si --out judge_smoke_si.json --limit 8 >> judge_smoke.log 2>&1 || fail "judge smoke"
echo "$(date) judge smoke ok"

for arm in en si; do
  ok=0
  for attempt in 1 2 3; do
    echo "$(date) judging $arm (attempt $attempt)"
    $PY judge.py --arm $arm --out judged_$arm.json >> judge_$arm.log 2>&1 && { ok=1; break; }
  done
  [ $ok = 1 ] || fail "judge $arm"
done

$PY - <<'EOF' || fail "summary"
import json
s = {arm: json.load(open(f"judged_{arm}.json"))["summary"] for arm in ["en", "si"]}
for arm, f in [("en", "sinllama_en_generated.json"), ("si", "sinllama_si_generated.json")]:
    s[arm]["generation_meta"] = json.load(open(f))["meta"]
s["si"]["translation_meta"] = json.load(open("sinllama_si_translated_en.json"))["meta"]
s["model"] = "SinLlama_wiki_uc_gen_mix"
s["judge"] = "Qwen3.5-9B-Q4_K_M (llama.cpp), see judge_criteria.json"
json.dump(s, open("summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({a: {"score_100": s[a]["score_100"], **s[a]["criteria_100"]} for a in ["en", "si"]}, indent=1))
EOF
pkill -f "[l]lama-server"
touch ALL_DONE
echo "$(date) ALL DONE"
