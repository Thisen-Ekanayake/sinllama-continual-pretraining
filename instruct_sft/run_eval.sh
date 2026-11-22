#!/usr/bin/env bash
# Post-training eval of an instruct-mix adapter on the COCO val2017 VQA test set (the same 500 image_ids as the
# VQA-only fine-tune, ft/test_ids.json; train2017 images are disjoint), with the exact chain of run_ft.sh:
# generate (greedy, rep 1.1, max_new 256, fp32 CPU head, batch 16) -> NLLB si->en -> Qwen3.5-9B judge -> summary
# against 0-shot and the VQA-only adapters. Lab box: run from ~/sinllama-continual-pretraining/instruct_sft.
#   ADAPTER=runs/r64/best NAME=r64 setsid nohup ./run_eval.sh > runs/eval_r64.log 2>&1 < /dev/null &
# Resumable (generation/judge from their jsonl). Markers <O>/ALL_DONE or <O>/FAILED.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ADAPTER=$(cd "$HERE" && realpath "${ADAPTER:-runs/r64/best}")
NAME=${NAME:-r64}
cd ~/sinllama-continual-pretraining/VQA/sinllama_eval
PY=~/tts_env/bin/python
LLAMA=~/llama.cpp
MODEL=~/sinllama-continual-pretraining/models/SinLlama_wiki_uc_gen_mix
O=mix_eval/$NAME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
fail() { echo "$(date) FAILED: $*"; pkill -x llama-server; touch $O/FAILED; exit 1; }
mkdir -p $O; rm -f $O/FAILED $O/ALL_DONE

for lang in en si; do
  out=$O/${lang}_generated.json
  if [ ! -e $out ]; then
    inp=../vqa_english.json; [ $lang = si ] && inp=../vqa_sinhala.json
    echo "$(date) generating $lang"
    $PY gen_sinllama.py --lang $lang --inp $inp --out $out --ids ft/test_ids.json \
      --adapter "$ADAPTER" --model $MODEL --batch 16 --head-fp32 > $O/gen_$lang.log 2>&1 || fail "generation $lang"
  fi
done
if [ ! -e $O/si_translated_en.json ]; then
  echo "$(date) NLLB si->en"
  $PY translate_si_en.py --inp $O/si_generated.json --out $O/si_translated_en.json > $O/translate.log 2>&1 \
    || fail translation
fi
if ! curl -sf localhost:8080/health > /dev/null; then
  export LD_LIBRARY_PATH="$LLAMA/bin:${LD_LIBRARY_PATH:-}"
  setsid nohup "$LLAMA/bin/llama-server" -m ~/qwen/Qwen3.5-9B-Q4_K_M.gguf \
    -ngl 99 -c 16384 -np 4 -fa on --jinja --host 127.0.0.1 --port 8080 > $O/server.log 2>&1 < /dev/null &
  for i in $(seq 120); do curl -sf localhost:8080/health > /dev/null && break; sleep 2; done
  curl -sf localhost:8080/health > /dev/null || fail "server start"
fi
for arm in en si; do
  ok=0
  for attempt in 1 2 3; do
    echo "$(date) judging $arm (attempt $attempt)"
    $PY judge.py --arm $arm --out $O/judged_$arm.json \
      --gen-en $O/en_generated.json --trans-si $O/si_translated_en.json >> $O/judge_$arm.log 2>&1 && { ok=1; break; }
  done
  [ $ok = 1 ] || fail "judge $arm"
done
pkill -x llama-server
$PY "$HERE/summarize_mix_eval.py" --dir $O || fail summary
touch $O/ALL_DONE
echo "$(date) ALL DONE"
