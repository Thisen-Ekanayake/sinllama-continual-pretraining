#!/bin/bash
# QLoRA fine-tune per language (ft_vqa.py) -> generate the 500 test answers with the merged
# adapter (0-shot decoding settings) -> NLLB si->en -> Qwen3.5-9B judge -> ft/summary_ft.json.
# Resumable: rerun to continue (training from its last checkpoint, generation/judge from jsonl).
# Markers ft/{<lang>/TRAIN_DONE,ALL_DONE,FAILED}. Log ft/run.log (launch with setsid nohup).
set -uo pipefail
cd ~/sinllama-continual-pretraining/VQA/sinllama_eval
PY=~/tts_env/bin/python
LLAMA=~/llama.cpp
MODEL=~/sinllama-continual-pretraining/models/SinLlama_wiki_uc_gen_mix
O=ft
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
fail() { echo "$(date) FAILED: $*"; pkill -f "[l]lama-server"; touch $O/FAILED; exit 1; }
mkdir -p $O
rm -f $O/FAILED $O/ALL_DONE

[ -e $O/split.json ] || $PY ft_vqa.py split || fail "split"

for lang in en si; do
  if [ ! -e $O/$lang/TRAIN_DONE ]; then
    echo "$(date) training $lang"
    mkdir -p $O/$lang
    $PY ft_vqa.py train --lang $lang >> $O/$lang/train.log 2>&1 || fail "train $lang"
    rm -rf $O/$lang/ckpt          # best adapter is in $O/$lang/best; disk on this box is ~3 GB
    touch $O/$lang/TRAIN_DONE
  fi
done

for lang in en si; do
  out=$O/ft_${lang}_generated.json
  if [ ! -e $out ]; then
    inp=../vqa_english.json; [ $lang = si ] && inp=../vqa_sinhala.json
    echo "$(date) generating $lang test answers"
    $PY gen_sinllama.py --lang $lang --inp $inp --out $out --ids $O/test_ids.json \
      --adapter $O/$lang/best --model $MODEL --batch 16 --head-fp32 > $O/gen_$lang.log 2>&1 \
      || fail "generation $lang"
  fi
done

if [ ! -e $O/ft_si_translated_en.json ]; then
  echo "$(date) NLLB si->en"
  $PY translate_si_en.py --inp $O/ft_si_generated.json --out $O/ft_si_translated_en.json \
    > $O/translate.log 2>&1 || fail "translation"
fi

if ! curl -sf localhost:8080/health > /dev/null; then
  export LD_LIBRARY_PATH="$LLAMA/bin:${LD_LIBRARY_PATH:-}"
  setsid nohup "$LLAMA/bin/llama-server" -m ~/qwen/Qwen3.5-9B-Q4_K_M.gguf \
    -ngl 99 -c 16384 -np 4 -fa on --jinja --host 127.0.0.1 --port 8080 > $O/server.log 2>&1 < /dev/null &
  for i in $(seq 120); do curl -sf localhost:8080/health > /dev/null && break; sleep 2; done
  curl -sf localhost:8080/health > /dev/null || fail "server start"
fi
echo "$(date) server up"

for arm in en si; do
  ok=0
  for attempt in 1 2 3; do
    echo "$(date) judging $arm (attempt $attempt)"
    $PY judge.py --arm $arm --out $O/judged_${arm}_ft.json \
      --gen-en $O/ft_en_generated.json --trans-si $O/ft_si_translated_en.json \
      >> $O/judge_$arm.log 2>&1 && { ok=1; break; }
  done
  [ $ok = 1 ] || fail "judge $arm"
done
pkill -f "[l]lama-server"

$PY summarize_ft.py || fail "summary"
touch $O/ALL_DONE
echo "$(date) ALL DONE"
