#!/bin/bash
# 5-shot re-run: generate en + si with shots_5.json (exemplars removed -> 4995 test items),
# NLLB si->en, judge both arms with Qwen3.5-9B, summary incl. 0-shot on the same 4995 items.
# Outputs in 5shot/. Markers 5shot/{GEN_DONE,ALL_DONE,FAILED}. Log 5shot/run.log.
set -uo pipefail
cd ~/sinllama-continual-pretraining/VQA/sinllama_eval
PY=~/tts_env/bin/python
LLAMA=~/llama.cpp
O=5shot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fail() { echo "$(date) FAILED: $*"; pkill -f "[l]lama-server"; touch $O/FAILED; exit 1; }
rm -f $O/FAILED $O/ALL_DONE

if [ ! -e $O/GEN_DONE ]; then
  for lang in en si; do
    inp=../vqa_english.json; [ $lang = si ] && inp=../vqa_sinhala.json
    echo "$(date) generating $lang (5-shot)"
    $PY gen_sinllama.py --lang $lang --inp $inp --out $O/sinllama_${lang}_generated_5shot.json \
      --shots shots_5.json --batch 6 --threads 24 > $O/gen_$lang.log 2>&1 || fail "generation $lang"
  done
  touch $O/GEN_DONE
fi

if [ ! -e $O/sinllama_si_translated_en_5shot.json ]; then
  echo "$(date) NLLB si->en"
  $PY translate_si_en.py --inp $O/sinllama_si_generated_5shot.json \
    --out $O/sinllama_si_translated_en_5shot.json > $O/translate.log 2>&1 || fail "translation"
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
    $PY judge.py --arm $arm --out $O/judged_${arm}_5shot.json \
      --gen-en $O/sinllama_en_generated_5shot.json --trans-si $O/sinllama_si_translated_en_5shot.json \
      >> $O/judge_$arm.log 2>&1 && { ok=1; break; }
  done
  [ $ok = 1 ] || fail "judge $arm"
done
pkill -f "[l]lama-server"

$PY - <<'EOF' || fail "summary"
import json, statistics as st
O = "5shot"
keys = ["question_understanding", "correctness", "completeness", "coherence"]
s = {"model": "SinLlama_wiki_uc_gen_mix", "setting": "5-shot (exemplars = earlier dialogue turns)",
     "shot_image_ids": json.load(open("shots_5.json")),
     "judge": "Qwen3.5-9B-Q4_K_M (llama.cpp), same rubric as 0-shot (judge_criteria.json)"}
for arm in ["en", "si"]:
    j5 = json.load(open(f"{O}/judged_{arm}_5shot.json"))
    ids = {r["image_id"] for r in j5["records"]}
    z = [r for r in json.load(open(f"judged_{arm}.json"))["records"] if r["image_id"] in ids]
    agg = lambda rs: {"score_100": round(10 * st.mean(r["score_10"] for r in rs), 2),
                      **{k: round(10 * st.mean(r["scores"][k] for r in rs), 2) for k in keys}}
    gen = json.load(open(f"{O}/sinllama_{arm}_generated_5shot.json"))
    s[arm] = {"five_shot": j5["summary"], "zero_shot_same_items": {"n": len(z), **agg(z)},
              "generation_meta": gen["meta"],
              "mean_generated_tokens_5shot": round(st.mean(r["generated_tokens"] for r in gen["records"]), 1)}
s["si"]["translation_meta"] = json.load(open(f"{O}/sinllama_si_translated_en_5shot.json"))["meta"]
json.dump(s, open(f"{O}/summary_5shot.json", "w"), ensure_ascii=False, indent=1)
for arm in ["en", "si"]:
    print(arm, "5-shot", s[arm]["five_shot"]["score_100"], s[arm]["five_shot"]["criteria_100"],
          "| 0-shot same items", s[arm]["zero_shot_same_items"])
EOF
touch $O/ALL_DONE
echo "$(date) ALL DONE"
