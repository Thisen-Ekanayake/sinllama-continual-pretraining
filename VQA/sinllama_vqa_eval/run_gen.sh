#!/bin/bash
cd ~/sinllama-continual-pretraining/VQA/sinllama_eval
PY=~/tts_env/bin/python
rm -f GEN_DONE GEN_FAILED
$PY gen_sinllama.py --lang en --inp ../vqa_english.json --out sinllama_en_generated.json --batch 16 > gen_en.log 2>&1 \
 && $PY gen_sinllama.py --lang si --inp ../vqa_sinhala.json --out sinllama_si_generated.json --batch 16 > gen_si.log 2>&1 \
 && touch GEN_DONE || touch GEN_FAILED
