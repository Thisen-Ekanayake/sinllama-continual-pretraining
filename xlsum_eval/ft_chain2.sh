#!/bin/bash
# Round 2, one model on disk at a time:
#   llama-3-8b: Sinhala fine-tune (same article content as SinLlama, no 2-gram block) + English fine-tune
#   each SinLlama: English fine-tune + Sinhala re-decode of its saved adapter with no 2-gram block
# Log: ~/xlsum_eval/ft_chain2.log; per-job logs ~/xlsum_eval/logs/; done markers ~/xlsum_eval/done/.
set -u
G=~/google-cloud-sdk/bin/gsutil
PY=~/tts_env/bin/python
M=~/sinllama-continual-pretraining/models
E=~/xlsum_eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $E/logs $E/ft $E/ft_en $E/done

fetch() {
  local name=$1
  if [ ! -f $M/$name/model.safetensors.index.json ]; then
    echo "$(date) downloading $name"
    $G -m -q cp -r gs://sinllama_cpt/$name $M/ || { echo "download failed: $name"; exit 1; }
  fi
  want=$($G du -s gs://sinllama_cpt/$name | awk '{print $1}')
  have=$(find $M/$name -type f -printf '%s\n' | awk '{s+=$1} END {print s}')
  [ "$have" = "$want" ] || { echo "size mismatch $name: gcs $want local $have"; exit 1; }
}

job() {  # job <marker> <model> <out_dir> <preds_file> [ENV=VAL ...]
  local marker=$1 name=$2 out=$3 preds=$4; shift 4
  [ -f $E/done/$marker ] && { echo "$(date) skip $marker (done)"; return; }
  echo "$(date) start $marker"
  env "$@" PREDS=$preds $PY $E/xlsum_ft.py $M/$name $out > $E/logs/$marker.log 2>&1 || { echo "failed: $marker"; exit 1; }
  n=$(wc -l < $out/$preds)
  [ "$n" -eq 500 ] || { echo "$marker has $n/500 predictions"; exit 1; }
  rm -rf $out/ckpt/*/optimizer.pt
  touch $E/done/$marker
  echo "$(date) done $marker"
}

EN="XL_LANG=english DATA=$E/english_XLSum_v2.0_sample"

LLSI="DATA=$E/sinhala_XLSum_v2.0_sincontent MAX_SOURCE=0 MAX_TARGET=0 MAX_NEW=768 NO_REPEAT=0"

fetch llama-3-8b
# llama Sinhala: train and generate in separate processes (beam search on ~6k-token
# prompts needs a clean GPU; 1 article at a time)
if [ ! -f $E/done/si_train_llama-3-8b ]; then
  echo "$(date) start si_train_llama-3-8b"
  env $LLSI TRAIN_BS=1 ACCUM=16 EVAL_BS=2 TRAIN_ONLY=1 $PY $E/xlsum_ft.py $M/llama-3-8b $E/ft/llama-3-8b \
      > $E/logs/si_train_llama-3-8b.log 2>&1 || { echo "failed: si_train_llama-3-8b"; exit 1; }
  [ -f $E/ft/llama-3-8b/best/adapter_model.safetensors ] || { echo "no adapter saved"; exit 1; }
  touch $E/done/si_train_llama-3-8b
  echo "$(date) done si_train_llama-3-8b"
fi
job si_llama-3-8b llama-3-8b $E/ft/llama-3-8b test_preds.jsonl $LLSI ADAPTER=$E/ft/llama-3-8b/best GEN_BS=1
job en_llama-3-8b llama-3-8b $E/ft_en/llama-3-8b test_preds.jsonl $EN
rm -rf $M/llama-3-8b

for name in SinLlama_v01 SinLlama_wiki_uc_gen_mix SinLlama_textbook_mix; do
  if [ ! -f $E/done/en_$name ] || [ ! -f $E/done/si_norep_$name ]; then
    fetch $name
    job en_$name $name $E/ft_en/$name test_preds.jsonl $EN
    job si_norep_$name $name $E/ft/$name test_preds_norepeat0.jsonl ADAPTER=$E/ft/$name/best NO_REPEAT=0
  fi
  rm -rf $M/$name
done
echo "$(date) ALL DONE"
