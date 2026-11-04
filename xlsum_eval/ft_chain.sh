#!/bin/bash
# One model on disk at a time: download -> QLoRA fine-tune on XL-Sum train ->
# generate test summaries -> delete model (adapter + preds kept in ~/xlsum_eval/ft/<name>/).
# Log: ~/xlsum_eval/ft_chain.log; per-model logs ~/xlsum_eval/logs/ft_<name>.log.
set -u
G=~/google-cloud-sdk/bin/gsutil
PY=~/tts_env/bin/python
M=~/sinllama-continual-pretraining/models
E=~/xlsum_eval
mkdir -p $E/logs $E/ft $E/done
for name in SinLlama_v01 SinLlama_wiki_uc_gen_mix SinLlama_textbook_mix; do
  [ -f $E/done/ft_$name ] && { echo "$(date) skip $name (done)"; continue; }
  if [ ! -f $M/$name/model.safetensors.index.json ]; then
    echo "$(date) downloading $name"
    $G -m -q cp -r gs://sinllama_cpt/$name $M/ || { echo "download failed: $name"; exit 1; }
  fi
  want=$($G du -s gs://sinllama_cpt/$name | awk '{print $1}')
  have=$(find $M/$name -type f -printf '%s\n' | awk '{s+=$1} END {print s}')
  [ "$have" = "$want" ] || { echo "size mismatch $name: gcs $want local $have"; exit 1; }
  echo "$(date) fine-tuning + generating $name"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $E/xlsum_ft.py $M/$name $E/ft/$name > $E/logs/ft_$name.log 2>&1 || { echo "failed: $name"; exit 1; }
  n=$(wc -l < $E/ft/$name/test_preds.jsonl)
  [ "$n" -eq 500 ] || { echo "$name has $n/500 predictions"; exit 1; }
  rm -rf $E/ft/$name/ckpt/*/optimizer.pt
  touch $E/done/ft_$name
  echo "$(date) $name done, deleting model"
  rm -rf $M/$name
done
echo "$(date) ALL DONE"
