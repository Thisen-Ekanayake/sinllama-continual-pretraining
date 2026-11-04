#!/bin/bash
# One model on disk at a time: (download) -> generate on XL-Sum test -> delete.
# Log: ~/xlsum_eval/chain.log; per-model logs ~/xlsum_eval/logs/; outputs ~/xlsum_eval/preds/.
set -u
G=~/google-cloud-sdk/bin/gsutil
PY=~/tts_env/bin/python
M=~/sinllama-continual-pretraining/models
E=~/xlsum_eval
mkdir -p $E/logs $E/preds $E/done
for name in llama-3-8b SinLlama_v01 SinLlama_wiki_uc_gen_mix SinLlama_textbook_mix; do
  [ -f $E/done/$name ] && { echo "$(date) skip $name (done)"; continue; }
  if [ ! -f $M/$name/model.safetensors.index.json ]; then
    echo "$(date) downloading $name"
    $G -m -q cp -r gs://sinllama_cpt/$name $M/ || { echo "download failed: $name"; exit 1; }
    want=$($G du -s gs://sinllama_cpt/$name | awk '{print $1}')
    have=$(du -sb $M/$name | awk '{print $1}')
    # du -sb counts the directory inode (4096); GCS du does not
    [ $((have - want)) -ge 0 ] && [ $((have - want)) -le 8192 ] || { echo "size mismatch $name: gcs $want local $have"; exit 1; }
  fi
  echo "$(date) generating $name"
  $PY $E/xlsum_gen.py $M/$name $E/preds/$name.jsonl > $E/logs/$name.log 2>&1 || { echo "generation failed: $name"; exit 1; }
  n=$(wc -l < $E/preds/$name.jsonl)
  [ "$n" -eq 500 ] || { echo "$name has $n/500 predictions"; exit 1; }
  touch $E/done/$name
  echo "$(date) $name done, deleting model"
  rm -rf $M/$name
done
echo "$(date) ALL DONE"
