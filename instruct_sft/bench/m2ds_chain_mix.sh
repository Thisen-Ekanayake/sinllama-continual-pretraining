#!/bin/bash
# m2ds_chain_mix.sh: the M2DS recipe of m2ds_chain.sh for the models in $MODELS (already on disk; not fetched
# or deleted), no scoring here (ROUGE/BERTScore run locally on the pulled predictions). Original header:
# M2DS benchmark, unattended. One GCS model on disk at a time: the three SinLlama
# models (Sinhala then English each), then llama-3-8b. Per job: QLoRA fine-tune
# on the train split (best adapter by val loss), beam-4 decode of the full test
# split; then the adapter is deleted. At the end: ROUGE (both languages) and
# English BERTScore (roberta-large) into ~/m2ds_eval/results/. Sinhala
# BERTScore (HelaBERT) and the LaTeX report are done on the local machine.
#
# Safe to re-run at any point (e.g. ./start_chain.sh after a reboot): finished
# jobs have done/ markers, training resumes from its last checkpoint, decoding
# resumes from the predictions written so far. A job that keeps failing is
# marked in failed/ and the chain moves on. If training cannot run at a source
# budget (OOM), the next smaller budget is tried; the budget used is stored in
# out/<lang>/<model>/max_source.txt and used for decoding.
# Data: ~/m2ds_dedup (m2ds_dedup.py): official validation/test, training clusters
# that share an article or summary sentence with them removed -- the official
# splits leak (Sinhala: 73% of test articles are also in train). The finished
# official-split SinLlama_v01 Sinhala run is kept in official_split/ and scored
# as a reference row.
# Log: ~/m2ds_eval/chain.log; per-job logs ~/m2ds_eval/logs/.
set -u
G=~/google-cloud-sdk/bin/gsutil
PY=~/tts_env/bin/python
M=~/sinllama-continual-pretraining/models
E=~/m2ds_eval
R=$E/results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=~/m2ds_dedup
mkdir -p $E/logs $E/out/sinhala $E/out/english $E/done $E/failed $M
log() { echo "$(date '+%F %T') $*"; }

# a training/decoding process left by an earlier chain instance: let it finish


# models come from the Google Drive backup via rclone (the GCS bucket is gone);
# set SRC=gcs to use gsutil + gs://sinllama_cpt instead
SRC=${SRC:-rclone}
RC=~/bin/rclone
REMOTE=${REMOTE:-gdrive:sinllama_cpt}
src_bytes() {
  if [ $SRC = rclone ]; then
    $RC size --json $REMOTE/$1 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['bytes'])" 2>/dev/null
  else
    $G du -s gs://sinllama_cpt/$1 | awk '{print $1}'
  fi
}
fetch() {
  local name=$1 want have
  for attempt in 1 2 3; do
    if [ ! -f $M/$name/model.safetensors.index.json ] || [ $attempt -gt 1 ]; then
      log "downloading $name from $SRC (attempt $attempt)"
      if [ $SRC = rclone ]; then
        $RC copy $REMOTE/$name $M/$name --transfers 4 --retries 5 --low-level-retries 20 --stats 0
      else
        $G -m -q cp -r gs://sinllama_cpt/$name $M/
      fi
    fi
    want=$(src_bytes $name)
    have=$(find $M/$name -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')
    [ -n "$want" ] && [ "$have" = "$want" ] && return 0
    log "size mismatch $name: source '$want' local $have; retrying"
    sleep 60
  done
  return 1
}

train() {  # train <lang> <name> <out> "<budgets>" [ENV=VAL ...]
  local lang=$1 name=$2 out=$3 budgets=$4; shift 4
  local tlog=$E/logs/train_${lang}_$name.log
  [ -f $out/best/adapter_model.safetensors ] && [ -f $out/max_source.txt ] && return 0
  for b in $budgets; do
    # a run that crashed earlier keeps its budget (and resumes from its checkpoint)
    if [ -f $out/max_source.txt ] && [ "$(cat $out/max_source.txt)" -lt "$b" ]; then continue; fi
    echo $b > $out/max_source.txt
    for attempt in 1 2; do
      log "train ${lang}_$name MAX_SOURCE=$b (attempt $attempt)"
      env M2_LANG=$lang "$@" MAX_SOURCE=$b TRAIN_ONLY=1 $PY $E/m2ds_ft.py $M/$name $out >> $tlog 2>&1
      [ -f $out/best/adapter_model.safetensors ] && { rm -rf $out/ckpt; return 0; }
      log "train ${lang}_$name failed: $(grep -oE '[A-Za-z]*Error[^(]{0,60}' $tlog | tail -1)"
      # no checkpoint yet: a crash at the first steps (longest clusters first) = too big
      ls -d $out/ckpt/checkpoint-* > /dev/null 2>&1 || break
    done
    rm -rf $out/ckpt
  done
  return 1
}

decode() {  # decode <lang> <name> <out> <n_test> [ENV=VAL ...]
  local lang=$1 name=$2 out=$3 ntest=$4; shift 4
  local n=0
  for attempt in 1 2 3; do
    log "decode ${lang}_$name (attempt $attempt)"
    env M2_LANG=$lang "$@" MAX_SOURCE=$(cat $out/max_source.txt) ADAPTER=$out/best \
      $PY $E/m2ds_ft.py $M/$name $out >> $E/logs/gen_${lang}_$name.log 2>&1
    n=$(cat $out/test_preds.jsonl 2>/dev/null | wc -l)
    [ "$n" -eq "$ntest" ] && return 0
    log "decode ${lang}_$name stopped at $n/$ntest"
    sleep 30
  done
  return 1
}

job() {  # job <lang> <model> <n_test> "<budgets>" [ENV=VAL ...]
  local lang=$1 name=$2 ntest=$3 budgets=$4; shift 4
  local marker=${lang}_$name out=$E/out/$lang/$name
  [ -f $E/done/$marker ] && { log "skip $marker (done)"; return 0; }
  [ -f $E/failed/$marker ] && { log "skip $marker (failed earlier)"; return 1; }
  mkdir -p $out
  if train $lang $name $out "$budgets" "$@" && decode $lang $name $out $ntest "$@"; then
    rm -rf $out/best $out/ckpt                       # adapters are not kept
    touch $E/done/$marker
    log "done $marker"
  else
    touch $E/failed/$marker
    log "FAILED $marker (kept $out for inspection)"
  fi
}

# ~735 train tokens/s on the 4070 Ti Super: English (10.7k clusters) and llama
# Sinhala (~7k-token sequences) are capped at 450 optimizer steps (7,200 clusters);
# Sinhala SinLlama: 2,396 clusters = 150 steps/epoch, <= 3 epochs
SI="MAX_TARGET=512 MAX_NEW=512 EPOCHS=3 EVAL_STEPS=50 GEN_BS=4 GEN_TOK=4400"
EN="MAX_TARGET=384 MAX_NEW=384 MAX_STEPS=450 EVAL_STEPS=50 VAL_N=300 GEN_BS=4 GEN_TOK=4400"
# llama-3-8b spends ~8.5 tokens per SinLlama token on Sinhala (references alone
# ~1.3k tokens median). Training sequences are capped at 6144 tokens (8192 OOMs in
# the layer backward on 16 GB): 3500 source tokens (~410 SinLlama tokens of text)
# leave ~2.5k for the summary; decoding keeps the 8192 context.
LLSI="MAX_TARGET=4096 MAX_NEW=3072 TRAIN_CTX=6144 MAX_STEPS=450 EVAL_STEPS=50 GEN_BS=1"

for name in $MODELS; do
  job sinhala $name 275 "4096 3072 2048" $SI
  job english $name 1687 "4096 3072 2048" $EN
done
log "M2DS_MIX_DONE"
