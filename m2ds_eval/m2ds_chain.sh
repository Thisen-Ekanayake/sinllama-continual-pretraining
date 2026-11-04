#!/bin/bash
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
while pgrep -f "^$PY $E/m2ds_ft.py" > /dev/null; do sleep 60; done

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

for name in SinLlama_v01 SinLlama_wiki_uc_gen_mix SinLlama_textbook_mix llama-3-8b; do
  todo=0
  for lang in sinhala english; do
    [ -f $E/done/${lang}_$name ] || [ -f $E/failed/${lang}_$name ] || todo=1
  done
  if [ $todo = 1 ]; then
    if fetch $name; then
      if [ $name = llama-3-8b ]; then          # user: English first, then Sinhala
        job english $name 1687 "4096 3072 2048" $EN
        job sinhala $name 275 "3500 3000 2500" $LLSI
      else
        job sinhala $name 275 "4096 3072 2048" $SI
        job english $name 1687 "4096 3072 2048" $EN
      fi
    else
      log "FAILED download $name"
      touch $E/failed/sinhala_$name $E/failed/english_$name
    fi
  fi
  rm -rf $M/$name
done

# ---- collect + score (on the server; Sinhala BERTScore runs locally) ----
log "scoring"
for lang in sinhala english; do
  mkdir -p $R/$lang/preds $R/$lang/train_states
  for name in llama-3-8b SinLlama_v01 SinLlama_wiki_uc_gen_mix SinLlama_textbook_mix; do
    o=$E/out/$lang/$name
    [ -f $E/done/${lang}_$name ] && cp $o/test_preds.jsonl $R/$lang/preds/$name.jsonl
    [ -f $o/train_state.json ] && cp $o/train_state.json $R/$lang/train_states/$name.json
    [ -f $o/max_source.txt ] && cp $o/max_source.txt $R/$lang/train_states/$name.max_source.txt
  done
  (cd $E/scoring && PYTHONPATH=pkgs NLTK_DATA=$E/scoring/nltk_data $E/rouge_env/bin/python score_m2ds.py \
      $R/$lang/preds ~/m2ds/$lang/test.json $R/$lang/rouge $lang > $R/$lang/rouge.txt 2>&1) \
    || log "ROUGE scoring failed for $lang (see $R/$lang/rouge.txt)"
done
# reference row: SinLlama_v01 fine-tuned on the official (leaky) Sinhala train split
O=$E/official_split/sinhala
if [ -f $O/SinLlama_v01/test_preds.jsonl ]; then
  mkdir -p $R/sinhala/preds_official
  cp $O/SinLlama_v01/test_preds.jsonl $R/sinhala/preds_official/SinLlama_v01.jsonl
  cp $O/SinLlama_v01/train_state.json $R/sinhala/train_states/SinLlama_v01.official.json
  (cd $E/scoring && PYTHONPATH=pkgs NLTK_DATA=$E/scoring/nltk_data $E/rouge_env/bin/python score_m2ds.py \
      $R/sinhala/preds_official ~/m2ds/sinhala/test.json $R/sinhala/rouge_official sinhala > $R/sinhala/rouge_official.txt 2>&1)
fi
cp $DATA/leakage_stats.json $R/
bs_ok=0
ROBERTA=$E/roberta-large $PY $E/scoring/bertscore_m2ds.py $R/english/preds $R/english/bertscore_roberta_large english \
  > $R/english/bertscore_roberta_large.txt 2>&1 && bs_ok=1 || log "English BERTScore failed (see $R/english/bertscore_roberta_large.txt)"
mkdir -p $R/logs $R/scripts
cp $E/logs/*.log $E/chain.log $R/logs/ 2>/dev/null
cp $E/m2ds_ft.py $E/m2ds_chain.sh $E/start_chain.sh $E/m2ds_lengths.py $E/m2ds_dedup.py $E/scoring/*.py $R/scripts/ 2>/dev/null
{ echo "finished $(date)"; echo "done:"; ls $E/done; echo "failed:"; ls $E/failed; } > $R/STATUS.txt
[ $bs_ok = 1 ] && rm -rf $E/roberta-large    # kept for a re-run if BERTScore failed
touch $E/ALL_DONE
log "ALL DONE -> $R"
