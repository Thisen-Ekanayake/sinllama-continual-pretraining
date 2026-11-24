#!/usr/bin/env bash
# Follow-up to bench_chain.sh: English MMLU (raw + permuted) re-run for the new model and the parent with
# tb_eval/mmlu16_fixed.py. The first run (permute_16gb.py alone) left 1,813/14,042 items unscored (OOM cascade).
# Waits for bench_chain.sh to finish. Markers ~/bench_mix/done/mmlu_en_fixed_<model>. Log ~/bench_mix/chain2.log.
set -u
exec 8> ~/bench_mix/.chain2.lock; flock -n 8 || { echo "chain2 already running"; exit 0; }
REPO=~/sinllama-continual-pretraining; cd $REPO
PY=~/tts_env/bin/python; BM=~/bench_mix
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
log() { echo "[$(date '+%F %T')] $*"; }
while pgrep -f "[b]ench_chain.sh" > /dev/null; do sleep 120; done
for m in SinLlama_instruct_mix SinLlama_wiki_uc_gen_mix; do
  n=mmlu_en_fixed_$m
  [ -f $BM/done/$n ] && { log "skip $n"; continue; }
  log "start $n"; t0=$(date +%s)
  if $PY ~/tb_eval/mmlu16_fixed.py --lang english --models models/$m --data-root benchmark/mmlu/english_mmlu \
       --out-dir $BM/out/EnglishMMLU_results_permuted_fixed --batch-size 4 --rotations 0 > $BM/logs/$n.log 2>&1; then
    echo $(( $(date +%s) - t0 )) > $BM/done/$n
    log "done $n: $(tr '\r' '\n' < $BM/logs/$n.log | grep -E 'raw \(rot|cyclic-permuted|WARNING|re-scored' | tr -s ' ' | tr '\n' ' ')"
  else
    log "FAILED $n"
  fi
done
log "CHAIN2_DONE"
