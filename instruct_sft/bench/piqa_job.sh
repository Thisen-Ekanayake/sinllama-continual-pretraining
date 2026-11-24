#!/usr/bin/env bash
# Global-PIQA (4 splits, benchmark/mrlbenchmarks/config.yml settings: 8-shot, seed 0, batch 4, max_len 2048, bf16,
# raw prompt) for SinLlama_instruct_mix and its parent on the 16 GB lab box, through piqa_16gb.py.
# Log ~/bench_mix/piqa.log; outputs ~/bench_mix/out/piqa/results_<variant>_<lang>/; PIQA_DONE / PIQA_FAILED.
#   setsid nohup ~/bench_mix/piqa_job.sh > ~/bench_mix/piqa.log 2>&1 < /dev/null &
set -uo pipefail
REPO=~/sinllama-continual-pretraining; cd $REPO
PY=~/tts_env/bin/python; BM=~/bench_mix; O=$BM/out/piqa
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
rc=0
for m in SinLlama_instruct_mix SinLlama_wiki_uc_gen_mix; do
  for v in parallel nonparallel; do for l in sinhala english; do
    out=$O/results_${v}_${l}; mkdir -p $out
    echo "[$(date '+%F %T')] $m $v/$l"
    $PY $BM/piqa_16gb.py $v $l --data benchmark/mrlbenchmarks/global-piqa-$v/0.jsonl --models models/$m \
      --alpaca-models SinLlama_Bactrianx_Instruct --chat-models SinLlama_uc_instruct \
      --kshot 8 --seed 0 --batch-size 4 --max-len 2048 --quant bf16 --out-dir $out --skip-existing \
      > $out/$m.log 2>&1 || { echo "  FAILED (see $out/$m.log)"; rc=1; }
    tail -n 3 $out/$m.log | sed 's/^/  /'
  done; done
done
[ $rc = 0 ] && echo "[$(date '+%F %T')] PIQA_DONE" || echo "[$(date '+%F %T')] PIQA_FAILED"
