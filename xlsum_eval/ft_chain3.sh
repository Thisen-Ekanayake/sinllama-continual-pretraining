#!/bin/bash
# After ft_chain2.sh: re-decode llama-3-8b's English test set without n-gram
# blocking (en_llama-3-8b ran with the old no_repeat_ngram_size=2 default, which
# on a decoder-only model also blocks copying from the article). The rep2 output
# is kept as test_preds_rep2.jsonl. Log: ~/xlsum_eval/ft_chain3.log.
set -u
G=~/google-cloud-sdk/bin/gsutil
PY=~/tts_env/bin/python
M=~/sinllama-continual-pretraining/models
E=~/xlsum_eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q -E "ALL DONE|failed|mismatch|predictions|no adapter" $E/ft_chain2.log; do sleep 120; done
grep -q "ALL DONE" $E/ft_chain2.log || { echo "$(date) chain2 did not finish cleanly, not running"; exit 1; }
O=$E/ft_en/llama-3-8b
[ -f $O/test_preds_rep2.jsonl ] || mv $O/test_preds.jsonl $O/test_preds_rep2.jsonl
echo "$(date) downloading llama-3-8b"
$G -m -q cp -r gs://sinllama_cpt/llama-3-8b $M/ || { echo "download failed"; exit 1; }
want=$($G du -s gs://sinllama_cpt/llama-3-8b | awk '{print $1}')
have=$(find $M/llama-3-8b -type f -printf '%s\n' | awk '{s+=$1} END {print s}')
[ "$have" = "$want" ] || { echo "size mismatch: gcs $want local $have"; exit 1; }
echo "$(date) start en_rep0_llama-3-8b"
XL_LANG=english DATA=$E/english_XLSum_v2.0_sample ADAPTER=$O/best NO_REPEAT=0 PREDS=test_preds.jsonl \
    $PY $E/xlsum_ft.py $M/llama-3-8b $O > $E/logs/en_rep0_llama-3-8b.log 2>&1 || { echo "failed: en_rep0"; exit 1; }
n=$(wc -l < $O/test_preds.jsonl)
[ "$n" -eq 500 ] || { echo "en_rep0 has $n/500 predictions"; exit 1; }
rm -rf $M/llama-3-8b
echo "$(date) ALL DONE"
