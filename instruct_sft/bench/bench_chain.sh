#!/usr/bin/env bash
# Every results/latex benchmark on the instruct-mix model, lab box, same methods as the earlier rows.
# New model: SinLlama_instruct_mix = SinLlama_wiki_uc_gen_mix + instruct_sft/runs/r64/best merged (bf16).
# Parent re-scored here where its earlier numbers came from pods (English MMLU, Global-PIQA, downstream).
# Waits for TRAIN_DONE + UPLOAD_DONE and for fetch_data.sh. Re-runnable: done/<job> markers, failed/<job> after
# 2 failed attempts (the chain moves on). Log ~/bench_mix/chain.log, per-job logs ~/bench_mix/logs/.
#   setsid nohup ~/bench_mix/bench_chain.sh >> ~/bench_mix/chain.log 2>&1 < /dev/null &
set -u
exec 9> ~/bench_mix/.chain.lock; flock -n 9 || { echo "chain already running"; exit 0; }
REPO=~/sinllama-continual-pretraining; cd $REPO
PY=~/tts_env/bin/python; BM=~/bench_mix; TB=~/tb_eval
M=$REPO/models; NEW=SinLlama_instruct_mix; PAR=SinLlama_wiki_uc_gen_mix
RUN=$REPO/instruct_sft/runs/r64; O=$BM/out
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
mkdir -p $BM/done $BM/failed $BM/logs $O
log() { echo "[$(date '+%F %T')] $*"; }

job() {  # job <name> <command...>: run once, retry once, mark done/failed
  local n=$1; shift
  [ -f $BM/done/$n ] && { log "skip $n (done)"; return 0; }
  [ -f $BM/failed/$n ] && { log "skip $n (failed earlier)"; return 1; }
  for a in 1 2; do
    log "start $n (attempt $a)"; local t0=$(date +%s)
    if bash -c "$*" >> $BM/logs/$n.log 2>&1; then
      echo $(( $(date +%s) - t0 )) > $BM/done/$n; log "done  $n in $(( ($(date +%s) - t0) / 60 )) min"; return 0
    fi
    log "attempt $a of $n failed: $(grep -aoE '[A-Za-z]*Error[^(]{0,80}' $BM/logs/$n.log | tail -1)"; sleep 60
  done
  touch $BM/failed/$n; log "FAILED $n"; return 1
}

until [ -f $RUN/TRAIN_DONE ] && [ -f $RUN/UPLOAD_DONE ]; do
  [ -f $RUN/FAILED ] || [ -f $RUN/UPLOAD_FAILED ] && { log "training or upload failed; not benchmarking"; exit 1; }
  sleep 120
done
until grep -q FETCH_DONE $BM/fetch.log 2>/dev/null; do sleep 60; done
log "training + data ready"
# checkpoints are only resume points; the best/final adapters stay
[ -d $RUN/ckpt ] && rm -rf $RUN/ckpt && log "removed $RUN/ckpt"

# 0. merged model + retrieval index
job merge "[ -f $M/$NEW/merged_from.json ] || $PY $BM/merge_adapter.py $M/$PAR $RUN/best $M/$NEW"

# 1. COCO VQA (base + adapter, exactly the VQA-only fine-tune chain)
job vqa "cd $REPO/instruct_sft && ADAPTER=runs/r64/best NAME=r64 ./run_eval.sh && test -f $REPO/VQA/sinllama_eval/mix_eval/r64/ALL_DONE"

# 2. SinhalaMMLU raw + permuted (parent's lab-box run exists: 38.67 / 41.44)
job mmlu_si "$PY $TB/permute_16gb.py --lang sinhala --models models/$NEW --data-root benchmark/mmlu/SinhalaMMLU \
  --prompt-file benchmark/prompts/mmlu_sinhala.txt --out-dir $O/SinhalaMMLU_results_permuted --batch-size 4 --rotations 0"

# 4. Belebele (si/en x passage/no-passage, raw + cyclic)
job belebele "$PY $TB/belebele/belebele_eval.py --model models/$NEW --out $O/belebele"

# 5. Global-PIQA: removed at the user's request (2026-11-22)

# 6. English MMLU raw + permuted -- new model and parent
for m in $NEW $PAR; do
  job mmlu_en_$m "$PY $TB/permute_16gb.py --lang english --models models/$m --data-root benchmark/mmlu/english_mmlu \
    --out-dir $O/EnglishMMLU_results_permuted --batch-size 4 --rotations 0"
done

# 7. XL-Sum (QLoRA r16, best by val loss, beam 4, no n-gram blocking), Sinhala then English
X=~/xlsum_eval
job xlsum_si "$PY $X/xlsum_ft.py $M/$NEW $X/ft/$NEW && [ \$(wc -l < $X/ft/$NEW/test_preds.jsonl) -eq 500 ] && rm -rf $X/ft/$NEW/ckpt"
job xlsum_en "XL_LANG=english DATA=$X/english_XLSum_v2.0_sample $PY $X/xlsum_ft.py $M/$NEW $X/ft_en/$NEW \
  && [ \$(wc -l < $X/ft_en/$NEW/test_preds.jsonl) -eq 500 ] && rm -rf $X/ft_en/$NEW/ckpt"

# 8. M2DS (dedup train, QLoRA, beam 4), the m2ds_chain.sh recipe for one model
job m2ds "MODELS=$NEW ~/m2ds_eval/m2ds_chain_mix.sh && test -f ~/m2ds_eval/done/sinhala_$NEW && test -f ~/m2ds_eval/done/english_$NEW"

# 3. Open-book SinhalaMMLU (Appendix C): bge-m3 index over the textbook chunks, then closed book (k=0) and top-3.
#    Needs ~/tb_eval/rag/chunks.jsonl (33,090 passages from textbooks/markdown).
if [ "$(wc -l < $TB/rag/chunks.jsonl 2>/dev/null)" = 33090 ]; then
  job rag_embed "$PY $TB/rag/embed_retrieve.py --embedder bge-m3"
  for e in closed:0 bge-m3:3; do
    job rag_${e%:*} "$PY $TB/rag/rag_eval.py --retrieval $TB/rag/bge-m3/retrieval.jsonl --k ${e#*:} --lang sinhala \
      --models models/$NEW --data-root benchmark/mmlu/SinhalaMMLU --prompt-file benchmark/prompts/mmlu_sinhala.txt \
      --out-dir $O/rag_results_${e%:*} --batch-size 4 --attn-budget 6e6 --rotations 1"
  done
else
  log "RAG skipped: $TB/rag/chunks.jsonl missing"
fi

# 9. Downstream LoRA probes, 4-bit on 16 GB -- new model and parent (needs the cleaned splits in $BM/data/downstream)
if [ -f $BM/data/downstream/READY ]; then
  for m in $NEW $PAR; do for t in news sentiment writing; do
    job down_${t}_$m "$BM/downstream_job.sh $t $m"
  done; done
else
  log "downstream skipped: cleaned splits not in $BM/data/downstream yet"
fi

log "CHAIN_DONE: done=$(ls $BM/done | wc -l) failed=$(ls $BM/failed | tr '\n' ' ')"
