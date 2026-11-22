#!/usr/bin/env bash
# One downstream LoRA probe (benchmark/downstream recipe, config.yml hyper-parameters) on the 16 GB lab box:
# the only change is --load_in_4bit (bf16 LoRA on an 8B model needs the 48 GB A40 it ran on before).
# If the configured micro-batch OOMs, the micro-batch is halved and grad-accum doubled (same effective batch).
#   downstream_job.sh <news|sentiment|writing> <model name under models/>
set -uo pipefail
T=$1; NAME=$2
REPO=~/sinllama-continual-pretraining; D=$REPO/benchmark/downstream; BM=~/bench_mix
cd $REPO
y() { python3 benchmark/lib/yaml_get.py benchmark/main.yml $D/config.yml "$@"; }
case $T in news) S=finetune_news_category.py; DIR=news ;; sentiment) S=finetune_sentiment.py; DIR=sentiment ;;
           writing) S=finetune_writing_style.py; DIR=Writing ;; esac
OUT=$BM/out/downstream/${T}_lora/$NAME; mkdir -p $OUT
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BS=$(y $T.train_batch_size); ACC=$(y $T.grad_accum)
for try in 1 2 3; do
  echo "$(date) $T/$NAME micro-batch $BS x accum $ACC" | tee -a $OUT/batch.txt
  if ~/tts_env/bin/python $D/$S --model_name_or_path models/$NAME --data_dir $BM/data/downstream/$DIR \
      --train_file $(y $T.train_file) --val_file $(y $T.val_file) --test_file $(y $T.test_file) \
      --output_dir $OUT --results_file $OUT/$(y $T.results_file) --max_seq_length $(y $T.max_seq_length) \
      --lora_rank $(y $T.lora_rank) --lora_alpha $(y $T.lora_alpha) --lora_dropout $(y $T.lora_dropout) \
      --lora_target_modules $(y $T.lora_target_modules) --num_train_epochs $(y $T.epochs) --learning_rate $(y $T.lr) \
      --per_device_train_batch_size $BS --per_device_eval_batch_size $(y $T.eval_batch_size) \
      --gradient_accumulation_steps $ACC --warmup_ratio $(y $T.warmup_ratio) --weight_decay $(y $T.weight_decay) \
      --logging_steps $(y $T.logging_steps) --eval_steps $(y $T.eval_steps) --save_steps $(y $T.save_steps) \
      --save_total_limit $(y $T.save_total_limit) --seed $(y $T.seed) --run_name $T-lora-$NAME --load_in_4bit \
      > $OUT/train_try$try.log 2>&1; then
    rm -rf $OUT/checkpoint-*; exit 0
  fi
  grep -q "OutOfMemoryError\|CUDA out of memory" $OUT/train_try$try.log || exit 1
  BS=$(( BS / 2 )); ACC=$(( ACC * 2 )); [ $BS -ge 1 ] || exit 1; rm -rf $OUT/checkpoint-*
done
exit 1
