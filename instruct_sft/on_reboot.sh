#!/usr/bin/env bash
# @reboot hook (crontab of cse_g2): resumes the instruct-mix run and its Drive uploader after the lab box reboots.
#   @reboot /home/cse_g2/sinllama-continual-pretraining/instruct_sft/on_reboot.sh
# Training resumes only if it is unfinished and did not end in FAILED (a real failure needs a human).
# The uploader restarts until UPLOAD_DONE. Log: runs/on_reboot.log
cd "$(dirname "$0")"
OUT=${OUT:-runs/r64}
RUN_NAME=${RUN_NAME:-wiki_uc_gen_mix_instruct_r64}
exec >> runs/on_reboot.log 2>&1
echo "$(date) boot hook"
for i in $(seq 60); do nvidia-smi > /dev/null 2>&1 && break; sleep 5; done   # wait for the driver
sleep 30                                                                        # and the network (wandb)
if [ ! -e "$OUT/TRAIN_DONE" ] && [ ! -e "$OUT/FAILED" ] && ! pgrep -f "train_instruct.py --out $OUT" > /dev/null; then
  echo "$(date) resuming training in $OUT"
  OUT=$OUT setsid nohup ./run_instruct.sh --run-name "$RUN_NAME" >> "$OUT.run.log" 2>&1 < /dev/null &
fi
if [ ! -e "$OUT/UPLOAD_DONE" ] && ! pgrep -f upload_adapters.sh > /dev/null; then
  echo "$(date) restarting uploader"
  OUT=$OUT setsid nohup ./upload_adapters.sh >> "$OUT.upload.log" 2>&1 < /dev/null &
fi
# benchmark chain (bench/bench_chain.sh, flock-guarded; it waits for training itself)
# (last line only: the log keeps CHAIN_DONE lines of earlier passes)
if [ -x ~/bench_mix/bench_chain.sh ] && ! tail -1 ~/bench_mix/chain.log 2>/dev/null | grep -q CHAIN_DONE; then
  echo "$(date) restarting benchmark chain"
  setsid nohup ~/bench_mix/bench_chain.sh >> ~/bench_mix/chain.log 2>&1 < /dev/null &
fi
if [ -x ~/bench_mix/bench_chain2.sh ] && ! tail -1 ~/bench_mix/chain2.log 2>/dev/null | grep -q CHAIN2_DONE; then
  setsid nohup ~/bench_mix/bench_chain2.sh >> ~/bench_mix/chain2.log 2>&1 < /dev/null &
fi
