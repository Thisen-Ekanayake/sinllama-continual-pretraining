#!/usr/bin/env bash
# Upload the merged SinLlama_instruct_mix (wiki_uc_gen_mix + r64/best, bf16) to Drive, then verify.
set -uo pipefail
M=~/sinllama-continual-pretraining/models/SinLlama_instruct_mix; D=gdrive:sinllama_cpt/SinLlama_instruct_mix
echo "$(date) start"
for a in 1 2 3; do
  ~/bin/rclone copy $M $D --transfers 4 --drive-chunk-size 128M --stats 60s --stats-one-line 2>&1 | grep -v NOTICE
  # rclone prints its verdict as a NOTICE line ("... 0 differences found"), so match it before any NOTICE filter
  if ~/bin/rclone check $M $D --one-way 2>&1 | tee /dev/stderr | grep -q " 0 differences found"; then
    echo "$(date) UPLOAD_MERGED_DONE"; exit 0; fi
  echo "$(date) attempt $a: check failed, retrying"; sleep 60
done
echo "$(date) UPLOAD_MERGED_FAILED"; exit 1
