#!/bin/bash
# Pull the UltraChat-Sinhala *sft* split from the data VM to the local training box
# and lay it out the way run_sft.sh expects:
#
#   data/sft_uc/train/train_sft.sinhala.jsonl   <- dataset_dir (train glob)
#   data/sft_uc/val_sft_2k.jsonl                <- validation_file (random 2k of test)
#   data/sft_uc/test_sft.sinhala.jsonl          <- full held-out test (final eval)
#
# Run from the repo root:  bash sft/prepare_data.sh
set -euo pipefail

ZONE="us-central1-b"
INSTANCE="instance-20260626-193424"
PROJECT="mlflow-00"
REMOTE_DIR="uc/final"                 # ~/uc/final on the VM

DEST=${DEST:-data/sft_uc}
VAL_N=${VAL_N:-2000}

mkdir -p "${DEST}/train"

echo "Copying train_sft (~3.2 GB) and test_sft (~363 MB) from the VM ..."
gcloud compute scp --zone "${ZONE}" --project "${PROJECT}" \
    "${INSTANCE}:${REMOTE_DIR}/train_sft.sinhala.jsonl" "${DEST}/train/train_sft.sinhala.jsonl"
gcloud compute scp --zone "${ZONE}" --project "${PROJECT}" \
    "${INSTANCE}:${REMOTE_DIR}/test_sft.sinhala.jsonl" "${DEST}/test_sft.sinhala.jsonl"

echo "Sampling ${VAL_N} validation dialogues from test_sft ..."
shuf -n "${VAL_N}" "${DEST}/test_sft.sinhala.jsonl" > "${DEST}/val_sft_2k.jsonl"

echo "Done. Layout:"
echo "  dataset_dir      : ${DEST}/train  ($(wc -l < "${DEST}/train/train_sft.sinhala.jsonl") dialogues)"
echo "  validation_file  : ${DEST}/val_sft_2k.jsonl  ($(wc -l < "${DEST}/val_sft_2k.jsonl") dialogues)"
echo "  test (held out)  : ${DEST}/test_sft.sinhala.jsonl  ($(wc -l < "${DEST}/test_sft.sinhala.jsonl") dialogues)"
