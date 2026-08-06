#!/usr/bin/env bash
# Downstream LoRA finetune: Sinhala Sentiment Analysis, one adapter per
# model in ../main.yml. Not meant to be run standalone — invoked by
# ../benchmark.sh.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/_run_downstream_task.sh" sentiment finetune_sentiment.py sentiment
