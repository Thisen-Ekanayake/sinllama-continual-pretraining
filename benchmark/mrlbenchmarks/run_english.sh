#!/usr/bin/env bash
# Run both Global-PIQA sub-datasets (parallel + non-parallel) for English,
# across all configured models. Not meant to be run standalone — invoked by
# ../benchmark.sh.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/_run_piqa_variant.sh" english parallel
"$SCRIPT_DIR/_run_piqa_variant.sh" english nonparallel
