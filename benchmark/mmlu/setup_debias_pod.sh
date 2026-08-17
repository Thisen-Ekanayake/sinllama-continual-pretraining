#!/usr/bin/env bash
# benchmark/mmlu/setup_debias_pod.sh
#
# Provision a fresh GPU pod for the position-bias study. The previous MI300X was
# destroyed on 2026-08-17, so nothing is cached: models and datasets both come
# from gs://sinllama_cpt (note the UNDERSCORE -- gs://sinllama-cpt does not
# exist, and both sft/config.yaml and benchmark/main.yml still have the wrong
# hyphenated form in places).
#
# Usage:
#   bash benchmark/mmlu/setup_debias_pod.sh
#   MODELS="SinLlama_v02 SinLlama_uc_instruct_cleaned SinLlama_v01" bash ...
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

BUCKET="${BUCKET:-gs://sinllama_cpt}"
MODELS="${MODELS:-SinLlama_v02 SinLlama_uc_instruct_cleaned}"

# The Cloud SDK has landed in a different place on every pod so far (~/, and
# inside the repo checkout), and it is never on a non-interactive PATH. Search
# instead of assuming.
find_gsutil() {
  local c
  c="$(command -v gsutil 2>/dev/null)" && [[ -x "$c" ]] && { echo "$c"; return; }
  for c in "$HOME/google-cloud-sdk/bin/gsutil" \
           "$REPO_DIR/google-cloud-sdk/bin/gsutil" \
           "/usr/lib/google-cloud-sdk/bin/gsutil" \
           "/snap/bin/gsutil"; do
    [[ -x "$c" ]] && { echo "$c"; return; }
  done
  c="$(find "$HOME" -maxdepth 4 -name gsutil -type f -perm -u+x 2>/dev/null | head -1)"
  [[ -n "$c" ]] && echo "$c"
}

GSUTIL="$(find_gsutil)"
if [[ -z "$GSUTIL" || ! -x "$GSUTIL" ]]; then
  echo "gsutil not found. Install the Cloud SDK, or add it to PATH:" >&2
  echo "  export PATH=\$PATH:~/google-cloud-sdk/bin" >&2
  exit 1
fi
echo "using gsutil: $GSUTIL"

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail=0

# --- models ----------------------------------------------------------------- #
mkdir -p models
for m in $MODELS; do
  if [[ -f "models/$m/config.json" ]]; then
    log "$m already present, skipping"
    continue
  fi
  log "downloading $m  (~16 GB)"
  if ! "$GSUTIL" -m cp -r "$BUCKET/$m" models/ ; then
    echo "FAILED to download $m from $BUCKET/$m" >&2
    fail=1
    continue
  fi
  # a truncated model is worse than a missing one -- it fails deep into a run
  n_shards=$(ls "models/$m"/model-*.safetensors 2>/dev/null | wc -l)
  if [[ ! -f "models/$m/config.json" || "$n_shards" -lt 1 ]]; then
    echo "INCOMPLETE download: models/$m (config.json or shards missing)" >&2
    fail=1
  else
    echo "  ok: $n_shards shards + config"
  fi
done

# --- datasets --------------------------------------------------------------- #
if [[ -d benchmark/mmlu/SinhalaMMLU ]]; then
  log "SinhalaMMLU already present"
else
  log "downloading SinhalaMMLU"
  "$GSUTIL" -m -q cp -r "$BUCKET/SinhalaMMLU" benchmark/mmlu/ || fail=1
fi

if [[ -d benchmark/mmlu/english_mmlu ]]; then
  log "english_mmlu already present"
else
  log "locating the English MMLU dataset in the bucket"
  # the prefix has been spelled several ways across uploads; take the first hit
  src=""
  for cand in mmlu-english mmlu_english english_mmlu english-mmlu; do
    if "$GSUTIL" ls "$BUCKET/$cand/" >/dev/null 2>&1; then src="$cand"; break; fi
  done
  if [[ -z "$src" ]]; then
    echo "Could not find English MMLU under $BUCKET. Candidates tried:" >&2
    "$GSUTIL" ls "$BUCKET/" 2>/dev/null | grep -i mmlu >&2
    fail=1
  else
    log "downloading $src -> benchmark/mmlu/english_mmlu"
    mkdir -p benchmark/mmlu/english_mmlu
    "$GSUTIL" -m -q cp -r "$BUCKET/$src/*" benchmark/mmlu/english_mmlu/ || fail=1
  fi
fi

# --- python deps ------------------------------------------------------------ #
log "checking python deps"
python - <<'PY'
import importlib, sys
missing = [m for m in ("torch", "transformers", "tqdm", "pandas", "yaml")
           if not importlib.util.find_spec(m)]
if missing:
    print("  MISSING:", ", ".join(missing))
    print("  install them into the ROCm venv before running run_debias.sh")
    sys.exit(1)
import torch
print(f"  torch {torch.__version__}, cuda/hip available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
else:
    print("  WARNING: no GPU visible -- run_debias.sh will be unusably slow")
PY
[[ $? -ne 0 ]] && fail=1

# --- summary ---------------------------------------------------------------- #
echo
if [[ $fail -ne 0 ]]; then
  echo "SETUP INCOMPLETE -- fix the errors above before running run_debias.sh" >&2
  exit 1
fi
echo "Setup OK. Next:"
echo "  DRY_RUN=1 bash benchmark/mmlu/run_debias.sh   # check the plan"
echo "  bash benchmark/mmlu/run_debias.sh             # run it"
