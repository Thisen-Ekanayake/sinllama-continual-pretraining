#!/usr/bin/env bash
#
# Execute the THREE weights-analysis notebooks IN PLACE on THIS machine (the
# MI300X pod), running every code cell so all outputs — prints, DataFrame
# tables and the inline matplotlib figures from plt.show() — are embedded
# straight into the .ipynb files.
#
#   embedding_analysis.ipynb   (CPU: safetensors + sklearn on the embeddings)
#   attention_analysis.ipynb   (CPU Part A + full-8B GPU Part B, eager attn)
#   mlp_analysis.ipynb         (CPU Part A + full-8B GPU Part B, MLP hooks)
#
# All three run **concurrently**: they overlap well because embedding_* is
# CPU-bound while the other two spend most of their time on GPU forwards, and
# one notebook's CPU Part A overlaps another's GPU Part B. A live dashboard
# shows a progress bar + ETA per notebook. Run this ON the pod (notebooks +
# models are already local here); it does NOT ssh or copy anything.
#
# Notes / trade-offs of running in parallel:
#   * GPU: attention_* and mlp_* each load ONE 8B model at a time in Part B and
#     free it, so peak VRAM is ~2 x 16 GB + activations — fine on the MI300X.
#   * CPU RAM: embedding_* (~18 GB) + attention_* Part A (~21 GB) can be live at
#     once (~40 GB). Comfortable on the pod; trim MODEL_DIRS on smaller boxes.
#   * BLAS: to avoid the three CPU-heavy jobs oversubscribing the cores, the
#     thread pools (OMP/OpenBLAS/MKL) are split evenly across the jobs.
#
# Usage:   bash weights_analysis/run_notebooks_on_pod.sh
#          SUFFIX=_executed bash weights_analysis/run_notebooks_on_pod.sh   # write copies
#          POLL=5 bash weights_analysis/run_notebooks_on_pod.sh             # slower dashboard refresh
set -euo pipefail

VENV_PY="${VENV_PY:-$HOME/sinllama-continual-pretraining/venv/bin/python}"
DIR="${DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"   # dir holding the notebooks
KERNEL="${KERNEL:-sinllama}"
SUFFIX="${SUFFIX:-}"                          # "" = in place; else write <name><SUFFIX>.ipynb
POLL="${POLL:-2}"                            # dashboard refresh seconds
LOG_DIR="${LOG_DIR:-$DIR/logs}"
NOTEBOOKS=(embedding_analysis.ipynb attention_analysis.ipynb mlp_analysis.ipynb)

echo "==> [1/3] ensuring execution deps + kernel '$KERNEL'"
"$VENV_PY" -c 'import nbclient, nbconvert, ipykernel, seaborn' 2>/dev/null \
  || "$VENV_PY" -m pip install -q --upgrade seaborn nbconvert nbclient ipykernel
[ -d "$HOME/.local/share/jupyter/kernels/$KERNEL" ] \
  || "$VENV_PY" -m ipykernel install --user --name "$KERNEL" --display-name "$KERNEL"

# Split CPU threads across the parallel jobs so their BLAS pools don't thrash.
NPROC="$(nproc 2>/dev/null || echo 8)"
TPJ=$(( NPROC / ${#NOTEBOOKS[@]} )); [ "$TPJ" -lt 1 ] && TPJ=1

# Per-notebook executor: runs each code cell in-place (allow_errors, no per-cell
# timeout), emitting a machine-readable "PROGRESS <done> <total>" line per cell
# that the bash dashboard below turns into a progress bar. Written to a temp file.
DRIVER="$(mktemp --suffix=.py)"
trap 'rm -f "$DRIVER"' EXIT
cat > "$DRIVER" <<'PYEOF'
import os, sys, nbformat
from nbclient import NotebookClient

path, kernel = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else path
nb = nbformat.read(path, as_version=4)
code_idx = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
total = len(code_idx)
run_dir = os.path.dirname(os.path.abspath(path)) or "."
client = NotebookClient(nb, timeout=-1, kernel_name=kernel, allow_errors=True,
                        resources={"metadata": {"path": run_dir}})
print(f"PROGRESS 0 {total}", flush=True)
try:
    with client.setup_kernel():
        for n, i in enumerate(code_idx, 1):
            client.execute_cell(nb.cells[i], i)   # runs cell, stores its outputs
            print(f"PROGRESS {n} {total}", flush=True)
finally:
    nbformat.write(nb, out)                        # save outputs (even if aborted)
    print("DONE", flush=True)
PYEOF

echo "==> [2/3] launching ${#NOTEBOOKS[@]} notebooks in parallel"
echo "    venv=$VENV_PY | threads/job=$TPJ | logs=$LOG_DIR${SUFFIX:+ | writing *$SUFFIX.ipynb}"
mkdir -p "$LOG_DIR"

PIDS=(); LOGS=(); STARTS=()
for nb in "${NOTEBOOKS[@]}"; do
  src="$DIR/$nb"
  [ -f "$src" ] || { echo "ERROR: $src not found" >&2; exit 1; }
  if [ -n "$SUFFIX" ]; then out="$DIR/${nb%.ipynb}${SUFFIX}.ipynb"; else out="$src"; fi
  log="$LOG_DIR/${nb%.ipynb}.log"; : > "$log"
  OMP_NUM_THREADS="$TPJ" OPENBLAS_NUM_THREADS="$TPJ" MKL_NUM_THREADS="$TPJ" \
    NUMEXPR_NUM_THREADS="$TPJ" \
    "$VENV_PY" "$DRIVER" "$src" "$KERNEL" "$out" >"$log" 2>&1 &
  PIDS+=($!); LOGS+=("$log"); STARTS+=("$(date +%s)")
done

fmt_hms() {                                   # seconds -> Hh MMm / MMmSSs
  local s=$1
  if [ "$s" -ge 3600 ]; then printf '%dh%02dm' $((s/3600)) $((s%3600/60))
  else printf '%dm%02ds' $((s/60)) $((s%60)); fi
}

is_running() {                                # true only if $1 is alive & not a zombie
  local s
  s=$(cat "/proc/$1/stat" 2>/dev/null) || return 1
  s=${s#*) }                                  # strip "pid (comm) "
  [ "${s%% *}" = "Z" ] && return 1            # reaped-but-unwaited child = finished
  return 0
}

draw() {
  local now i nb log start line cur total pct bar filled empty el eta eta_str
  now=$(date +%s)
  for i in "${!NOTEBOOKS[@]}"; do
    nb="${NOTEBOOKS[$i]}"; log="${LOGS[$i]}"; start="${STARTS[$i]}"
    line=$(grep '^PROGRESS ' "$log" 2>/dev/null | tail -1 || true)
    el=$(( now - start ))
    if [ -z "$line" ]; then
      printf '  %-26s  starting…  (el %s)\033[K\n' "$nb" "$(fmt_hms "$el")"
      continue
    fi
    cur=$(awk '{print $2}' <<<"$line"); total=$(awk '{print $3}' <<<"$line")
    [ "$total" -gt 0 ] && pct=$(( cur * 100 / total )) || pct=0
    filled=$(( pct * 24 / 100 )); empty=$(( 24 - filled ))
    bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' "$empty" '' | tr ' ' '.')"
    if [ "$cur" -gt 0 ] && [ "$cur" -lt "$total" ]; then
      eta=$(( el * (total - cur) / cur )); eta_str="$(fmt_hms "$eta")"
    elif [ "$total" -gt 0 ] && [ "$cur" -ge "$total" ]; then eta_str="done"
    else eta_str="--"; fi
    printf '  %-26s [%s] %3d%%  %2d/%-2d  el %s  eta %s\033[K\n' \
           "$nb" "$bar" "$pct" "$cur" "$total" "$(fmt_hms "$el")" "$eta_str"
  done
}

N=${#NOTEBOOKS[@]}; first=1
while :; do
  [ "$first" -eq 1 ] && first=0 || printf '\033[%dA' "$N"
  draw
  alive=0
  for pid in "${PIDS[@]}"; do is_running "$pid" && alive=1; done
  [ "$alive" -eq 0 ] && break
  sleep "$POLL"
done

echo "==> [3/3] collecting results"
fail=0
for i in "${!NOTEBOOKS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "    OK    ${NOTEBOOKS[$i]}"
  else
    fail=1
    echo "    FAILED ${NOTEBOOKS[$i]}  (see ${LOGS[$i]})"
  fi
done

if [ "$fail" -eq 0 ]; then
  echo "==> done. Outputs embedded in ${#NOTEBOOKS[@]} notebooks in $DIR."
else
  echo "==> finished with FAILURES. Notebooks still saved (allow_errors); check logs in $LOG_DIR." >&2
  exit 1
fi
