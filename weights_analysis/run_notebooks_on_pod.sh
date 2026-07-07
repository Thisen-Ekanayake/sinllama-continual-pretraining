#!/usr/bin/env bash
#
# Run both weights-analysis notebooks on the MI300X pod and download the
# executed copies (with all outputs embedded) back to this machine.
#
# Why the pod: the attention notebook's Part B runs full 8B forward passes,
# which need the MI300X GPU + the models that live on the pod. The pod's venv
# already has torch/transformers/sklearn/umap; this script adds the few missing
# pieces (seaborn, nbconvert, ipykernel, tqdm).
#
# Execution uses a tiny nbclient driver (uploaded to the pod) that runs EVERY
# code cell in-place and shows a tqdm progress bar with a live ETA per notebook.
# Each cell's output — prints, DataFrame tables, and the inline matplotlib
# figures from plt.show() — is written straight into the .ipynb, so the
# downloaded notebooks show all outputs in Jupyter / VS Code. (Figures are also
# saved as PNGs on the pod and optionally fetched below.)
#
# Usage:   bash weights_analysis/run_notebooks_on_pod.sh
# Override any of these via env vars, e.g.  SUFFIX="" bash run_notebooks_on_pod.sh
set -euo pipefail

POD="${POD:-root@165.245.134.128}"
VENV_PY="${VENV_PY:-/root/sinllama-continual-pretraining/venv/bin/python}"
REMOTE_DIR="${REMOTE_DIR:-/root/sinllama-continual-pretraining/weights_analysis}"
LOCAL_DIR="${LOCAL_DIR:-/ml/SinLlama_CPT/weights_analysis}"
KERNEL="${KERNEL:-sinllama}"                 # kernel name nbconvert will execute with
SUFFIX="${SUFFIX:-_executed}"                # downloaded files: <name><SUFFIX>.ipynb
FETCH_FIGURES="${FETCH_FIGURES:-1}"          # also download the PNG figure dirs
NOTEBOOKS=(embedding_analysis.ipynb attention_analysis.ipynb)

echo "==> [1/6] preparing pod venv (seaborn / nbconvert / ipykernel / tqdm) + kernel '$KERNEL'"
ssh "$POD" "$VENV_PY -m pip install -q --upgrade seaborn nbconvert ipykernel jupyter_client tqdm \
            && $VENV_PY -m ipykernel install --user --name '$KERNEL' --display-name '$KERNEL'"

echo "==> [2/6] uploading current notebooks to the pod"
ssh "$POD" "mkdir -p '$REMOTE_DIR'"
for nb in "${NOTEBOOKS[@]}"; do
  [ -f "$LOCAL_DIR/$nb" ] || { echo "ERROR: missing $LOCAL_DIR/$nb" >&2; exit 1; }
  scp "$LOCAL_DIR/$nb" "$POD:$REMOTE_DIR/$nb"
done

echo "==> [3/6] uploading the progress-aware executor to the pod"
# Small nbclient driver: runs every code cell in-place (allow_errors, no per-cell
# timeout) and shows a tqdm progress bar with a live ETA for each notebook.
ssh "$POD" "cat > '$REMOTE_DIR/_exec_progress.py'" <<'PYEOF'
import os, sys, nbformat
from nbclient import NotebookClient
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

path, kernel = sys.argv[1], sys.argv[2]
name = os.path.basename(path)
nb = nbformat.read(path, as_version=4)
code_idx = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
run_dir = os.path.dirname(os.path.abspath(path)) or "."
client = NotebookClient(nb, timeout=-1, kernel_name=kernel, allow_errors=True,
                        resources={"metadata": {"path": run_dir}})
bar = tqdm(total=len(code_idx), unit="cell", desc=name,
           dynamic_ncols=True) if tqdm else None
try:
    with client.setup_kernel():
        for n, i in enumerate(code_idx, 1):
            client.execute_cell(nb.cells[i], i)     # runs cell, stores its outputs
            if bar is not None:
                bar.update(1)
            else:
                print(f"  {name}: cell {n}/{len(code_idx)}", flush=True)
finally:
    if bar is not None:
        bar.close()
    nbformat.write(nb, path)                        # save outputs (even if aborted)
PYEOF

echo "==> [4/6] executing notebooks on the pod (full 8B forwards run on the MI300X)"
TTY_FLAG=""; [ -t 1 ] && TTY_FLAG="-t"       # request a pty so the bar renders live
for nb in "${NOTEBOOKS[@]}"; do
  echo "    - running $nb ..."
  ssh $TTY_FLAG "$POD" "cd '$REMOTE_DIR' && '$VENV_PY' _exec_progress.py '$nb' '$KERNEL'"
done

echo "==> [5/6] downloading executed notebooks to this machine"
for nb in "${NOTEBOOKS[@]}"; do
  dest="$LOCAL_DIR/${nb%.ipynb}${SUFFIX}.ipynb"
  scp "$POD:$REMOTE_DIR/$nb" "$dest"
  echo "    saved $dest"
done

if [ "$FETCH_FIGURES" = "1" ]; then
  echo "==> [6/6] downloading PNG figure directories"
  for d in figures figures_attention; do
    if ssh "$POD" "[ -d '$REMOTE_DIR/$d' ]"; then
      scp -r "$POD:$REMOTE_DIR/$d" "$LOCAL_DIR/"
      echo "    saved $LOCAL_DIR/$d/"
    fi
  done
else
  echo "==> [6/6] skipping figure download (FETCH_FIGURES=0)"
fi

ssh "$POD" "rm -f '$REMOTE_DIR/_exec_progress.py'" 2>/dev/null || true
echo "==> done. Open the *${SUFFIX}.ipynb files — every cell's output is embedded."
