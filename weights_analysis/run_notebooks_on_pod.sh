#!/usr/bin/env bash
#
# Run both weights-analysis notebooks on the MI300X pod and download the
# executed copies (with all outputs embedded) back to this machine.
#
# Why the pod: the attention notebook's Part B runs full 8B forward passes,
# which need the MI300X GPU + the models that live on the pod. The pod's venv
# already has torch/transformers/sklearn/umap; this script adds the few missing
# pieces (seaborn, nbconvert, ipykernel).
#
# `nbconvert --execute` runs EVERY cell through the kernel and writes each cell's
# output — prints, DataFrame tables, and the inline matplotlib figures produced
# by plt.show() — straight into the .ipynb. So the downloaded notebooks show all
# outputs when opened in Jupyter / VS Code. (Figures are also saved as PNGs on
# the pod and optionally fetched below.)
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

echo "==> [1/5] preparing pod venv (seaborn / nbconvert / ipykernel) + kernel '$KERNEL'"
ssh "$POD" "$VENV_PY -m pip install -q --upgrade seaborn nbconvert ipykernel jupyter_client \
            && $VENV_PY -m ipykernel install --user --name '$KERNEL' --display-name '$KERNEL'"

echo "==> [2/5] uploading current notebooks to the pod"
ssh "$POD" "mkdir -p '$REMOTE_DIR'"
for nb in "${NOTEBOOKS[@]}"; do
  [ -f "$LOCAL_DIR/$nb" ] || { echo "ERROR: missing $LOCAL_DIR/$nb" >&2; exit 1; }
  scp "$LOCAL_DIR/$nb" "$POD:$REMOTE_DIR/$nb"
done

echo "==> [3/5] executing notebooks on the pod (full 8B forwards run on the MI300X)"
# --inplace      : write outputs back into the uploaded copy on the pod
# --allow-errors : don't abort the whole run on one failing cell; the traceback
#                  is embedded in that cell so the rest still executes
# timeout=-1     : no per-cell time limit (long ablation / model loads are fine)
for nb in "${NOTEBOOKS[@]}"; do
  echo "    - running $nb ..."
  ssh "$POD" "cd '$REMOTE_DIR' && $VENV_PY -m nbconvert --to notebook --execute --inplace \
      --allow-errors \
      --ExecutePreprocessor.kernel_name='$KERNEL' \
      --ExecutePreprocessor.timeout=-1 '$nb'"
done

echo "==> [4/5] downloading executed notebooks to this machine"
for nb in "${NOTEBOOKS[@]}"; do
  dest="$LOCAL_DIR/${nb%.ipynb}${SUFFIX}.ipynb"
  scp "$POD:$REMOTE_DIR/$nb" "$dest"
  echo "    saved $dest"
done

if [ "$FETCH_FIGURES" = "1" ]; then
  echo "==> [5/5] downloading PNG figure directories"
  for d in figures figures_attention; do
    if ssh "$POD" "[ -d '$REMOTE_DIR/$d' ]"; then
      scp -r "$POD:$REMOTE_DIR/$d" "$LOCAL_DIR/"
      echo "    saved $LOCAL_DIR/$d/"
    fi
  done
else
  echo "==> [5/5] skipping figure download (FETCH_FIGURES=0)"
fi

echo "==> done. Open the *${SUFFIX}.ipynb files — every cell's output is embedded."
