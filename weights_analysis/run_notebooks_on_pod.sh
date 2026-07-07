#!/usr/bin/env bash
#
# Execute both weights-analysis notebooks IN PLACE on THIS machine (the MI300X
# pod), running every code cell with a tqdm progress bar + live ETA so all
# outputs — prints, DataFrame tables, and the inline matplotlib figures from
# plt.show() — are embedded straight into the .ipynb files.
#
# Run this ON the pod (the notebooks + models are already local here); it does
# NOT ssh or copy anything. The attention notebook's Part B runs the full 8B
# forward passes on the MI300X GPU.
#
# Usage:   bash weights_analysis/run_notebooks_on_pod.sh
#          SUFFIX=_executed bash weights_analysis/run_notebooks_on_pod.sh   # write copies instead
set -euo pipefail

VENV_PY="${VENV_PY:-$HOME/sinllama-continual-pretraining/venv/bin/python}"
DIR="${DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"   # dir holding the notebooks
KERNEL="${KERNEL:-sinllama}"
SUFFIX="${SUFFIX:-}"                          # "" = in place; else write <name><SUFFIX>.ipynb
NOTEBOOKS=(embedding_analysis.ipynb attention_analysis.ipynb)

echo "==> [1/2] ensuring execution deps + kernel '$KERNEL'"
"$VENV_PY" -c 'import nbclient, nbconvert, ipykernel, tqdm, seaborn' 2>/dev/null \
  || "$VENV_PY" -m pip install -q --upgrade seaborn nbconvert nbclient ipykernel tqdm
[ -d "$HOME/.local/share/jupyter/kernels/$KERNEL" ] \
  || "$VENV_PY" -m ipykernel install --user --name "$KERNEL" --display-name "$KERNEL"

# Progress-aware executor: runs each code cell in-place (allow_errors, no per-cell
# timeout) and drives a tqdm bar with a live ETA. Written to a temp file.
DRIVER="$(mktemp --suffix=.py)"
trap 'rm -f "$DRIVER"' EXIT
cat > "$DRIVER" <<'PYEOF'
import os, sys, nbformat
from nbclient import NotebookClient
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

path, kernel = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else path
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
    nbformat.write(nb, out)                         # save outputs (even if aborted)
PYEOF

echo "==> [2/2] executing notebooks (outputs embedded${SUFFIX:+ into *$SUFFIX.ipynb})"
for nb in "${NOTEBOOKS[@]}"; do
  src="$DIR/$nb"
  [ -f "$src" ] || { echo "ERROR: $src not found" >&2; exit 1; }
  if [ -n "$SUFFIX" ]; then out="$DIR/${nb%.ipynb}${SUFFIX}.ipynb"; else out="$src"; fi
  echo "    - $nb"
  "$VENV_PY" "$DRIVER" "$src" "$KERNEL" "$out"
done

echo "==> done. Outputs embedded in the 2 notebooks in $DIR."
