#!/usr/bin/env python3
"""
Evaluate a Hugging Face Hub model on SinhalaMMLU on a remote GPU pod, with the
exact scorers this repo already uses (evaluate_sinhala_mmlu.py / permute_eval.py).

Runs LOCALLY and drives the pod over SSH:

  1. push   -- scp the SinhalaMMLU dataset, the prompt file and the eval scripts
               to <remote-dir> on the pod
  2. setup  -- on the pod: snapshot_download the model (safetensors only, no
               GGUF), then make it loadable by the pod's transformers 4.x (see
               PATCHING below) and verify the patch took
  3. run    -- on the pod, detached (nohup), so a dropped SSH session doesn't
               kill it: the three arms of the position-bias study --
                 raw         evaluate_sinhala_mmlu.py  (3-shot, bs 32, max_len 4096
                             -- same values as config.yml)
                 calibrated  evaluate_sinhala_mmlu.py --calibrate
                 permuted    permute_eval.py --rotations 0  (the numbers to report)
  4. fetch  -- scp the result directories back into benchmark/mmlu/

PATCHING
--------
Models saved with transformers 5.x don't load correctly under 4.x, and one of
the failures is SILENT:
  * config.json stores RoPE theta under the v5 key `rope_parameters`. 4.x ignores
    it and falls back to rope_theta=10000 -- the model loads fine and scores
    garbage. We copy it to the top-level `rope_theta` (and `rope_scaling`).
  * tokenizer_config.json names `tokenizer_class: TokenizersBackend` (v5-only)
    and gives `extra_special_tokens` as a list (4.x expects a dict). We switch to
    PreTrainedTokenizerFast (same tokenizer.json) and move the list to
    `additional_special_tokens`.
The originals are kept as *.orig. After patching, the setup step loads the
config + tokenizer under the pod's transformers and asserts rope_theta and the
digit tokens, so a bad patch fails before any GPU time is spent.

Usage
-----
  python benchmark/mmlu/eval_hf_model_on_pod.py                 # all steps
  python benchmark/mmlu/eval_hf_model_on_pod.py --steps fetch   # pull results
  python benchmark/mmlu/eval_hf_model_on_pod.py --steps status  # tail the log
  python benchmark/mmlu/eval_hf_model_on_pod.py --limit 5 --arms raw  # smoke test
"""
import argparse
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# files the scorers need (permute_eval imports both evaluator modules)
SCRIPTS = ["evaluate_sinhala_mmlu.py", "evaluate_english_mmlu.py", "permute_eval.py"]
PROMPT = os.path.join(REPO, "benchmark", "prompts", "mmlu_sinhala.txt")
DATASET = os.path.join(HERE, "SinhalaMMLU")

OUT_DIRS = {"raw": "SinhalaMMLU_results_final",
            "calibrated": "SinhalaMMLU_results_calibrated",
            "permuted": "SinhalaMMLU_results_permuted"}


# ----------------------------------------------------------------------------- #
# Remote-side code. Sent over ssh as a python heredoc, so it runs with whatever
# python/transformers the pod has -- which is exactly what we must check.
# ----------------------------------------------------------------------------- #
REMOTE_SETUP = r'''
import json, os, shutil, sys
from huggingface_hub import snapshot_download

repo, local = sys.argv[1], sys.argv[2]
print(f"downloading {repo} -> {local}  (no-op for files already present)")
snapshot_download(repo, local_dir=local,
                  allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])

def load(p):  return json.load(open(p, encoding="utf-8"))
def save(p, d): json.dump(d, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
def backup(p):
    if not os.path.exists(p + ".orig"):
        shutil.copy(p, p + ".orig")

# --- config.json: v5 rope_parameters -> v4 rope_theta / rope_scaling -------- #
cp = os.path.join(local, "config.json")
cfg = load(cp)
rp = cfg.get("rope_parameters")
if rp and "rope_theta" not in cfg:
    backup(cp)
    cfg["rope_theta"] = rp["rope_theta"]
    rest = {k: v for k, v in rp.items() if k != "rope_theta"}
    cfg["rope_scaling"] = None if rest.get("rope_type", "default") == "default" else rest
    if "dtype" in cfg and "torch_dtype" not in cfg:
        cfg["torch_dtype"] = cfg["dtype"]
    save(cp, cfg)
    print(f"patched config.json: rope_theta={cfg['rope_theta']}")
want_theta = (load(cp + ".orig") if os.path.exists(cp + ".orig") else cfg) \
    .get("rope_parameters", {}).get("rope_theta", cfg.get("rope_theta"))

# --- tokenizer_config.json: v5 class / list-form special tokens -------------- #
tp = os.path.join(local, "tokenizer_config.json")
tc = load(tp)
changed = False
if tc.get("tokenizer_class") == "TokenizersBackend":
    tc["tokenizer_class"] = "PreTrainedTokenizerFast"; changed = True
if isinstance(tc.get("extra_special_tokens"), list):
    extra = tc.pop("extra_special_tokens")
    tc["additional_special_tokens"] = [t for t in extra if t != tc.get("eos_token")]
    changed = True
for k in ("backend", "is_local", "local_files_only"):
    if k in tc:
        tc.pop(k); changed = True
# the generic fast tokenizer emits token_type_ids by default; Llama takes none
if tc.get("tokenizer_class") == "PreTrainedTokenizerFast" and "model_input_names" not in tc:
    tc["model_input_names"] = ["input_ids", "attention_mask"]; changed = True
if changed:
    backup(tp); save(tp, tc)
    print("patched tokenizer_config.json")

# --- verify under THIS transformers ----------------------------------------- #
import transformers
from transformers import AutoConfig, AutoTokenizer
c = AutoConfig.from_pretrained(local)
t = AutoTokenizer.from_pretrained(local)
print(f"transformers {transformers.__version__}: rope_theta={c.rope_theta} "
      f"(want {want_theta}), vocab={c.vocab_size}, tokenizer={type(t).__name__} "
      f"len={len(t)}")
assert float(c.rope_theta) == float(want_theta), "rope_theta patch did not take"
digits = [t.encode(str(d), add_special_tokens=False) for d in range(1, 10)]
assert all(len(x) == 1 for x in digits) and len({x[0] for x in digits}) == 9, \
    f"answer digits are not distinct single tokens: {digits}"
s = "ශ්‍රී ලංකාව"
assert t.decode(t.encode(s, add_special_tokens=False)) == s, "tokenizer round-trip failed"
assert set(t(s).keys()) == {"input_ids", "attention_mask"}, f"unexpected inputs {list(t(s))}"
print(f"digit ids {[x[0] for x in digits]}; Sinhala round-trip ok; "
      f"'{s}' -> {len(t.encode(s, add_special_tokens=False))} tokens")
print("SETUP OK")
'''


def run_script(args):
    """Shell script that runs the requested arms on the pod, one after another,
    skipping arms whose metrics file already exists (so it is resumable)."""
    name = os.path.basename(args.local_model_dir.rstrip("/"))
    rd = args.remote_dir
    model = args.local_model_dir
    common = (f"--models {shlex.quote(model)} --data-root {rd}/SinhalaMMLU "
              f"--prompt-file {rd}/mmlu_sinhala.txt --max-len {args.max_len}")
    if args.limit:
        common += f" --limit {args.limit}"
    cmds = {
        "raw": (f"python -u {rd}/evaluate_sinhala_mmlu.py {common} --kshot 3 "
                f"--batch-size {args.batch_size} --out-dir {rd}/{OUT_DIRS['raw']}"),
        "calibrated": (f"python -u {rd}/evaluate_sinhala_mmlu.py {common} --kshot 3 "
                       f"--batch-size {args.batch_size} --calibrate "
                       f"--out-dir {rd}/{OUT_DIRS['calibrated']}"),
        "permuted": (f"python -u {rd}/permute_eval.py --lang sinhala {common} "
                     f"--batch-size {args.batch_size} --rotations 0 "
                     f"--out-dir {rd}/{OUT_DIRS['permuted']}"),
    }
    lines = ["set -uo pipefail",
             "export TOKENIZERS_PARALLELISM=false",
             "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
             f"cd {rd}"]
    for arm in args.arms:
        out = f"{rd}/{OUT_DIRS[arm]}/{name}_metrics.json"
        lines += [
            f'if [[ -s {out} && "${{FORCE:-0}}" != 1 ]]; then '
            f'echo "[$(date +%T)] skip {arm} (done)"; else',
            f'  echo "[$(date +%T)] === {arm} ==="',
            f"  {cmds[arm]} || echo \"[$(date +%T)] {arm} FAILED rc=$?\"",
            "fi"]
    lines.append('echo "[$(date +%T)] ALL DONE"')
    return "\n".join(lines) + "\n"


def sh(cmd, **kw):
    print("+", " ".join(shlex.quote(c) for c in cmd) if isinstance(cmd, list) else cmd)
    return subprocess.run(cmd, check=True, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="root@194.68.245.38")
    ap.add_argument("--port", default="22149")
    ap.add_argument("--key", default=os.path.expanduser("~/.ssh/id_ed25519"))
    ap.add_argument("--model", default="namindu94/SinhalaLLM-1.7B",
                    help="HF Hub repo id")
    ap.add_argument("--remote-dir", default="/workspace/mmlu_eval")
    ap.add_argument("--local-model-dir", default=None,
                    help="where the model lands on the pod "
                         "(default /workspace/models/<repo name>)")
    ap.add_argument("--arms", nargs="+", default=["raw", "calibrated", "permuted"],
                    choices=list(OUT_DIRS))
    ap.add_argument("--steps", nargs="+",
                    default=["push", "setup", "run"],
                    choices=["push", "setup", "run", "status", "fetch"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0,
                    help="questions per TEST file, for a smoke test (0 = all)")
    args = ap.parse_args()
    if args.local_model_dir is None:
        args.local_model_dir = f"/workspace/models/{args.model.split('/')[-1]}"

    ssh = ["ssh", "-p", args.port, "-i", args.key,
           "-o", "StrictHostKeyChecking=accept-new", args.host]
    scp = ["scp", "-P", args.port, "-i", args.key, "-q"]
    rd = args.remote_dir
    log = f"{rd}/run_{os.path.basename(args.local_model_dir)}.log"

    if "push" in args.steps:
        sh(ssh + [f"mkdir -p {rd} {os.path.dirname(args.local_model_dir)}"])
        sh(scp + ["-r", DATASET, PROMPT] + [os.path.join(HERE, s) for s in SCRIPTS]
           + [f"{args.host}:{rd}/"])

    if "setup" in args.steps:
        sh(ssh + [f"python3 - {shlex.quote(args.model)} "
                  f"{shlex.quote(args.local_model_dir)}"], input=REMOTE_SETUP.encode())

    if "run" in args.steps:
        script = f"{rd}/run_{os.path.basename(args.local_model_dir)}.sh"
        sh(ssh + [f"cat > {script}"], input=run_script(args).encode())
        sh(ssh + [f"nohup bash {script} > {log} 2>&1 < /dev/null & echo started pid $!"])
        print(f"\nRunning detached on the pod. Check progress with:\n"
              f"  python {os.path.relpath(__file__, REPO)} --steps status\n"
              f"and pull results when it prints ALL DONE with:\n"
              f"  python {os.path.relpath(__file__, REPO)} --steps fetch")

    if "status" in args.steps:
        sh(ssh + [f"tail -c 3000 {log} | tr '\\r' '\\n' | tail -15; "
                  "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"])

    if "fetch" in args.steps:
        name = os.path.basename(args.local_model_dir.rstrip("/"))
        for arm in args.arms:
            dst = os.path.join(HERE, OUT_DIRS[arm])
            os.makedirs(dst, exist_ok=True)
            # only this model's files: never overwrite other models' results or
            # the combined results.md that was built from them
            r = subprocess.run(scp + [f"{args.host}:{rd}/{OUT_DIRS[arm]}/{name}_*", dst + "/"])
            print(f"  {arm}: {'ok' if r.returncode == 0 else 'nothing yet'} -> {dst}")
        sh(scp + [f"{args.host}:{log}", HERE + "/"])


if __name__ == "__main__":
    sys.exit(main())
