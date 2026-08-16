"""Verify a merged model really is base + the LoRA adapter, and nothing else.

Reconstructs `B @ A * (alpha/r)` from the adapter and compares it against
`merged - base` for a few tensors, then confirms the tensors that were supposed
to stay frozen are bit-identical. Reads single tensors out of the safetensors
shards, so it never loads a full model.

    python sft/verify_merge.py
    python sft/verify_merge.py --base models/SinLlama_v02 \
        --merged models/SinLlama_uc_instruct \
        --adapter sft/runs/sft_uc_lora/early_stop

On the residual tolerance: the merged model is stored in bf16, so recovering dW
as a difference of two bf16 tensors carries relative error ~2^-9 * |W|/|dW|.
That is 0.9% where dW is 22% of W but 2.9% where it is 6.6%, so the check scales
its tolerance with the observed dW rather than using a flat threshold.
"""
import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file

DEFAULT_TENSORS = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.15.self_attn.v_proj.weight",
    "model.layers.31.mlp.down_proj.weight",
]
FROZEN = ["model.embed_tokens.weight", "lm_head.weight"]
BF16_EPS = 2.0 ** -9


def get(root: Path, name: str):
    idx = json.load(open(root / "model.safetensors.index.json"))["weight_map"]
    with safe_open(root / idx[name], framework="pt") as f:
        return f.get_tensor(name).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/SinLlama_v02")
    ap.add_argument("--merged", default="models/SinLlama_uc_instruct")
    ap.add_argument("--adapter", default="sft/runs/sft_uc_lora/early_stop")
    ap.add_argument("--tensors", nargs="*", default=DEFAULT_TENSORS)
    args = ap.parse_args()

    base, merged, adapter = Path(args.base), Path(args.merged), Path(args.adapter)

    cfg = json.load(open(adapter / "adapter_config.json"))
    scale = cfg["lora_alpha"] / cfg["r"]
    print(f"base    : {base}")
    print(f"merged  : {merged}")
    print(f"adapter : {adapter}  (r={cfg['r']} alpha={cfg['lora_alpha']} "
          f"scale={scale} modules_to_save={cfg.get('modules_to_save')})")
    A = load_file(adapter / "adapter_model.safetensors")

    failures = 0
    print()
    print(f"{'tensor':<44} {'|dW|/|W|':>9} {'residual':>9} {'tol':>7}  verdict")
    for n in args.tensors:
        w0 = get(base, n)
        d = get(merged, n) - w0
        p = "base_model.model." + n[: -len(".weight")]
        exp = (A[p + ".lora_B.weight"].float() @ A[p + ".lora_A.weight"].float()) * scale
        rel = (d.norm() / w0.norm()).item()
        resid = ((d - exp).norm() / exp.norm().clamp(min=1e-12)).item()
        tol = max(3.0 * BF16_EPS / max(rel, 1e-9), 1e-3)
        ok = resid <= tol
        failures += not ok
        print(f"{n:<44} {rel * 100:>8.2f}% {resid * 100:>8.2f}% {tol * 100:>6.2f}%  "
              f"{'ok' if ok else 'MISMATCH'}")

    print()
    for n in FROZEN:
        d = (get(merged, n) - get(base, n)).abs().max().item()
        failures += d != 0
        print(f"{n:<44} max|delta|={d:.2e}  "
              f"{'frozen (bit-identical)' if d == 0 else 'CHANGED -- expected frozen'}")

    print()
    print("PASS" if not failures else f"FAIL ({failures} check(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
