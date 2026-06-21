"""
Merge a LoRA adapter trained by `run_clm_pt_with_peft.py` (see `run_pt.sh`)
back into the `SinLlama_merged_bf16` base model, producing a standalone HF model.

The training in `run_pt.sh` does NOT pass `--modules_to_save`, so the adapter
only contains LoRA `lora_A` / `lora_B` matrices for the target projections:
    q_proj, k_proj, v_proj, o_proj, gate_proj, down_proj, up_proj
`embed_tokens` and `lm_head` are frozen and are taken verbatim from the base.

The merge is done shard-by-shard to keep peak memory low: each base safetensors
shard is loaded, its LoRA-targeted weights get `W += (alpha / r) * (B @ A)`
folded in, and the shard is written straight back out. The adapter is opened
lazily so only the two small LoRA matrices needed per weight are ever held.

Usage:
    python merge_sinllama_lora_low_mem.py \
        --base_model ../models/SinLlama_merged_bf16 \
        --lora_model output_dir \
        --output_dir ../models/SinLlama_cpt_merged
"""
import argparse
import gc
import json
import os
import re

import peft
import safetensors
import torch
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from transformers import AutoTokenizer

WEIGHTS_NAME = "adapter_model.bin"
SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors"

parser = argparse.ArgumentParser(
    description="Merge a SinLlama LoRA adapter into the SinLlama_merged_bf16 base model (low memory)."
)
parser.add_argument("--base_model", required=True, type=str,
                    help="Path to the base model (e.g. SinLlama_merged_bf16, HF format).")
parser.add_argument("--lora_model", required=True, type=str,
                    help="Path to the trained LoRA adapter (the training `output_dir`).")
parser.add_argument("--output_dir", default="./SinLlama_merged_lora", type=str,
                    help="Where to write the merged model.")
parser.add_argument("--verbose", action="store_true",
                    help="Print every weight that gets a LoRA delta folded in.")


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def list_base_shards(base_model_path):
    """Return (shard_filenames, is_safetensors) for the base model."""
    if os.path.exists(os.path.join(base_model_path, "model.safetensors.index.json")):
        idx = json.load(open(os.path.join(base_model_path, "model.safetensors.index.json")))
        return sorted(set(idx["weight_map"].values())), True
    if os.path.exists(os.path.join(base_model_path, "model.safetensors")):
        return ["model.safetensors"], True
    if os.path.exists(os.path.join(base_model_path, "pytorch_model.bin.index.json")):
        idx = json.load(open(os.path.join(base_model_path, "pytorch_model.bin.index.json")))
        return sorted(set(idx["weight_map"].values())), False
    if os.path.exists(os.path.join(base_model_path, "pytorch_model.bin")):
        return ["pytorch_model.bin"], False
    raise FileNotFoundError(
        f"Cannot find HF-format checkpoints in {base_model_path}."
    )


def main():
    args = parser.parse_args()
    base_model_path = args.base_model
    lora_model_path = args.lora_model
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"Base model : {base_model_path}")
    print(f"LoRA model : {lora_model_path}")
    print(f"Output dir : {output_dir}")

    # ---- Load LoRA config + open adapter weights lazily -------------------
    lora_config = peft.LoraConfig.from_pretrained(lora_model_path)
    scaling = lora_config.lora_alpha / lora_config.r
    if getattr(lora_config, "use_rslora", False):
        scaling = lora_config.lora_alpha / (lora_config.r ** 0.5)
    fan_in_fan_out = lora_config.fan_in_fan_out
    print(f"LoRA r={lora_config.r}, alpha={lora_config.lora_alpha}, "
          f"scaling={scaling:.6f}, fan_in_fan_out={fan_in_fan_out}")
    print(f"target_modules: {sorted(lora_config.target_modules)}")
    if lora_config.modules_to_save:
        print(f"NOTE: adapter declares modules_to_save={lora_config.modules_to_save}; "
              "those full weights will be applied if present.")

    if os.path.exists(os.path.join(lora_model_path, SAFETENSORS_WEIGHTS_NAME)):
        adapter_st = safe_open(
            os.path.join(lora_model_path, SAFETENSORS_WEIGHTS_NAME),
            framework="pt", device="cpu",
        )
        adapter_keys = set(adapter_st.keys())
        get_adapter = adapter_st.get_tensor
    elif os.path.exists(os.path.join(lora_model_path, WEIGHTS_NAME)):
        adapter_sd = torch.load(os.path.join(lora_model_path, WEIGHTS_NAME), map_location="cpu")
        adapter_keys = set(adapter_sd.keys())
        get_adapter = adapter_sd.__getitem__
    else:
        raise FileNotFoundError(
            f"No {SAFETENSORS_WEIGHTS_NAME} or {WEIGHTS_NAME} found in {lora_model_path}."
        )

    consumed = set()  # adapter keys we actually used, for a sanity check at the end

    # ---- Walk the base shards and fold in the LoRA deltas ------------------
    shard_files, base_is_safetensors = list_base_shards(base_model_path)
    print(f"Found {len(shard_files)} base shard(s).")

    for filename in shard_files:
        path = os.path.join(base_model_path, filename)
        print(f"\nLoading base shard {filename} ...")
        if base_is_safetensors:
            state_dict = safe_load_file(path, device="cpu")
        else:
            state_dict = torch.load(path, map_location="cpu")

        merged_here = 0
        for k in list(state_dict.keys()):
            prefix = "base_model.model." + k[: -len(".weight")]  # strip trailing ".weight"

            # Full-weight override (only present if trained with modules_to_save).
            mts_key = prefix + ".modules_to_save.weight"
            if mts_key in adapter_keys:
                orig_dtype = state_dict[k].dtype
                state_dict[k] = get_adapter(mts_key).to(orig_dtype).contiguous()
                consumed.add(mts_key)
                if args.verbose:
                    print(f"  override {k} <- {mts_key}")

            # LoRA delta.
            lora_a_key = prefix + ".lora_A.weight"
            lora_b_key = prefix + ".lora_B.weight"
            if lora_a_key in adapter_keys and lora_b_key in adapter_keys:
                A = get_adapter(lora_a_key).float()
                B = get_adapter(lora_b_key).float()
                delta = transpose(B @ A, fan_in_fan_out) * scaling
                orig_dtype = state_dict[k].dtype
                state_dict[k] = (state_dict[k].float() + delta).to(orig_dtype).contiguous()
                consumed.add(lora_a_key)
                consumed.add(lora_b_key)
                merged_here += 1
                if args.verbose:
                    print(f"  merged   {k}  (+= scaling * B@A)")
                del A, B, delta

        out_path = os.path.join(output_dir, filename)
        print(f"Saving merged shard -> {out_path}  ({merged_here} weights merged)")
        if base_is_safetensors:
            safetensors.torch.save_file(state_dict, out_path, metadata={"format": "pt"})
        else:
            torch.save(state_dict, out_path)

        del state_dict
        gc.collect()

    # ---- Sanity check: every LoRA matrix should have been folded in -------
    unused = {k for k in adapter_keys
              if (".lora_A.weight" in k or ".lora_B.weight" in k or ".modules_to_save.weight" in k)
              and k not in consumed}
    if unused:
        print("\nWARNING: the following adapter weights were NOT merged "
              "(no matching base weight found):")
        for k in sorted(unused):
            print(f"  {k}")
    else:
        print("\nAll LoRA / modules_to_save weights were matched and merged.")

    # ---- Copy index + config files, save tokenizer -----------------------
    print("\nCopying config / index files ...")
    for fname in ("config.json", "generation_config.json",
                  "model.safetensors.index.json", "pytorch_model.bin.index.json"):
        src = os.path.join(base_model_path, fname)
        if os.path.exists(src):
            with open(src) as f:
                obj = json.load(f)
            with open(os.path.join(output_dir, fname), "w") as f:
                json.dump(obj, f, indent=2)
            print(f"  {fname}")

    print("Saving tokenizer ...")
    tok_src = lora_model_path if os.path.exists(
        os.path.join(lora_model_path, "tokenizer_config.json")) else base_model_path
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(output_dir)

    print("\nDone.")
    print(f"Merged model written to: {output_dir}")


if __name__ == "__main__":
    main()
