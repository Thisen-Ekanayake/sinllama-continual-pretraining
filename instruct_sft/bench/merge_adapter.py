#!/usr/bin/env python3
"""Merge the instruct-mix LoRA (QLoRA-trained, r64/a16) into SinLlama_wiki_uc_gen_mix in bf16 -> a full model dir.
usage: merge_adapter.py <base> <adapter> <out>
Same arithmetic as permute_16gb.merge_lora (W += alpha/r * B @ A, fp32 accumulate), done by peft on CPU."""
import json, os, shutil, sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base, adapter, out = sys.argv[1:4]
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
w0 = m.model.layers[0].self_attn.q_proj.weight.detach().float().clone()
m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
d = (m.model.layers[0].self_attn.q_proj.weight.float() - w0).abs().mean().item()
print(f"layer-0 q_proj mean |dW| = {d:.3e}")
assert d > 0, "adapter did not change the weights"
m.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(base).save_pretrained(out)
for f in ("generation_config.json",):
    if os.path.exists(f"{base}/{f}"):
        shutil.copy(f"{base}/{f}", f"{out}/{f}")
json.dump({"base": os.path.abspath(base), "adapter": os.path.abspath(adapter)},
          open(f"{out}/merged_from.json", "w"), indent=1)
print("saved", out)
