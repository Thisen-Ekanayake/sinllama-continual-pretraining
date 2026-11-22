#!/usr/bin/env python3
"""Run benchmark/mmlu/permute_eval.py unchanged on a 16 GB card.

Only the model loader is swapped:
  * bf16 weights (same precision as every other row in MMLU_Debiased.tex),
    eager attention (the evaluators assert it);
  * lm_head (139k x 4096) stays on CPU so the decoder fits in 16 GB; only the
    last position's logits are computed, so that is one small CPU matmul;
  * "SinLlama_textbook_mix" = SinLlama_wiki_uc_gen_mix + the step-500 LoRA
    merged in place (W += alpha/r * B @ A), i.e. what
    cpt/merge_sinllama_lora_low_mem.py wrote to gs://sinllama_cpt/SinLlama_textbook_mix.
  * transformers 4.46 names the kwarg num_logits_to_keep; permute_eval passes
    logits_to_keep, so it is translated.

usage (repo root): python permute_16gb.py <permute_eval args...>
  --models takes model directories (models/<name>); names in ADAPTERS are
  built from PARENT + that adapter instead.
"""
import json
import os
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "benchmark/mmlu")
import evaluate_english_mmlu as EN   # noqa: E402
import evaluate_sinhala_mmlu as SI   # noqa: E402
import permute_eval as PE            # noqa: E402

PARENT = "models/SinLlama_wiki_uc_gen_mix"
ADAPTERS = {  # name -> LoRA dir merged onto PARENT
    "SinLlama_textbook_mix": os.path.expanduser("~/tb_eval/textbook_mix_ckpt500"),
}


@torch.no_grad()
def merge_lora(model, adapter_dir):
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    assert not cfg.get("use_rslora") and not cfg.get("use_dora"), cfg
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    mods = dict(model.named_modules())
    n = 0
    for k, a in sd.items():
        if ".lora_A." not in k:
            continue
        b = sd[k.replace(".lora_A.", ".lora_B.")]
        name = k.split(".lora_A.")[0].replace("base_model.model.", "", 1)
        lin = mods[name]
        w = lin.weight
        delta = (b.to(w.device, torch.float32) @ a.to(w.device, torch.float32)) * scale
        w.copy_((w.float() + delta).to(w.dtype))
        n += 1
    print(f"  merged {n} LoRA pairs from {adapter_dir} (scale {scale})")
    assert n == 7 * model.config.num_hidden_layers, n


def load_model(path, load_in_8bit=False):
    name = os.path.basename(path.rstrip("/"))
    src = PARENT if name in ADAPTERS else path
    tok = AutoTokenizer.from_pretrained(src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager")                      # on CPU
    if name in ADAPTERS:
        merge_lora(model, ADAPTERS[name])
    model.model.to("cuda")
    model.lm_head.register_forward_pre_hook(lambda m, a: (a[0].to("cpu"),))
    SI.assert_eager(model, path)
    model.eval()
    fwd = model.forward

    def forward(*a, logits_to_keep=None, **kw):
        if logits_to_keep is not None:
            kw["num_logits_to_keep"] = logits_to_keep
        assert kw.get("num_logits_to_keep"), "full-sequence logits on CPU"
        return fwd(*a, **kw)
    model.forward = forward
    print(f"  GPU after load: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    return model, tok


SI.load_model = load_model
EN.load_model = load_model
if __name__ == "__main__":
    PE.main()
