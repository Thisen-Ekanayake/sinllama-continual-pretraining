#!/usr/bin/env python3
"""Run a benchmark/mrlbenchmarks/evaluate_piqa_*.py evaluator unchanged on the 16 GB lab box.
Only C.load_model is swapped: bf16 + sdpa as before, decoder on GPU, lm_head (139k x 4096) on CPU (the scorer asks
for last-position logits only), transformers 4.46 kwarg name translated (logits_to_keep -> num_logits_to_keep).
usage (repo root, PYTHONPATH=.): python piqa_16gb.py <parallel|nonparallel> <sinhala|english> <evaluator args...>"""
import importlib, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import benchmark.mrlbenchmarks.common as C


def load_model(path, quant="bf16"):
    assert quant == "bf16", quant
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side, tok.truncation_side = "left", "left"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                 attn_implementation="sdpa")
    model.model.to("cuda")
    model.lm_head.register_forward_pre_hook(lambda m, a: (a[0].to("cpu"),))
    model.eval()
    fwd = model.forward

    def forward(*a, logits_to_keep=None, **kw):
        if logits_to_keep is not None:
            kw["num_logits_to_keep"] = logits_to_keep
        assert kw.get("num_logits_to_keep"), "full-sequence logits would be computed on CPU"
        return fwd(*a, **kw)
    model.forward = forward
    print(f"  GPU after load: {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    return model, tok


C.load_model = load_model
variant, lang = sys.argv[1], sys.argv[2]
sys.argv = [f"evaluate_piqa_{variant}_{lang}.py"] + sys.argv[3:]
importlib.import_module(f"benchmark.mrlbenchmarks.evaluate_piqa_{variant}_{lang}").main()
