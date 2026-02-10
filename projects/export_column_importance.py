import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "./SinLlama_merged"
VAL_TEXT_FILE = "./eval.txt"
OUTPUT_FILE = "./column_importance.pt"
MAX_LENGTH = 128

torch.set_default_dtype(torch.float32)
device = torch.device("cpu")

# -----------------------------
# Load model & tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": "cpu"},
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()

# -----------------------------
# Load validation text
# -----------------------------
with open(VAL_TEXT_FILE, "r", encoding="utf-8") as f:
    texts = [l.strip() for l in f if l.strip()]

# -----------------------------
# Storage
# -----------------------------
column_importance = defaultdict(lambda: None)
counts = defaultdict(int)

# -----------------------------
# Hook
# -----------------------------
def make_hook(name, weight):
    def hook(module, inp, out):
        x = inp[0].detach().cpu()
        if x.dim() == 3:
            col_scale = x.abs().mean(dim=(0, 1))
        else:
            col_scale = x.abs().mean(dim=0)

        W = weight.detach().cpu().abs()
        imp = (W * col_scale.unsqueeze(0)).sum(dim=0)

        if column_importance[name] is None:
            column_importance[name] = imp.clone()
        else:
            column_importance[name] += imp

        counts[name] += 1
    return hook

# -----------------------------
# Register hooks
# -----------------------------
hooks = []
for name, module in model.named_modules():
    if ("mlp" in name or "self_attn" in name) and isinstance(module, torch.nn.Linear):
        hooks.append(
            module.register_forward_hook(make_hook(name, module.weight))
        )

# -----------------------------
# Run forward passes
# -----------------------------
with torch.no_grad():
    for text in tqdm(texts, desc="Collecting column importance"):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        _ = model(**enc)

# -----------------------------
# Normalize + save
# -----------------------------
for k in column_importance:
    column_importance[k] /= counts[k]

torch.save(column_importance, OUTPUT_FILE)

# Cleanup
for h in hooks:
    h.remove()

print(f"\nSaved column importance to: {OUTPUT_FILE}")
print(f"Total layers saved: {len(column_importance)}")