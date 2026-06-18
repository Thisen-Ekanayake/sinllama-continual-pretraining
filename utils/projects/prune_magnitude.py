import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU ONLY

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
ROOT = Path(__file__).parent.parent
MODEL_PATH = str(ROOT / "SinLlama_merged")
OUTPUT_PATH = str(ROOT / "SinLlama_pruned")

PRUNE_RATIO = 0.5   # prune 50% of weights (start here)
DTYPE = torch.float32

# -----------------------------
# Sanity checks
# -----------------------------
assert not torch.cuda.is_available(), "CUDA must be disabled"

torch.set_default_dtype(DTYPE)
device = torch.device("cpu")

print("✓ CPU-only mode confirmed")

# -----------------------------
# Load model
# -----------------------------
print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": "cpu"},
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
)
model.eval()
model.to(device)

print("✓ Model loaded")

# -----------------------------
# Collect prunable weights
# -----------------------------
print("\nCollecting weights...")
params = []
shapes = []
param_refs = []

for name, param in model.named_parameters():
    if param.requires_grad and param.dim() >= 2:  # skip biases + LN
        tensor = param.data.view(-1)
        params.append(tensor.abs())
        shapes.append(param.data.shape)
        param_refs.append(param)

all_weights = torch.cat(params)
total_params = all_weights.numel()

print(f"✓ Total prunable parameters: {total_params:,}")

# -----------------------------
# Compute threshold
# -----------------------------
k = int(PRUNE_RATIO * total_params)
threshold = torch.kthvalue(all_weights, k).values.item()

print(f"✓ Pruning threshold |w| ≤ {threshold:.6e}")

# -----------------------------
# Apply pruning
# -----------------------------
print("\nApplying pruning...")
pruned_count = 0

for param in tqdm(param_refs):
    mask = param.data.abs() > threshold
    pruned_count += mask.numel() - mask.sum().item()
    param.data.mul_(mask)

print(f"✓ Pruned parameters: {pruned_count:,}")
print(f"✓ Pruned ratio: {pruned_count / total_params:.2%}")

# -----------------------------
# Save pruned model
# -----------------------------
print("\nSaving pruned model...")
os.makedirs(OUTPUT_PATH, exist_ok=True)

model.save_pretrained(
    OUTPUT_PATH,
    safe_serialization=True,
    max_shard_size="2GB",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print("\n====================================")
print("✅ Magnitude-based pruning complete")
print(f"📁 Output path: {OUTPUT_PATH}")
print(f"🧹 Prune ratio: {PRUNE_RATIO:.0%}")
print("====================================")