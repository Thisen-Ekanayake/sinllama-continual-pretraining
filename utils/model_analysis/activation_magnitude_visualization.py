import os
# -----------------------------
# ENABLE GPU
# -----------------------------
os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # allow CUDA

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "./SinLlama_merged"
VAL_TEXT_FILE = "eval.txt"
OUTPUT_DIR = "./activation_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LENGTH = 128
BATCH_SIZE = 1  # increase to 2–4 if VRAM allows

# Histogram bins (log-scale)
NUM_BINS = 300
MIN_A = 1e-8
MAX_A = 1e1
BINS = np.logspace(np.log10(MIN_A), np.log10(MAX_A), NUM_BINS)

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

print(f"✓ Using device: {device}")

# -----------------------------
# Load model & tokenizer
# -----------------------------
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",            # 👈 GPU placement
    torch_dtype=torch.float16,    # 👈 critical for GPU
    low_cpu_mem_usage=True,
)

model.eval()
print("✓ Model ready")

# -----------------------------
# Load validation data
# -----------------------------
print("Loading validation text...")
with open(VAL_TEXT_FILE, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

print(f"✓ Loaded {len(texts)} validation sentences")

# -----------------------------
# Histogram storage (CPU-side)
# -----------------------------
global_hist = np.zeros(NUM_BINS - 1)
mlp_hist = np.zeros(NUM_BINS - 1)
attn_hist = np.zeros(NUM_BINS - 1)
layer_hists = defaultdict(lambda: np.zeros(NUM_BINS - 1))

# -----------------------------
# Hook factory
# -----------------------------
def activation_hook(name):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]

        # ⛔ DO NOT keep activations on GPU
        a = out.detach().float().cpu().numpy()
        abs_a = np.abs(a)
        abs_a = abs_a[(abs_a >= MIN_A) & (abs_a <= MAX_A)]

        if abs_a.size == 0:
            return

        hist, _ = np.histogram(abs_a, bins=BINS)
        global_hist[:] += hist

        if "mlp" in name:
            mlp_hist[:] += hist
        elif "self_attn" in name:
            attn_hist[:] += hist

        if "layers." in name:
            layer_id = name.split("layers.")[1].split(".")[0]
            layer_hists[layer_id] += hist

    return hook

# -----------------------------
# Register hooks
# -----------------------------
hooks = []
for name, module in model.named_modules():
    if "mlp" in name or "self_attn" in name:
        hooks.append(module.register_forward_hook(activation_hook(name)))

print(f"✓ Registered {len(hooks)} activation hooks")

# -----------------------------
# Forward passes
# -----------------------------
print("Running validation data through model...")
with torch.no_grad():
    for text in tqdm(texts, desc="Forward passes"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _ = model(**inputs)

# -----------------------------
# Remove hooks
# -----------------------------
for h in hooks:
    h.remove()

print("✓ Activation collection complete")

# -----------------------------
# Plot helpers
# -----------------------------
def plot_hist(hist, title, filename):
    centers = (BINS[:-1] + BINS[1:]) / 2
    plt.figure(figsize=(7, 5))
    plt.plot(centers, hist)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("|activation|")
    plt.ylabel("count (log)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_cdf(hist, title, filename):
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    centers = (BINS[:-1] + BINS[1:]) / 2
    plt.figure(figsize=(7, 5))
    plt.plot(centers, cdf)
    plt.xscale("log")
    plt.xlabel("|activation|")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# -----------------------------
# Global plots
# -----------------------------
plot_hist(
    global_hist,
    "Global Activation Magnitude Distribution",
    f"{OUTPUT_DIR}/global_activation_hist.png",
)

plot_cdf(
    global_hist,
    "Global Activation Magnitude CDF",
    f"{OUTPUT_DIR}/global_activation_cdf.png",
)

# -----------------------------
# Component plots
# -----------------------------
plot_hist(
    mlp_hist,
    "MLP Activation Magnitude Distribution",
    f"{OUTPUT_DIR}/mlp_activation_hist.png",
)

plot_hist(
    attn_hist,
    "Attention Activation Magnitude Distribution",
    f"{OUTPUT_DIR}/attention_activation_hist.png",
)

# -----------------------------
# Layer-wise CDFs
# -----------------------------
layer_ids = sorted(layer_hists.keys(), key=int)
sample_layers = [
    layer_ids[0],
    layer_ids[len(layer_ids) // 2],
    layer_ids[-1],
]

for lid in sample_layers:
    plot_cdf(
        layer_hists[lid],
        f"Layer {lid} Activation Magnitude CDF",
        f"{OUTPUT_DIR}/layer_{lid}_activation_cdf.png",
    )

print("\n====================================")
print("✅ Activation magnitude analysis done")
print(f"📁 Plots saved in: {OUTPUT_DIR}")
print("====================================")