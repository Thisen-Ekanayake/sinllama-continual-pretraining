import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "./SinLlama_merged"
OUTPUT_DIR = "./weight_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.set_default_dtype(torch.float32)
device = torch.device("cpu")

# Histogram config (log-scale)
NUM_BINS = 400
MIN_W = 1e-8
MAX_W = 1e-1
BINS = np.logspace(np.log10(MIN_W), np.log10(MAX_W), NUM_BINS)

# -----------------------------
# Load model
# -----------------------------
print("Loading merged model (CPU)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": "cpu"},
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()
print("✓ Model loaded")

# -----------------------------
# Histogram accumulators
# -----------------------------
global_hist = np.zeros(NUM_BINS - 1)
attn_hist = np.zeros(NUM_BINS - 1)
mlp_hist = np.zeros(NUM_BINS - 1)
layer_hists = defaultdict(lambda: np.zeros(NUM_BINS - 1))

# -----------------------------
# Iterate parameters safely
# -----------------------------
print("Collecting weight magnitude histograms...")

for name, param in tqdm(model.named_parameters(), desc="Scanning parameters"):
    w = param.detach().cpu().numpy()
    abs_w = np.abs(w)

    # Clip to histogram range
    abs_w = abs_w[(abs_w >= MIN_W) & (abs_w <= MAX_W)]
    if abs_w.size == 0:
        continue

    hist, _ = np.histogram(abs_w, bins=BINS)
    global_hist += hist

    if "self_attn" in name:
        attn_hist += hist
    elif "mlp" in name:
        mlp_hist += hist

    if "layers." in name:
        layer_id = name.split("layers.")[1].split(".")[0]
        layer_hists[layer_id] += hist

print("✓ Histogram collection complete")

# -----------------------------
# Plot helpers
# -----------------------------
def plot_hist_from_bins(hist, bins, title, filename):
    centers = (bins[:-1] + bins[1:]) / 2
    plt.figure(figsize=(7, 5))
    plt.plot(centers, hist)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("|weight|")
    plt.ylabel("count (log)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_cdf_from_hist(hist, bins, title, filename):
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(7, 5))
    plt.plot(centers, cdf)
    plt.xscale("log")
    plt.xlabel("|weight|")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# -----------------------------
# Global plots
# -----------------------------
plot_hist_from_bins(
    global_hist,
    BINS,
    "Global Weight Magnitude Distribution",
    f"{OUTPUT_DIR}/global_weight_hist.png",
)

plot_cdf_from_hist(
    global_hist,
    BINS,
    "Global Weight Magnitude CDF",
    f"{OUTPUT_DIR}/global_weight_cdf.png",
)

# -----------------------------
# Attention vs MLP
# -----------------------------
plot_hist_from_bins(
    attn_hist,
    BINS,
    "Attention Weight Magnitude Distribution",
    f"{OUTPUT_DIR}/attention_weight_hist.png",
)

plot_hist_from_bins(
    mlp_hist,
    BINS,
    "MLP Weight Magnitude Distribution",
    f"{OUTPUT_DIR}/mlp_weight_hist.png",
)

# -----------------------------
# Layer-wise (sampled)
# -----------------------------
layer_ids = sorted(layer_hists.keys(), key=int)
sample_layers = [
    layer_ids[0],
    layer_ids[len(layer_ids) // 2],
    layer_ids[-1],
]

for lid in sample_layers:
    plot_cdf_from_hist(
        layer_hists[lid],
        BINS,
        f"Layer {lid} Weight Magnitude CDF",
        f"{OUTPUT_DIR}/layer_{lid}_cdf.png",
    )

print("\n====================================")
print("✅ Memory-safe weight analysis complete")
print(f"📁 Plots saved in: {OUTPUT_DIR}")
print("Peak RAM usage stays low")
print("====================================")