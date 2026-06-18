import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from collections import defaultdict
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "./SinLlama_merged"
VAL_TEXT_FILE = "./eval.txt"
OUTPUT_DIR = "./wa_importance_analysis_cpu_fp32"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")  # CPU for full precision
MAX_LENGTH = 128
NUM_BINS = 300
DETECTION_SAMPLES = 20  # Fewer samples for range detection since it's slow

# Optional: Reduce dataset size for faster testing
USE_FULL_DATASET = True  # Set to False to use only first 100 samples
MAX_SAMPLES = 100 if not USE_FULL_DATASET else None

print("="*60)
print("FULL PRECISION CPU MODE")
print("="*60)
print("⚠️  This will be SLOW but numerically stable")
print("⚠️  Expect ~1-2 min per sample on CPU")
print("="*60)

# ============================================================
# LOAD TOKENIZER + MODEL
# ============================================================
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model in FULL PRECISION on CPU...")
print("(This may take several minutes...)")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,  # Full precision
    device_map="cpu",           # CPU only
    low_cpu_mem_usage=True,     # Load efficiently
)
model.eval()
print("✓ Model loaded in FP32 on CPU")

# ============================================================
# LOAD VALIDATION TEXT
# ============================================================
print("\nLoading validation text...")
with open(VAL_TEXT_FILE, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

if MAX_SAMPLES is not None:
    texts = texts[:MAX_SAMPLES]
    print(f"✓ Using {len(texts)} samples (reduced for faster testing)")
else:
    print(f"✓ Loaded {len(texts)} sentences")

# ============================================================
# PHASE 1: DETECT RANGE
# ============================================================
print("\n" + "="*60)
print("PHASE 1: Detecting importance value range...")
print("="*60)

range_detection_values = []

def make_range_detection_hook(name: str, module):
    def hook(mod, inp, out):
        x = inp[0]
        if x is None:
            return

        # Get activations (already FP32)
        if x.dim() == 3:
            col_scale = x.abs().mean(dim=(0, 1))
        elif x.dim() == 2:
            col_scale = x.abs().mean(dim=0)
        else:
            return

        # Get weight (already FP32, correct shape)
        w = module.weight.data
        
        # Calculate column-wise importance
        absW = w.abs()  # [out_features, in_features]
        col_sum = absW.sum(dim=0)  # Sum over output dim -> [in_features]
        
        col_importance = col_sum * col_scale
        
        # Check for issues
        if not torch.isfinite(col_importance).all():
            print(f"Warning: {name} has non-finite importance values")
            return
        
        # Sample for range detection
        if col_importance.numel() > 5000:
            idx = torch.randperm(col_importance.numel())[:5000]
            col_importance = col_importance[idx]
        
        range_detection_values.append(col_importance.detach())
    
    return hook

# Register temporary hooks
temp_hooks = []
for name, module in model.named_modules():
    if ("self_attn" in name or "mlp" in name) and isinstance(module, torch.nn.Linear):
        temp_hooks.append(
            module.register_forward_hook(make_range_detection_hook(name, module))
        )

print(f"✓ Registered {len(temp_hooks)} hooks")

# Run detection passes
print(f"\nRunning range detection on {DETECTION_SAMPLES} samples...")
print("(This will take a few minutes on CPU...)")

with torch.no_grad():
    for i, text in enumerate(tqdm(texts[:DETECTION_SAMPLES], desc="Range detection")):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        # No need to move to device - already on CPU
        _ = model(**inputs)

# Remove temp hooks
for h in temp_hooks:
    h.remove()

# Analyze range
if not range_detection_values:
    print("❌ ERROR: No importance values collected!")
    exit(1)

all_detection_vals = torch.cat(range_detection_values)
all_detection_vals = all_detection_vals[all_detection_vals > 0]

if all_detection_vals.numel() == 0:
    print("❌ ERROR: No positive values detected!")
    exit(1)

# Check for invalid values
if not torch.isfinite(all_detection_vals).all():
    print("⚠️  Warning: Some non-finite values detected, filtering...")
    all_detection_vals = all_detection_vals[torch.isfinite(all_detection_vals)]

if all_detection_vals.numel() == 0:
    print("❌ ERROR: No valid values after filtering!")
    exit(1)

vals_cpu = all_detection_vals.numpy()
log_vals = np.log10(vals_cpu)

LOG_MIN = float(np.floor(log_vals.min()) - 1)
LOG_MAX = float(np.ceil(log_vals.max()) + 1)

print(f"\n✓ Detected value range:")
print(f"  Raw values: [{vals_cpu.min():.6e}, {vals_cpu.max():.6e}]")
print(f"  Log10 range: [{log_vals.min():.2f}, {log_vals.max():.2f}]")
print(f"  Using: LOG_MIN={LOG_MIN:.0f}, LOG_MAX={LOG_MAX:.0f}")

# Create bins
LOG_BINS = np.linspace(LOG_MIN, LOG_MAX, NUM_BINS)

# ============================================================
# PHASE 2: FULL ANALYSIS
# ============================================================
print("\n" + "="*60)
print("PHASE 2: Full importance analysis...")
print("="*60)

# Clear detection data
del range_detection_values, all_detection_vals, vals_cpu, log_vals

# Histogram accumulators (keep on CPU as numpy arrays for efficiency)
global_hist = np.zeros(NUM_BINS - 1)
attn_hist = np.zeros(NUM_BINS - 1)
mlp_hist = np.zeros(NUM_BINS - 1)
layer_hists = defaultdict(lambda: np.zeros(NUM_BINS - 1))

def add_hist(hist_accum: np.ndarray, values: torch.Tensor):
    """Add values to histogram (numpy version for CPU)"""
    values_np = values.numpy()
    values_np = values_np[values_np > 0]
    if values_np.size == 0:
        return

    values_np = values_np[np.isfinite(values_np)]
    if values_np.size == 0:
        return

    logv = np.log10(values_np)
    logv = logv[(logv >= LOG_MIN) & (logv <= LOG_MAX)]
    if logv.size == 0:
        return

    h, _ = np.histogram(logv, bins=LOG_BINS)
    hist_accum += h

def make_linear_input_hook(name: str, module):
    def hook(mod, inp, out):
        x = inp[0]
        if x is None:
            return

        # Get activations
        if x.dim() == 3:
            col_scale = x.abs().mean(dim=(0, 1))
        elif x.dim() == 2:
            col_scale = x.abs().mean(dim=0)
        else:
            return

        # Get weight
        w = module.weight.data
        
        # Calculate importance
        absW = w.abs()
        col_sum = absW.sum(dim=0)
        col_importance = col_sum * col_scale

        if not torch.isfinite(col_importance).all():
            return

        # Subsample if too large (for memory/speed)
        if col_importance.numel() > 200_000:
            idx = torch.randperm(col_importance.numel())[:200_000]
            col_importance = col_importance[idx]

        add_hist(global_hist, col_importance)

        if "self_attn" in name:
            add_hist(attn_hist, col_importance)
        elif "mlp" in name:
            add_hist(mlp_hist, col_importance)

        if "layers." in name:
            layer_id = name.split("layers.")[1].split(".")[0]
            add_hist(layer_hists[layer_id], col_importance)

    return hook

# Register hooks
hooks = []
for name, module in model.named_modules():
    if ("self_attn" in name or "mlp" in name) and isinstance(module, torch.nn.Linear):
        hooks.append(
            module.register_forward_hook(make_linear_input_hook(name, module))
        )

print(f"✓ Registered {len(hooks)} hooks")

# Forward passes
print(f"\nCollecting importance statistics on {len(texts)} samples...")
print("⏰ Estimated time: ~{:.1f} minutes".format(len(texts) * 1.5))
print("(You can reduce dataset size by setting USE_FULL_DATASET=False)")

with torch.no_grad():
    for text in tqdm(texts, desc="Forward passes"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        # Inputs already on CPU
        _ = model(**inputs)

# Remove hooks
for h in hooks:
    h.remove()

print("✓ Collection complete")

# ============================================================
# PLOTTING
# ============================================================
def plot_hist(hist, title, filename):
    centers = 10 ** ((LOG_BINS[:-1] + LOG_BINS[1:]) / 2)
    if np.count_nonzero(hist) == 0:
        print(f"[WARN] Empty histogram for {title}")
        return

    plt.figure(figsize=(7, 5))
    plt.plot(centers, hist, linewidth=1.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Column Importance", fontsize=12)
    plt.ylabel("Count (log)", fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  ✓ {os.path.basename(filename)}")

def plot_cdf(hist, title, filename):
    if np.count_nonzero(hist) == 0:
        print(f"[WARN] Empty CDF for {title}")
        return

    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    centers = 10 ** ((LOG_BINS[:-1] + LOG_BINS[1:]) / 2)

    plt.figure(figsize=(7, 5))
    plt.plot(centers, cdf, linewidth=1.5)
    plt.xscale("log")
    plt.xlabel("Column Importance", fontsize=12)
    plt.ylabel("CDF", fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  ✓ {os.path.basename(filename)}")

print("\nGenerating plots...")
plot_hist(global_hist, "Global Column Importance (FP32)", f"{OUTPUT_DIR}/global_col_importance_hist.png")
plot_cdf(global_hist, "Global Column Importance CDF (FP32)", f"{OUTPUT_DIR}/global_col_importance_cdf.png")
plot_hist(attn_hist, "Attention Column Importance (FP32)", f"{OUTPUT_DIR}/attn_col_importance_hist.png")
plot_hist(mlp_hist, "MLP Column Importance (FP32)", f"{OUTPUT_DIR}/mlp_col_importance_hist.png")

layer_ids = sorted(layer_hists.keys(), key=int)
if layer_ids:
    sample_layers = [layer_ids[0], layer_ids[len(layer_ids) // 2], layer_ids[-1]]
    for lid in sample_layers:
        plot_cdf(layer_hists[lid], f"Layer {lid} Column Importance CDF", f"{OUTPUT_DIR}/layer_{lid}_col_importance_cdf.png")

print("\n" + "="*60)
print("✅ COMPLETE")
print(f"📁 {OUTPUT_DIR}")
print(f"📊 Histogram range: 10^{LOG_MIN:.0f} to 10^{LOG_MAX:.0f}")
print("="*60)