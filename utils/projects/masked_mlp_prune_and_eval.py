import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import numpy as np
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "./SinLlama_merged"
VAL_TEXT_FILE = "./eval.txt"
IMPORTANCE_FILE = "./column_importance.pt"

PRUNE_RATIO = 0.40     # 40% MLP columns
MAX_LENGTH = 128

torch.set_default_dtype(torch.float32)
device = torch.device("cpu")

# -----------------------------
# Load model, tokenizer, importance
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": "cpu"},
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()

importance = torch.load(IMPORTANCE_FILE)

# -----------------------------
# Apply masked MLP pruning
# -----------------------------
pruned_count = 0
skipped_count = 0

with torch.no_grad():
    for name, module in model.named_modules():
        if "mlp" in name and isinstance(module, torch.nn.Linear):
            # Skip if importance wasn't collected for this layer
            if name not in importance:
                skipped_count += 1
                continue
            
            imp = importance[name]
            threshold = np.quantile(imp.numpy(), PRUNE_RATIO)
            mask = (imp > threshold).float().to(module.weight.device)
            module.weight.data *= mask.unsqueeze(0)
            pruned_count += 1

print(f"✓ Masked MLP pruning applied to {pruned_count} layers")
print(f"⚠ Skipped {skipped_count} layers (no importance data)")

# -----------------------------
# Load validation text
# -----------------------------
with open(VAL_TEXT_FILE, "r", encoding="utf-8") as f:
    texts = [l.strip() for l in f if l.strip()]

# -----------------------------
# Evaluate perplexity
# -----------------------------
total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for text in tqdm(texts, desc="Evaluating pruned PPL"):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc, labels=enc["input_ids"])
        loss = out.loss
        n_tokens = enc["input_ids"].numel()

        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

ppl = math.exp(total_loss / total_tokens)
print(f"\nPruned Perplexity (MLP {PRUNE_RATIO*100:.0f}%): {ppl:.3f}")