import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU ONLY

import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
ROOT = Path(__file__).parent.parent
MODEL_PATH = str(ROOT / "SinLlama_merged")
EVAL_TEXT_PATH = str(ROOT / "eval.txt")

BATCH_SIZE = 1          # keep 1 for CPU safety
MAX_LENGTH = 1024       # must match model context
STRIDE = 512            # sliding window stride

# -----------------------------
# Sanity checks
# -----------------------------
assert not torch.cuda.is_available(), "CUDA must be disabled"

torch.set_default_dtype(torch.float32)
device = torch.device("cpu")

print("✓ CPU-only mode confirmed")

# -----------------------------
# Load tokenizer & model
# -----------------------------
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model (this may take time)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": "cpu"},
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()
model.config.use_cache = False
model.to(device)

print("✓ Model loaded")

# -----------------------------
# Load evaluation text
# -----------------------------
with open(EVAL_TEXT_PATH, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

assert len(texts) > 0, "Evaluation file is empty"

print(f"✓ Loaded {len(texts)} evaluation samples")

# -----------------------------
# Perplexity computation
# -----------------------------
total_loss = 0.0
total_tokens = 0

print("\nComputing perplexity...")
for text in tqdm(texts):
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]

    seq_len = input_ids.size(0)

    for i in range(0, seq_len - 1, STRIDE):
        begin = max(i + STRIDE - MAX_LENGTH, 0)
        end = min(i + STRIDE, seq_len)

        input_chunk = input_ids[begin:end]
        target_chunk = input_chunk.clone()

        target_chunk[:-1] = -100  # causal masking

        with torch.no_grad():
            outputs = model(
                input_chunk.unsqueeze(0).to(device),
                labels=target_chunk.unsqueeze(0).to(device),
            )

        loss = outputs.loss
        num_tokens = (target_chunk != -100).sum().item()

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

# -----------------------------
# Final perplexity
# -----------------------------
avg_loss = total_loss / total_tokens
perplexity = math.exp(avg_loss)

print("\n" + "=" * 60)
print("📊 PERPLEXITY RESULTS")
print("=" * 60)
print(f"Tokens evaluated : {total_tokens}")
print(f"Average loss     : {avg_loss:.4f}")
print(f"Perplexity       : {perplexity:.4f}")
print("=" * 60)