import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "./SinLlama_merged"
VAL_TEXT_FILE = "./eval.txt"
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
# Perplexity computation
# -----------------------------
total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for text in tqdm(texts, desc="Evaluating PPL"):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        outputs = model(**enc, labels=enc["input_ids"])
        loss = outputs.loss

        n_tokens = enc["input_ids"].numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

ppl = math.exp(total_loss / total_tokens)
print(f"\nBaseline Perplexity: {ppl:.3f}")