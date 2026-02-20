import os
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/sinllama_cpt_out/stage2/merged_bf16")
DATA_PATH  = os.environ.get("DATA_PATH",  "/workspace/data/eval.txt")

SEQ_LEN    = int(os.environ.get("SEQ_LEN", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nModel: {MODEL_PATH}")
print(f"Dataset: {DATA_PATH}")
print(f"Seq len: {SEQ_LEN}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Device: {DEVICE}\n")

# ============================================================
# Load tokenizer + model
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)

model.eval()

# ============================================================
# Load dataset
# ============================================================

dataset = load_dataset(
    "text",
    data_files={"eval": DATA_PATH}
)["eval"]

print(f"Loaded {len(dataset)} lines")

# ============================================================
# Tokenize
# ============================================================

def tokenize_fn(example):
    text = example["text"].strip()
    if not text:
        return {"input_ids": []}
    return {
        "input_ids": tokenizer(
            text + tokenizer.eos_token,
            add_special_tokens=False,
        ).input_ids
    }

dataset = dataset.map(tokenize_fn, remove_columns=["text"], num_proc=4)

# Flatten
all_ids = []
for item in dataset:
    all_ids.extend(item["input_ids"])

total_tokens = len(all_ids)
print(f"Total tokens: {total_tokens:,}")

# Truncate to multiple of SEQ_LEN
usable = (total_tokens // SEQ_LEN) * SEQ_LEN
all_ids = all_ids[:usable]

input_ids = torch.tensor(all_ids).view(-1, SEQ_LEN)

print(f"Total sequences: {input_ids.shape[0]:,}")

# ============================================================
# Compute perplexity
# ============================================================

total_loss = 0.0
total_tokens_eval = 0

with torch.no_grad():
    for i in range(0, input_ids.size(0), BATCH_SIZE):
        batch = input_ids[i:i+BATCH_SIZE].to(DEVICE)

        outputs = model(batch, labels=batch)
        loss = outputs.loss

        tokens_in_batch = batch.numel()

        total_loss += loss.item() * tokens_in_batch
        total_tokens_eval += tokens_in_batch

avg_loss = total_loss / total_tokens_eval
perplexity = math.exp(avg_loss)

print("\n" + "="*50)
print(f"Average loss: {avg_loss:.4f}")
print(f"Perplexity  : {perplexity:.4f}")
print("="*50)