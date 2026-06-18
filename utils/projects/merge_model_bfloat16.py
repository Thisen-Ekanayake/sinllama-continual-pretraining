import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import time

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL_PATH = "llama-3-8b"
ADAPTER_PATH    = "SinLlama_v01"
OUTPUT_PATH     = "./SinLlama_merged_bf16"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# -----------------------------
# FORCE CPU
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs
torch.set_default_dtype(torch.float32)
device = "cpu"

def log_step(name, start_time):
    elapsed = time.time() - start_time
    print(f"[DONE] {name} — {elapsed:.2f}s")
    return elapsed

total_start = time.time()

# -----------------------------
# Load tokenizer
# -----------------------------
print("Loading tokenizer...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
    use_fast=False,
)
log_step("Tokenizer load", t0)

# -----------------------------
# Load base model (float32)
# -----------------------------
print("Loading base model on CPU...")
t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float32,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
)
log_step("Base model load", t0)

# -----------------------------
# Resize token embeddings
# -----------------------------
print("Resizing token embeddings...")
t0 = time.time()
base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
log_step("Embedding resize", t0)

# -----------------------------
# Load LoRA adapter
# -----------------------------
print("Loading LoRA adapter...")
t0 = time.time()
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    device_map={"": "cpu"},
    is_trainable=False,
)
log_step("LoRA adapter load", t0)

# -----------------------------
# Merge LoRA weights
# -----------------------------
print("Merging LoRA into base model...")
t0 = time.time()
model = model.merge_and_unload()
log_step("LoRA merge", t0)

# -----------------------------
# Convert to bfloat16
# -----------------------------
print("Converting model to bfloat16...")
t0 = time.time()
model = model.to(torch.bfloat16)
log_step("Conversion to bfloat16", t0)

# Optional: verify dtype
print("Verifying parameter dtype...")
for name, param in model.named_parameters():
    print(f"{name} dtype:", param.dtype)
    break

# -----------------------------
# Save merged model
# -----------------------------
print("Saving merged model (bfloat16)...")
t0 = time.time()
model.save_pretrained(
    OUTPUT_PATH,
    safe_serialization=True,
    max_shard_size="2GB",
)
tokenizer.save_pretrained(OUTPUT_PATH)
log_step("Model + tokenizer save", t0)

# -----------------------------
# Final report
# -----------------------------
total_elapsed = time.time() - total_start

print("\n====================================")
print("✅ SinLlama merge completed successfully (bfloat16)")
print(f"📁 Output path: {OUTPUT_PATH}")
print(f"⏱️ Total elapsed time: {total_elapsed / 60:.2f} minutes")
print("====================================")