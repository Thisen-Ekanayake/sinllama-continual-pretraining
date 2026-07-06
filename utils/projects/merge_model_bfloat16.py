import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors import safe_open
import os
import time

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL_PATH = "models/llama-3-8b"
ADAPTER_PATH    = "models/SinLlama_Backtrinx_Instruct_adapters"
OUTPUT_PATH     = "./SinLlama_Backtrinx_Instruct"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# -----------------------------
# FORCE CPU
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs
torch.set_default_dtype(torch.bfloat16)
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
# Load base model (bfloat16)
# -----------------------------
print("Loading base model on CPU (bfloat16)...")
t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
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
# Verify embedding size matches adapter
# -----------------------------
# The adapter carries full embed_tokens/lm_head (modules_to_save) sized to the
# extended Sinhala vocab. If the resized base doesn't match those saved rows,
# the merge loads mismatched embeddings — a silent quality failure, not an error.
# Read the adapter's row count from the safetensors header (metadata only, no
# tensor load) and assert everything lines up before merging.
print("Verifying embedding size against adapter...")
t0 = time.time()
adapter_weights = os.path.join(ADAPTER_PATH, "adapter_model.safetensors")
with safe_open(adapter_weights, framework="pt") as f:
    adapter_vocab = f.get_slice("base_model.model.model.embed_tokens.weight").get_shape()[0]

model_vocab = base_model.get_input_embeddings().weight.shape[0]
assert model_vocab == adapter_vocab == len(tokenizer), (
    f"Vocab size mismatch — resized model: {model_vocab}, "
    f"adapter embed_tokens: {adapter_vocab}, tokenizer: {len(tokenizer)}. "
    "The resized base must match the adapter's saved embeddings before merging."
)
print(f"[OK] Vocab size aligned: {model_vocab} rows")
log_step("Embedding size verification", t0)

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
# Merge LoRA weights (bfloat16)
# -----------------------------
print("Merging LoRA into base model (bfloat16)...")
t0 = time.time()
model = model.merge_and_unload()
log_step("LoRA merge", t0)

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