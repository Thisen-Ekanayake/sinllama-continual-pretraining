import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from pathlib import Path

# -----------------------------
# ENABLE GPU
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

assert torch.cuda.is_available(), "CUDA is not available"
device = torch.device("cuda")
print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).parent.parent
MERGED_MODEL_PATH = str(ROOT / "SinLlama_merged_bf16")

print(f"✓ Model path: {MERGED_MODEL_PATH}")

# -----------------------------
# 4-bit Quantization Config
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # Best for LLMs
    bnb_4bit_use_double_quant=True,     # Memory efficient
    bnb_4bit_compute_dtype=torch.bfloat16  # Stable compute
)

# -----------------------------
# Load tokenizer
# -----------------------------
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

# -----------------------------
# Load model in 4-bit on GPU
# -----------------------------
print("\nLoading model in 4-bit (NF4)...")

model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

print("✓ Model loaded in 4-bit")

# -----------------------------
# Inference setup
# -----------------------------
model.eval()
model.config.use_cache = True

# -----------------------------
# Example inference
# -----------------------------
print("\n" + "=" * 60)
print("Starting GPU 4-bit inference")
print("=" * 60)

prompt = "ආයුබෝවන්, ඔබට කොහොමද?"
print(f"\nPrompt:\n{prompt}")

inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\nGenerating response...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "=" * 60)
print("RESPONSE")
print("=" * 60)
print(response)
print("=" * 60)

print("\n✓ 4-bit GPU inference completed")