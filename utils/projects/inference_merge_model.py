import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # MUST be first

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# -----------------------------
# Verify CPU-only mode
# -----------------------------
assert not torch.cuda.is_available(), "CUDA should not be available"
print("✓ Running in CPU-only mode")

# -----------------------------
# Torch defaults
# -----------------------------
torch.set_default_dtype(torch.float32)
device = torch.device("cpu")
print(f"✓ Using device: {device}")

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).parent.parent
MERGED_MODEL_PATH = str(ROOT / "SinLlama_merged")

print(f"✓ Merged model path: {MERGED_MODEL_PATH}")

# -----------------------------
# Load tokenizer (from merged model)
# -----------------------------
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

# -----------------------------
# Load merged model (CPU only)
# -----------------------------
print("\nLoading merged model...")
model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    device_map={"": "cpu"},
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
print("✓ Model loaded")

# -----------------------------
# CPU inference settings
# -----------------------------
model.eval()
model.config.use_cache = False
model.to(device)
print("✓ Model ready for inference")

# -----------------------------
# Example inference
# -----------------------------
print("\n" + "=" * 60)
print("Starting inference")
print("=" * 60)

prompt = "ආයුබෝවන්, ඔබට කොහොමද?"
print(f"\nPrompt:\n{prompt}")

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

print("\nGenerating response...")
print("(Note: This is a BASE model, not instruct-tuned)")

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

print("\nℹ️ Reminder:")
print("- This is a base LLM (not chat-tuned)")
print("- Expect free-form text, not assistant-style replies")
print("- Fine-tune for tasks like sentiment, QA, or dialogue")