import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Must be FIRST, before torch import

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

# -----------------------------
# Verify CPU-only mode
# -----------------------------
assert not torch.cuda.is_available(), "CUDA should not be available"
print("✓ Running in CPU-only mode")

# -----------------------------
# Set default tensor settings
# -----------------------------
torch.set_default_dtype(torch.float32)

# -----------------------------
# Force CPU device
# -----------------------------
device = torch.device("cpu")
print(f"✓ Using device: {device}")

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).parent.parent
BASE_MODEL_PATH = str(ROOT / "llama-3-8b")
ADAPTER_PATH = str(ROOT / "SinLlama_v01")

print(f"✓ Base model path: {BASE_MODEL_PATH}")
print(f"✓ Adapter path: {ADAPTER_PATH}")

# -----------------------------
# Load the EXTENDED tokenizer (critical!)
# -----------------------------
print("\nLoading extended Sinhala tokenizer...")
# According to model card, use the extended tokenizer from separate repo
tokenizer = AutoTokenizer.from_pretrained("polyglots/Extended-Sinhala-LLaMA")
print(f"✓ Extended tokenizer loaded (vocab size: {len(tokenizer)})")

# -----------------------------
# Load base model on CPU
# -----------------------------
print("\nLoading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map={"": "cpu"},
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
print(f"✓ Base model loaded (original vocab size: {base_model.config.vocab_size})")

# -----------------------------
# CRITICAL: Resize embeddings to 139336 (as per model card)
# -----------------------------
print("\nResizing model embeddings to match extended tokenizer...")
target_vocab_size = 139336  # From model card
original_vocab_size = base_model.config.vocab_size

print(f"  Resizing from {original_vocab_size} to {target_vocab_size} tokens...")
base_model.resize_token_embeddings(target_vocab_size)
print(f"✓ Embeddings resized to {target_vocab_size}")

# Verify tokenizer matches
assert len(tokenizer) == target_vocab_size, f"Tokenizer size mismatch: {len(tokenizer)} != {target_vocab_size}"
print(f"✓ Tokenizer and model vocab sizes match: {target_vocab_size}")

# -----------------------------
# Load LoRA adapter
# -----------------------------
print("\nLoading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    device_map={"": "cpu"},
    torch_dtype=torch.float32
)
print("✓ Adapter loaded")

# -----------------------------
# CPU inference settings
# -----------------------------
model.eval()
model.config.use_cache = False

# Ensure everything is on CPU
model.to(device)
print("✓ Model moved to CPU")

# -----------------------------
# Example inference
# -----------------------------
print("\n" + "="*50)
print("Starting inference...")
print("="*50)

prompt = "ආයුබෝවන්, ඔබට කොහොමද?"
print(f"\nPrompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt")
# Ensure inputs are on CPU
inputs = {k: v.to(device) for k, v in inputs.items()}

print("\nGenerating response...")
print("(Note: This is a base model, not instruct-tuned)")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,      # Changed to True for better generation
        temperature=0.7,     # Add temperature for variety
        top_p=0.9,          # Add nucleus sampling
        use_cache=False,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n" + "="*50)
print("RESPONSE:")
print("="*50)
print(response)
print("="*50)
print("\nNote: For better results, fine-tune on your specific task")
print("(sentiment analysis, news categorization, etc.)")