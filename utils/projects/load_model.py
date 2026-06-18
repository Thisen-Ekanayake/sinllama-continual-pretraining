from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
from pathlib import Path

# -----------------------------
# Clean CUDA state (important)
# -----------------------------
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# -----------------------------
# Paths (absolute, robust)
# -----------------------------
ROOT = Path(__file__).parent.parent
BASE_MODEL_PATH = str(ROOT / "llama-3-8b")
ADAPTER_PATH = str(ROOT / "SinLlama_v01")

# -----------------------------
# Tokenizer (from adapter)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

# -----------------------------
# 4-bit quantization config
# -----------------------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  # FP16 > BF16 for 4060
)

# -----------------------------
# Load base model (ONE device_map)
# -----------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=quant_config,
    device_map="cuda"
    # max_memory={0: "7GB"},  # leave VRAM headroom
)

# -----------------------------
# Load LoRA adapter (NO device_map)
# -----------------------------
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

# -----------------------------
# Inference safety settings
# -----------------------------
model.eval()
model.config.use_cache = False  # critical for avoiding OOM

# -----------------------------
# Example inference
# -----------------------------
prompt = "ආයුබෝවන්, ඔබට කොහොමද?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=False
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))