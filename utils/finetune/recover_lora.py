import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# PATHS
# ==========================================

BASE_MODEL = "/workspace/sinllama_cpt_out/stage2/merged_bf16"
CHECKPOINT_PATH = "/workspace/classification_out/writing_stage2/checkpoint-1875"
OUT_DIR = "/workspace/classification_out/writing_stage2"

adapter_path = os.path.join(OUT_DIR, "adapters")
merged_path  = os.path.join(OUT_DIR, "merged_bf16")

os.makedirs(adapter_path, exist_ok=True)
os.makedirs(merged_path, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

print("Loading LoRA checkpoint...")
model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)

# ==========================================
# SAVE ADAPTERS
# ==========================================

print("Saving adapters...")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

# ==========================================
# MERGE
# ==========================================

print("Merging LoRA into base model...")
merged = model.merge_and_unload()

print("Saving merged model...")
merged.save_pretrained(merged_path, safe_serialization=True)
tokenizer.save_pretrained(merged_path)

print("Done.")