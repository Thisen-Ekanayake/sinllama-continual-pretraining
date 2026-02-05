import torch
import time
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

# ============================================================
# 🔧 USER CONFIG — EDIT THESE TWO PATHS ONLY
# ============================================================

BASE_MODEL_PATH = "llama-3-8b"
SINLLAMA_LORA_PATH = "SinLlama_v01"

# ============================================================
# CPU CONFIG
# ============================================================

# Force CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""

DEVICE = "cpu"
DTYPE = torch.float32
NEW_LORA_RANK = 16
OUTPUT_DIR = "sinllama_compact_lora_r16_cpu"

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj", "gate_proj",
]

# ============================================================
# UTILITIES
# ============================================================

def log_step(name):
    print(f"\n{'=' * 60}")
    print(f"▶ {name}")
    print(f"{'=' * 60}")
    return time.perf_counter()

def end_step(start_time):
    elapsed = time.perf_counter() - start_time
    print(f"✔ Done in {elapsed/60:.2f} minutes")

def low_rank_factorize(delta_w, rank):
    """
    Factorize delta_w using SVD on CPU.
    
    Args:
        delta_w: Weight delta tensor
        rank: Target rank for factorization
    
    Returns:
        A, B: Factorized matrices such that delta_w ≈ A @ B
    """
    U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)

    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]

    sqrt_S = torch.sqrt(S)
    A = U * sqrt_S.unsqueeze(0)
    B = sqrt_S.unsqueeze(1) * Vh
    
    return A, B

class ProgressTracker:
    """Context manager for tracking progress with tqdm."""
    
    def __init__(self, desc, total=None, unit="it"):
        self.desc = desc
        self.total = total
        self.unit = unit
        self.pbar = None
        self.start_time = None
    
    def __enter__(self):
        print(f"\n{'=' * 60}")
        print(f"▶ {self.desc}")
        print(f"{'=' * 60}")
        self.start_time = time.perf_counter()
        if self.total is not None:
            self.pbar = tqdm(total=self.total, desc=self.desc, unit=self.unit)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()
        elapsed = time.perf_counter() - self.start_time
        print(f"✔ Done in {elapsed/60:.2f} minutes")
        return False
    
    def update(self, n=1):
        if self.pbar:
            self.pbar.update(n)

# ============================================================
# STEP 1 — Tokenizer (Extended for Sinhala)
# ============================================================

print("\n" + "=" * 60)
print("🚀 SINLLAMA LORA COMPRESSION PIPELINE")
print("=" * 60)
print("Steps: 7 total")
print("  1. Load tokenizer")
print("  2. Load & resize base model")
print("  3. Attach & merge LoRA")
print("  4. Reload models for comparison")
print("  5. Create compact LoRA structure")
print("  6. Refactorize weights (SVD compression)")
print("  7. Save compact LoRA")
print("=" * 60)

pipeline_start = time.perf_counter()

t = log_step("[1/7] Loading extended Sinhala tokenizer")
# SinLlama uses an extended tokenizer with 139,336 tokens
tokenizer = AutoTokenizer.from_pretrained("polyglots/Extended-Sinhala-LLaMA")
EXTENDED_VOCAB_SIZE = 139336
end_step(t)

# ============================================================
# STEP 2 — Base + LoRA load
# ============================================================

t = log_step("[2/7] Loading base model (CPU)")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=DTYPE,
    device_map={"": "cpu"},  # Force CPU device map
    low_cpu_mem_usage=True,
)
base_model.eval()
end_step(t)

t = log_step("[2/7] Resizing embeddings for extended vocabulary")
# CRITICAL: Resize embeddings to match SinLlama's extended vocab (139,336 tokens)
base_model.resize_token_embeddings(EXTENDED_VOCAB_SIZE)
print(f"✓ Embeddings resized to {EXTENDED_VOCAB_SIZE} tokens")
end_step(t)

t = log_step("[2/7] Attaching SinLlama LoRA")
model_with_lora = PeftModel.from_pretrained(
    base_model,
    SINLLAMA_LORA_PATH,
    device_map={"": "cpu"},  # Force CPU device map
)
model_with_lora.eval()
end_step(t)

# ============================================================
# STEP 3 — Merge LoRA
# ============================================================

t = log_step("[3/7] Merging LoRA into base")
merged_model = model_with_lora.merge_and_unload()
merged_model.eval()
end_step(t)

MERGED_DIR = "sinllama_merged_fp32_cpu"
os.makedirs(MERGED_DIR, exist_ok=True)

t = log_step("[3/7] Saving merged model")
print("Writing model shards to disk...")
merged_model.save_pretrained(MERGED_DIR)
print("Saving tokenizer...")
tokenizer.save_pretrained(MERGED_DIR)
end_step(t)

# Clean up memory
del model_with_lora
del base_model
import gc
gc.collect()

# ============================================================
# STEP 4 — Reload base & merged
# ============================================================

t = log_step("[4/7] Reloading base reference model")
base_ref = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=DTYPE,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
)
# Resize to match extended vocab
base_ref.resize_token_embeddings(EXTENDED_VOCAB_SIZE)
base_ref.eval()
end_step(t)

t = log_step("[4/7] Reloading merged model")
merged_ref = AutoModelForCausalLM.from_pretrained(
    MERGED_DIR,
    torch_dtype=DTYPE,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
)
merged_ref.eval()
end_step(t)

# ============================================================
# STEP 5 — Create compact LoRA shell
# ============================================================

t = log_step("[5/7] Creating compact LoRA structure")
lora_config = LoraConfig(
    r=NEW_LORA_RANK,
    lora_alpha=NEW_LORA_RANK,
    target_modules=TARGET_MODULES,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
compact_model = get_peft_model(base_ref, lora_config)
compact_model.eval()
end_step(t)

# ============================================================
# STEP 6 — Refactorize ΔW with progress bars
# ============================================================

print(f"\n{'=' * 60}")
print(f"▶ [6/7] Refactorizing weight deltas (SVD compression)")
print(f"{'=' * 60}")
step_start = time.perf_counter()

base_params = dict(base_ref.named_parameters())
merged_params = dict(merged_ref.named_parameters())

# Only get linear layer weights (2D tensors)
linear_weights = [
    name for name, p in merged_params.items()
    if name in base_params and p.ndim == 2
]

# Build a mapping from base parameter names to LoRA modules
lora_modules = {}
for name, module in compact_model.named_modules():
    # Check if this is a LoRA-enabled module
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
        # Extract the base parameter name this LoRA applies to
        # e.g., "base_model.model.model.layers.0.self_attn.q_proj" -> we want the full path
        lora_modules[name] = module

print(f"Found {len(lora_modules)} LoRA modules")
print(f"Processing {len(linear_weights)} linear weight matrices\n")

with torch.no_grad():
    compressed_count = 0
    
    # Create progress bar with ETA
    pbar = tqdm(
        linear_weights,
        desc="Compressing layers",
        unit="layer",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )
    
    for name in pbar:
        merged_param = merged_params[name]
        base_param = base_params[name]

        delta = merged_param - base_param
        
        # Skip if no actual change
        if torch.allclose(delta, torch.zeros_like(delta), atol=1e-6):
            continue

        # Find matching LoRA module
        # Parameter names look like: "model.layers.0.self_attn.q_proj.weight"
        # Module names in PEFT look like: "base_model.model.model.layers.0.self_attn.q_proj"
        param_base_name = name.replace('.weight', '')
        
        matched = False
        for lora_name, lora_module in lora_modules.items():
            if lora_name.endswith(param_base_name):
                # Update progress bar with current layer
                pbar.set_postfix_str(f"SVD: {param_base_name.split('.')[-1]}")
                
                # Perform SVD factorization
                A, B = low_rank_factorize(delta, NEW_LORA_RANK)
                
                # LoRA convention: W = B @ A, where A is (rank, in_features) and B is (out_features, rank)
                # Our SVD gives us delta_W ≈ A @ B where A is (out, rank) and B is (rank, in)
                # So we assign: lora_A.weight = B.T and lora_B.weight = A
                lora_module.lora_A['default'].weight.copy_(B.T)
                lora_module.lora_B['default'].weight.copy_(A)
                
                compressed_count += 1
                matched = True
                break
        
        if not matched and not torch.allclose(delta, torch.zeros_like(delta), atol=1e-6):
            print(f"Warning: Could not find LoRA module for {name}")

print(f"\n✓ Compressed {compressed_count} layers")
elapsed = time.perf_counter() - step_start
print(f"✔ Done in {elapsed/60:.2f} minutes")

# ============================================================
# STEP 7 — Save compact LoRA
# ============================================================

t = log_step("[7/7] Saving compact LoRA adapter")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Writing LoRA adapter weights...")
compact_model.save_pretrained(OUTPUT_DIR)
print("Saving tokenizer configuration...")
tokenizer.save_pretrained(OUTPUT_DIR)
end_step(t)

# ============================================================
# PIPELINE COMPLETE
# ============================================================

total_elapsed = time.perf_counter() - pipeline_start
print("\n" + "=" * 60)
print("🎉 ALL STEPS COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"📦 Compact LoRA saved to: {OUTPUT_DIR}")
print(f"⏱️  Total pipeline time: {total_elapsed/60:.2f} minutes")
print("=" * 60)