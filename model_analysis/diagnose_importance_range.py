import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "./SinLlama_merged"
VAL_TEXT_FILE = "./eval.txt"
DEVICE = torch.device("cuda")
MAX_LENGTH = 128

# ============================================================
# 4-BIT QUANTIZATION CONFIG
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ============================================================
# LOAD TOKENIZER + MODEL
# ============================================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model in 4-bit (NF4) on GPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="cuda",
)
model.eval()
print("✓ Model loaded")

# ============================================================
# INSPECT A SINGLE LINEAR LAYER
# ============================================================
print("\n" + "="*60)
print("INSPECTING QUANTIZED WEIGHTS")
print("="*60)

# Find first linear layer
test_module = None
test_name = None
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        test_module = module
        test_name = name
        break

if test_module is None:
    print("❌ No Linear layers found!")
    exit(1)

print(f"\nTesting layer: {test_name}")
print(f"Weight type: {type(test_module.weight)}")
print(f"Weight dtype: {test_module.weight.dtype}")
print(f"Weight shape: {test_module.weight.shape}")

# Try to access the weight
try:
    w = test_module.weight
    print(f"\nDirect weight access:")
    print(f"  Type: {type(w)}")
    print(f"  Has .data: {hasattr(w, 'data')}")
    
    # Check if it's a quantized tensor
    if hasattr(w, 'quant_state'):
        print(f"  Quantized: YES")
        print(f"  Quant state: {w.quant_state}")
    else:
        print(f"  Quantized: NO")
    
    # Try to get actual values
    print(f"\nTrying to materialize weight...")
    
    # Method 1: Direct access
    try:
        w_data = w.data
        print(f"  Method 1 (w.data): shape={w_data.shape}, dtype={w_data.dtype}")
        if torch.isfinite(w_data).all():
            print(f"    ✓ All values finite")
            print(f"    Min: {w_data.min().item():.6e}")
            print(f"    Max: {w_data.max().item():.6e}")
        else:
            print(f"    ❌ Contains inf/nan!")
            print(f"    Finite: {torch.isfinite(w_data).sum().item()}/{w_data.numel()}")
    except Exception as e:
        print(f"  Method 1 failed: {e}")
    
    # Method 2: Through forward pass
    print(f"\nMethod 2: Testing with actual input...")
    test_input = torch.randn(1, 10, w.shape[1], device=DEVICE, dtype=torch.float16)
    with torch.no_grad():
        output = test_module(test_input)
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    if torch.isfinite(output).all():
        print(f"  ✓ Output is finite")
    else:
        print(f"  ❌ Output contains inf/nan!")
        
except Exception as e:
    print(f"❌ Error accessing weight: {e}")

# ============================================================
# TEST IMPORTANCE CALCULATION
# ============================================================
print("\n" + "="*60)
print("TESTING IMPORTANCE CALCULATION")
print("="*60)

# Load one validation sample
with open(VAL_TEXT_FILE, "r", encoding="utf-8") as f:
    test_text = f.readline().strip()

print(f"Test input: {test_text[:50]}...")

# Create a hook to inspect intermediate values
def diagnostic_hook(name, module):
    def hook(mod, inp, out):
        x = inp[0]
        print(f"\n[HOOK] {name}")
        print(f"  Input shape: {x.shape}")
        print(f"  Input dtype: {x.dtype}")
        print(f"  Input finite: {torch.isfinite(x).all()}")
        
        if x.dim() == 3:
            col_scale = x.abs().mean(dim=(0, 1))
        elif x.dim() == 2:
            col_scale = x.abs().mean(dim=0)
        else:
            return
        
        print(f"  col_scale shape: {col_scale.shape}")
        print(f"  col_scale dtype: {col_scale.dtype}")
        print(f"  col_scale finite: {torch.isfinite(col_scale).all()}")
        if torch.isfinite(col_scale).all():
            print(f"  col_scale min: {col_scale.min().item():.6e}")
            print(f"  col_scale max: {col_scale.max().item():.6e}")
        
        # Try to get weight
        try:
            w = module.weight.data
            print(f"  Weight shape: {w.shape}")
            print(f"  Weight dtype: {w.dtype}")
            print(f"  Weight finite: {torch.isfinite(w).all()}")
            
            if not torch.isfinite(w).all():
                print(f"  ❌ WEIGHT CONTAINS INF/NAN!")
                inf_count = torch.isinf(w).sum().item()
                nan_count = torch.isnan(w).sum().item()
                print(f"     Inf: {inf_count}, NaN: {nan_count}")
                return
            
            # Calculate importance
            absW = w.abs()
            print(f"  absW finite: {torch.isfinite(absW).all()}")
            
            col_sum = absW.sum(dim=0)
            print(f"  col_sum finite: {torch.isfinite(col_sum).all()}")
            if torch.isfinite(col_sum).all():
                print(f"  col_sum min: {col_sum.min().item():.6e}")
                print(f"  col_sum max: {col_sum.max().item():.6e}")
            
            col_importance = col_sum * col_scale
            print(f"  col_importance finite: {torch.isfinite(col_importance).all()}")
            
            if not torch.isfinite(col_importance).all():
                print(f"  ❌ IMPORTANCE CONTAINS INF/NAN!")
                inf_count = torch.isinf(col_importance).sum().item()
                nan_count = torch.isnan(col_importance).sum().item()
                print(f"     Inf: {inf_count}, NaN: {nan_count}")
            else:
                print(f"  ✓ Importance is finite")
                print(f"  Importance min: {col_importance.min().item():.6e}")
                print(f"  Importance max: {col_importance.max().item():.6e}")
                
        except Exception as e:
            print(f"  ❌ Error in importance calculation: {e}")
    
    return hook

# Register hook on first few layers
hooks = []
hook_count = 0
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and ("self_attn" in name or "mlp" in name):
        hooks.append(module.register_forward_hook(diagnostic_hook(name, module)))
        hook_count += 1
        if hook_count >= 3:  # Just test first 3 layers
            break

print(f"\n✓ Registered {len(hooks)} diagnostic hooks")

# Run forward pass
print("\nRunning forward pass...")
with torch.no_grad():
    inputs = tokenizer(
        test_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    _ = model(**inputs)

# Clean up
for h in hooks:
    h.remove()

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)