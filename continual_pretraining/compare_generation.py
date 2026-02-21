import os
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "/workspace/model/SinLlama_merged_bf16")
CPT_MODEL_PATH  = os.environ.get("CPT_MODEL_PATH", "/workspace/sinllama_cpt_out/stage2/merged_bf16")

MAX_NEW_TOKENS  = int(os.environ.get("MAX_NEW_TOKENS", "256"))
TEMPERATURE     = float(os.environ.get("TEMPERATURE", "0.8"))
TOP_P           = float(os.environ.get("TOP_P", "0.9"))

RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)

print("Loading CPT model...")
cpt_model = AutoModelForCausalLM.from_pretrained(
    CPT_MODEL_PATH,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)

base_model.eval()
cpt_model.eval()

# ============================================================
# Prompts (~20 words each)
# ============================================================

prompts = [
    "ශ්‍රී ලංකාවේ ආර්ථික අර්බුදය පසුගිය වසර කිහිපයේ ජනතාවගේ ජීවන තත්ත්වයට විශාල බලපෑමක් ඇති කළා.",
    "කෘත්‍රිම බුද්ධිය සහ යන්ත්‍ර ඉගෙනීම අනාගතයේ රැකියා වෙළඳපොළට කෙසේ බලපානු ඇතැයි ඔබ සිතනවාද?",
    "විශ්ව විද්‍යාල ශිෂ්‍යයන්ට පර්යේෂණ හා නවෝත්පාදන සඳහා වැඩි අවස්ථා ලබාදීම වැදගත් වේ.",
    "පරිසර හිතකාමී බලශක්ති මාර්ග වෙත මාරුවීම ශ්‍රී ලංකාවට දිගුකාලීන වාසි ලබාදෙනු ඇත.",
    "සමාජ මාධ්‍ය මගින් තොරතුරු ව්‍යාප්තිය වේගවත් වුවද වැරදි තොරතුරුද පහසුවෙන් පැතිරේ.",
    "නව තාක්ෂණික උපාංග භාවිතයෙන් කෘෂිකර්ම ක්ෂේත්‍රයේ ඵලදායීතාව වැඩි කළ හැක.",
    "භාෂා මාදිලි සංවර්ධනය කිරීමේදී දත්තයේ ගුණාත්මකභාවය ඉතා වැදගත් සාධකයකි.",
    "තරුණ පරපුරට මූල්‍ය දැනුම ලබාදීම ඔවුන්ගේ අනාගත සැලසුම් සඳහා ප්‍රයෝජනවත් වේ.",
    "ගෝලීය උෂ්ණත්වය ඉහළ යාම හේතුවෙන් ස්වාභාවික විපත් වැඩි වීමක් දැකිය හැක.",
    "සංස්කෘතික උරුමය සහ නවීනත්වය අතර සම්මතයක් සොයා ගැනීම සමාජයට අභියෝගයකි."
]

# ============================================================
# Generation Function
# ============================================================

def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text[len(prompt):].strip()


# ============================================================
# Generate & Save
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(RESULTS_DIR, f"generation_compare_{timestamp}.txt")

with open(output_path, "w") as f:
    for i, prompt in enumerate(prompts, 1):

        print(f"Generating prompt {i}/10")

        base_out = generate(base_model, prompt)
        cpt_out  = generate(cpt_model, prompt)

        f.write("="*80 + "\n")
        f.write("PROMPT:\n")
        f.write(prompt + "\n\n")

        f.write("BASE MODEL OUTPUT:\n")
        f.write(base_out + "\n\n")

        f.write("CPT MODEL OUTPUT:\n")
        f.write(cpt_out + "\n\n")

print(f"\nGeneration comparison saved to: {output_path}")