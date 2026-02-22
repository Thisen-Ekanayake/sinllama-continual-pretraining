import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report
from tqdm import tqdm
from datetime import datetime

MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/model/SinLlama_CPT")
TEST_FILE  = os.environ.get("TEST_FILE", "/workspace/data/classification")
TASK       = os.environ.get("TASK", "sentiment")  # sentiment | writing | news
MAX_NEW_TOKENS = 5

RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(RESULTS_DIR, f"{TASK}_classification.txt")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def build_prompt(text):
    if TASK == "writing":
        return f"""Classify the Sinhala comment into ACADEMIC, CREATIVE, NEWS, BLOG.

Comment: {text}
Answer:"""

    elif TASK == "sentiment":
        return f"""Does the following Sinhala sentence have a POSITIVE, NEGATIVE or NEUTRAL sentiment?

{text}

Answer:"""

    elif TASK == "news":
        return f"""Classify into:
Political: 0, Business: 1, Technology: 2, Sports: 3, Entertainment: 4.

Comment: {text}
Answer:"""

y_true = []
y_pred = []

with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        row = json.loads(line)
        text = row["text"]
        label = str(row["label"])

        prompt = build_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = decoded.split("Answer:")[-1].strip().split()[0]

        y_true.append(label)
        y_pred.append(prediction)

report = classification_report(y_true, y_pred, digits=4)

# Save results
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("Classification Evaluation Report\n")
    f.write("========================================\n")
    f.write(f"Timestamp     : {datetime.now()}\n")
    f.write(f"Task          : {TASK}\n")
    f.write(f"Model Path    : {MODEL_PATH}\n")
    f.write(f"Test File     : {TEST_FILE}\n")
    f.write(f"Max New Tokens: {MAX_NEW_TOKENS}\n")
    f.write("----------------------------------------\n\n")
    f.write(report)

print(report)
print(f"\nResults saved to: {OUTPUT_FILE}")