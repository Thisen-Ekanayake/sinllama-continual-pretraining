import os
import re
import json
from datetime import datetime
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ============================================================
# CONFIG (ENV-STYLE)
# ============================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/model/SinLlama_CPT")
TEST_FILE  = os.environ.get("TEST_FILE", "/workspace/data/classification/test.jsonl")
TASK       = os.environ.get("TASK", "sentiment")  # sentiment | writing | news
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "5"))

RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(RESULTS_DIR, f"{TASK}_classification.txt")

# ============================================================
# PROMPTS
# ============================================================

def build_prompt(text: str) -> str:
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

    else:
        raise ValueError(f"Unknown TASK='{TASK}'. Use: sentiment | writing | news")


# ============================================================
# LABEL SETUP (for consistent confusion matrix ordering)
# ============================================================

def get_label_order():
    if TASK == "news":
        return ["0", "1", "2", "3", "4"]
    if TASK == "sentiment":
        return ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    if TASK == "writing":
        return ["ACADEMIC", "CREATIVE", "NEWS", "BLOG"]
    return None


# ============================================================
# PREDICTION PARSING (FIXES '0,' ISSUE)
# ============================================================

def parse_prediction(decoded: str) -> str:
    raw = decoded.split("Answer:")[-1].strip()

    if TASK == "news":
        m = re.search(r"\b([0-4])\b", raw)
        return m.group(1) if m else "UNK"

    if TASK == "sentiment":
        ru = raw.upper()
        if "POSITIVE" in ru:
            return "POSITIVE"
        if "NEGATIVE" in ru:
            return "NEGATIVE"
        if "NEUTRAL" in ru:
            return "NEUTRAL"
        return "UNK"

    if TASK == "writing":
        ru = raw.upper()
        if "ACADEMIC" in ru:
            return "ACADEMIC"
        if "CREATIVE" in ru:
            return "CREATIVE"
        if re.search(r"\bNEWS\b", ru):
            return "NEWS"
        if "BLOG" in ru:
            return "BLOG"
        return "UNK"

    return "UNK"


# ============================================================
# PRETTY PRINT HELPERS (TEXT FILE FRIENDLY)
# ============================================================

def format_matrix(labels, matrix, title=None):
    """
    Returns a monospaced table string for confusion matrices.
    """
    col_width = max(6, max(len(l) for l in labels) + 2)
    header = " " * col_width + "".join(l.rjust(col_width) for l in labels)
    lines = []
    if title:
        lines.append(title)
    lines.append(header)
    for i, row_label in enumerate(labels):
        row = "".join(str(v).rjust(col_width) for v in matrix[i])
        lines.append(row_label.rjust(col_width) + row)
    return "\n".join(lines)

def safe_div(a, b):
    return (a / b) if b else 0.0


# ============================================================
# LOAD MODEL
# ============================================================

print(f"[INFO] Loading model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# ============================================================
# EVALUATE
# ============================================================

y_true = []
y_pred = []

pred_counter = Counter()
true_counter = Counter()
unk_examples = []

print(f"[INFO] Reading test file: {TEST_FILE}")
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for idx, line in enumerate(tqdm(f), start=1):
        row = json.loads(line)
        text = row["text"]
        label = str(row["label"])

        prompt = build_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                # avoid warnings if generation_config contains sampling params
                temperature=None,
                top_p=None,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = parse_prediction(decoded)

        y_true.append(label)
        y_pred.append(prediction)

        true_counter[label] += 1
        pred_counter[prediction] += 1

        if prediction == "UNK" and len(unk_examples) < 5:
            unk_examples.append({
                "index": idx,
                "label": label,
                "text": text[:200],
                "decoded_tail": decoded[-300:]
            })

# ============================================================
# METRICS
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    digits=4,
    zero_division=0
)

label_order = get_label_order()

cm = confusion_matrix(y_true, y_pred, labels=label_order)

# Per-class accuracy (correct/total for each true class = diagonal / row sum)
per_class = []
for i, lab in enumerate(label_order):
    total = cm[i].sum()
    correct = cm[i, i]
    acc = safe_div(correct, total)
    per_class.append((lab, int(correct), int(total), acc))

# Normalized confusion matrix (row-normalized)
cm_norm = []
for i in range(len(label_order)):
    row_sum = cm[i].sum()
    if row_sum == 0:
        cm_norm.append([0.0] * len(label_order))
    else:
        cm_norm.append([round(v / row_sum, 4) for v in cm[i]])

# ============================================================
# SAVE OUTPUT
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    out.write("Classification Evaluation Report\n")
    out.write("========================================\n")
    out.write(f"Timestamp      : {datetime.now()}\n")
    out.write(f"Task           : {TASK}\n")
    out.write(f"Model Path     : {MODEL_PATH}\n")
    out.write(f"Test File      : {TEST_FILE}\n")
    out.write(f"Max New Tokens : {MAX_NEW_TOKENS}\n")
    out.write("----------------------------------------\n\n")

    out.write("Label Distribution (True)\n")
    out.write("-------------------------\n")
    for k, v in sorted(true_counter.items(), key=lambda x: x[0]):
        out.write(f"{k}: {v}\n")

    out.write("\nLabel Distribution (Pred)\n")
    out.write("-------------------------\n")
    for k, v in sorted(pred_counter.items(), key=lambda x: str(x[0])):
        out.write(f"{k}: {v}\n")

    out.write("\n\nClassification Report\n")
    out.write("---------------------\n")
    out.write(report)

    out.write("\n\nConfusion Matrix (Counts)\n")
    out.write("-------------------------\n")
    out.write(format_matrix(label_order, cm) + "\n")

    out.write("\nPer-Class Accuracy (Correct/Total)\n")
    out.write("----------------------------------\n")
    for lab, correct, total, acc in per_class:
        out.write(f"{lab:>12} : {correct:>4}/{total:<4}  acc={acc:.4f}\n")

    out.write("\n\nConfusion Matrix (Row-normalized)\n")
    out.write("---------------------------------\n")
    out.write(format_matrix(label_order, cm_norm) + "\n")

    if pred_counter.get("UNK", 0) > 0:
        out.write("\nUNK Predictions\n")
        out.write("---------------\n")
        out.write(f"UNK count: {pred_counter['UNK']}\n")

    if unk_examples:
        out.write("\nUNK Examples (first 5)\n")
        out.write("----------------------\n")
        for ex in unk_examples:
            out.write(f"\n[#{ex['index']}] true={ex['label']}\n")
            out.write(f"text: {ex['text']}\n")
            out.write("decoded_tail:\n")
            out.write(ex["decoded_tail"] + "\n")

print(report)
print(f"\n[INFO] Results saved to: {OUTPUT_FILE}")

if pred_counter.get("UNK", 0) > 0:
    print(f"[WARN] UNK predictions: {pred_counter['UNK']} (see file for examples)")