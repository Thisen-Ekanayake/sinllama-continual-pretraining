import csv
import json
import os
import re
from collections import Counter
from sklearn.model_selection import train_test_split

# ==============================
# CONFIG
# ==============================

INPUT_CSV = os.environ.get("CSV_PATH", "data/evaluate/Sinhala-writing-style-classfication/writesty3.csv")
OUTPUT_DIR = os.environ.get("OUT_DIR", "data/evaluate/processed_writing_style")
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# HELPERS
# ==============================

def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

# ==============================
# LOAD DATA
# ==============================

data = []
seen_texts = set()

with open(INPUT_CSV, "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        text = clean_text(row["comments"])
        label = row["labels"].strip().upper()

        if not text:
            continue

        if text in seen_texts:
            continue

        data.append({
            "text": text,
            "label": label
        })

        seen_texts.add(text)

print(f"Total after cleaning & dedup: {len(data)}")

# ==============================
# DETECT LABELS
# ==============================

labels = [item["label"] for item in data]
label_counts = Counter(labels)

print("Detected labels and counts:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

# ==============================
# STRATIFIED SPLIT 80/10/10
# ==============================

texts = [item["text"] for item in data]
labels = [item["label"] for item in data]

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=RANDOM_SEED
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=0.5,
    stratify=temp_labels,
    random_state=RANDOM_SEED
)

# ==============================
# SAVE JSONL
# ==============================

def save_jsonl(filename, texts, labels):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for text, label in zip(texts, labels):
            row = {
                "text": text,
                "label": label
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

save_jsonl("train.jsonl", train_texts, train_labels)
save_jsonl("val.jsonl", val_texts, val_labels)
save_jsonl("test.jsonl", test_texts, test_labels)

print("Done.")
print(f"Train: {len(train_texts)}")
print(f"Val: {len(val_texts)}")
print(f"Test: {len(test_texts)}")