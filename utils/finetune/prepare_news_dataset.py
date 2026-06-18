import os
import json
import random
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ==============================
# CONFIG
# ==============================

DATA_DIR = os.environ.get("IN_DIR", "data/evaluate/Sinhala_News_Classification")  # directory where txt files are located
OUTPUT_DIR = os.environ.get("OUT_DIR", "data/evaluate/processed_news")
SPLIT_RATIO = (0.8, 0.1, 0.1)
RANDOM_SEED = 42

# Label mapping (keep consistent)
LABEL_MAP = {
    "Politics.txt": 0,
    "business.txt": 1,
    "Science_technology.txt": 2,
    "Sports.txt": 3,
    "entertainment.txt": 4,
}

LABEL_NAMES = {
    0: "politics",
    1: "business",
    2: "science_technology",
    3: "sports",
    4: "entertainment",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(RANDOM_SEED)

# ==============================
# HELPER FUNCTIONS
# ==============================

def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def read_articles(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by multiple & separators
    articles = re.split(r"&{5,}", content)

    cleaned = []
    for article in articles:
        article = clean_text(article)
        if len(article) > 20:  # ignore very short junk
            cleaned.append(article)

    return cleaned

# ==============================
# LOAD DATA
# ==============================

all_data = []

for filename, label_id in LABEL_MAP.items():
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {filename} not found.")
        continue

    articles = read_articles(path)

    for article in articles:
        all_data.append({
            "text": article,
            "label": label_id,
            "label_name": LABEL_NAMES[label_id]
        })

print(f"Total articles before dedup: {len(all_data)}")

# ==============================
# DEDUPLICATION
# ==============================

unique_data = []
seen = set()

for item in all_data:
    if item["text"] not in seen:
        unique_data.append(item)
        seen.add(item["text"])

print(f"Total articles after dedup: {len(unique_data)}")

# ==============================
# STRATIFIED SPLIT
# ==============================

texts = [item["text"] for item in unique_data]
labels = [item["label"] for item in unique_data]

# First split: train vs temp (val+test)
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts,
    labels,
    test_size=(1 - SPLIT_RATIO[0]),
    stratify=labels,
    random_state=RANDOM_SEED,
)

# Second split: val vs test
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=SPLIT_RATIO[2] / (SPLIT_RATIO[1] + SPLIT_RATIO[2]),
    stratify=temp_labels,
    random_state=RANDOM_SEED,
)

def save_jsonl(filename, texts, labels):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for text, label in zip(texts, labels):
            row = {
                "text": text,
                "label": label,
                "label_name": LABEL_NAMES[label]
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

save_jsonl("train.jsonl", train_texts, train_labels)
save_jsonl("val.jsonl", val_texts, val_labels)
save_jsonl("test.jsonl", test_texts, test_labels)

print("Dataset saved successfully.")
print(f"Train size: {len(train_texts)}")
print(f"Val size: {len(val_texts)}")
print(f"Test size: {len(test_texts)}")