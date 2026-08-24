import json
import os

SRC = "/ml/Datasets/Wikipedia_Dataset/extracted_categorized"
OUT = {
    "short": "/ml/Datasets/Wikipedia_Dataset/short_articles.json",
    "medium": "/ml/Datasets/Wikipedia_Dataset/medium_articles.json",
}

buckets = {"short": [], "medium": []}

for root, dirs, files in os.walk(SRC):
    for fname in files:
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(root, fname), "r", encoding="utf-8") as f:
            articles = json.load(f)
        for article in articles:
            cat = article.get("category")
            if cat in buckets:
                buckets[cat].append(article)

for cat, path in OUT.items():
    with open(path, "w", encoding="utf-8") as f:
        json.dump(buckets[cat], f, ensure_ascii=False, indent=2)
    print(f"{cat}: {len(buckets[cat])} articles -> {path}")
