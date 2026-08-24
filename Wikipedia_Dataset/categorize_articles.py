import json
import os

SRC = "/ml/Datasets/Wikipedia_Dataset/extracted_json"
DST = "/ml/Datasets/Wikipedia_Dataset/extracted_categorized"

# Length buckets based on text word count (standard Wikipedia stub/short/medium/long convention)
CATEGORY_BOUNDS = [
    (150, "stub"),
    (500, "short"),
    (2000, "medium"),
]
DEFAULT_CATEGORY = "long"


def word_count(s):
    return len(s.split())


def categorize(count):
    for bound, label in CATEGORY_BOUNDS:
        if count < bound:
            return label
    return DEFAULT_CATEGORY


def main():
    total_files = 0
    total_articles = 0
    category_totals = {}

    for root, dirs, files in os.walk(SRC):
        rel = os.path.relpath(root, SRC)
        for fname in files:
            if not fname.endswith(".json"):
                continue
            src_path = os.path.join(root, fname)
            dst_dir = os.path.join(DST, rel) if rel != "." else DST
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, fname)

            with open(src_path, "r", encoding="utf-8") as f:
                articles = json.load(f)

            for article in articles:
                title_wc = word_count(article.get("title", ""))
                text_wc = word_count(article.get("text", ""))
                article["title_word_count"] = title_wc
                article["text_word_count"] = text_wc
                article["category"] = categorize(text_wc)
                category_totals[article["category"]] = category_totals.get(article["category"], 0) + 1

            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)

            total_files += 1
            total_articles += len(articles)

    print(f"Processed {total_files} files, {total_articles} articles")
    for label, count in sorted(category_totals.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
