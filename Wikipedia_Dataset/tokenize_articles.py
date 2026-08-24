import json
import statistics
from tokenizers import Tokenizer

TOKENIZER_PATH = "/ml/Datasets/Wikipedia_Dataset/tokenizer/tokenizer.json"

FILES = {
    "short": "/ml/Datasets/Wikipedia_Dataset/short_articles.json",
    "medium": "/ml/Datasets/Wikipedia_Dataset/medium_articles.json",
}

BUCKET_ORDER = ["<256", "256-512", "512-1024", ">1024"]


def bucket(n):
    if n < 256:
        return "<256"
    elif n < 512:
        return "256-512"
    elif n < 1024:
        return "512-1024"
    else:
        return ">1024"


def main():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    grand_total_tokens = 0
    grand_counts = {b: 0 for b in BUCKET_ORDER}
    grand_token_counts = []

    for label, path in FILES.items():
        with open(path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        token_counts = []
        for article in articles:
            ids = tokenizer.encode(article.get("text", ""), add_special_tokens=False).ids
            n = len(ids)
            article["token_count"] = n
            token_counts.append(n)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        total = sum(token_counts)
        counts = {b: 0 for b in BUCKET_ORDER}
        for n in token_counts:
            counts[bucket(n)] += 1

        print(f"{label} ({path}) — {len(articles)} articles")
        print(f"  total tokens: {total}")
        print(f"  mean: {statistics.mean(token_counts):.1f}  median: {statistics.median(token_counts):.1f}  "
              f"min: {min(token_counts)}  max: {max(token_counts)}")
        for b in BUCKET_ORDER:
            print(f"  {b}: {counts[b]}")
        print()

        grand_total_tokens += total
        grand_token_counts.extend(token_counts)
        for b in BUCKET_ORDER:
            grand_counts[b] += counts[b]

    print(f"Combined — {len(grand_token_counts)} articles")
    print(f"  total tokens: {grand_total_tokens}")
    print(f"  mean: {statistics.mean(grand_token_counts):.1f}  median: {statistics.median(grand_token_counts):.1f}  "
          f"min: {min(grand_token_counts)}  max: {max(grand_token_counts)}")
    for b in BUCKET_ORDER:
        print(f"  {b}: {grand_counts[b]}")


if __name__ == "__main__":
    main()
