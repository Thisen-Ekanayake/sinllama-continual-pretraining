import json

FILES = [
    "/ml/Datasets/Wikipedia_Dataset/short_articles.json",
    "/ml/Datasets/Wikipedia_Dataset/medium_articles.json",
]

BOUNDS = [256, 512, 1024]


def bucket(wc):
    if wc < 256:
        return "<256"
    elif wc < 512:
        return "256-512"
    elif wc < 1024:
        return "512-1024"
    else:
        return ">1024"


order = ["<256", "256-512", "512-1024", ">1024"]
totals = {b: 0 for b in order}

for path in FILES:
    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    counts = {b: 0 for b in order}
    for article in articles:
        wc = article.get("text_word_count", len(article.get("text", "").split()))
        b = bucket(wc)
        counts[b] += 1
        totals[b] += 1

    print(f"{path} ({len(articles)} articles)")
    for b in order:
        print(f"  {b}: {counts[b]}")

print("\nCombined")
for b in order:
    print(f"  {b}: {totals[b]}")
