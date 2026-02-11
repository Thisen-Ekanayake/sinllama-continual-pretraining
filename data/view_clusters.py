import json
from collections import defaultdict

clusters = defaultdict(list)

with open("data/oasst2_clustered.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        clusters[obj["cluster_id"]].append(obj)

for cid in sorted(clusters.keys()):
    print("\n====================")
    print("Cluster:", cid)
    print("Total items:", len(clusters[cid]))

    seen = set()
    unique_count = 0

    print("Unique samples:\n")

    for sample in clusters[cid]:
        text = sample["messages"][0]["content"].strip()

        if text not in seen:
            seen.add(text)
            print(text[:200])
            unique_count += 1

        if unique_count >= 10:  # show 10 unique samples
            break

    print("Unique shown:", unique_count)
