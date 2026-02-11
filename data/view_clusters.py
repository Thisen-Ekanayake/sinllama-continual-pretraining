import json
import re
from collections import defaultdict

INPUT_FILE = "data/oasst2_clustered.jsonl"
OUTPUT_FILE = "data/filtered_clusters.txt"

# ----------------------------------
# Keywords (expandable)
# ----------------------------------

TECH_KEYWORDS = [
    "python", "bash", "terraform", "aws", "linux", "git",
    "machine learning", "function", "code", "deep learning",
    "api", "script", "program", "docker", "kubernetes"
]

MATH_KEYWORDS = [
    "equation", "integral", "derivative", "matrix",
    "vector", "theorem", "proof", "algebra",
    "calculus", "probability", "statistics",
    "geometry", "trigonometry", "logarithm"
]

ALL_KEYWORDS = TECH_KEYWORDS + MATH_KEYWORDS

# ----------------------------------
# Load clusters
# ----------------------------------

clusters = defaultdict(list)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        clusters[obj["cluster_id"]].append(obj)

# ----------------------------------
# Process and write
# ----------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for cid in sorted(clusters.keys()):

        seen = set()
        unique_prompts = []

        # Collect unique prompts
        for sample in clusters[cid]:
            text = sample["messages"][0]["content"].strip()

            norm = re.sub(r"\s+", " ", text.lower())

            if norm not in seen:
                seen.add(norm)
                unique_prompts.append(text)

        # Count keyword occurrences across entire cluster
        keyword_count = 0

        for prompt in unique_prompts:
            text_lower = prompt.lower()

            for kw in ALL_KEYWORDS:
                # Count full word matches
                matches = re.findall(rf"\b{re.escape(kw)}\b", text_lower)
                keyword_count += len(matches)

        # If cluster looks technical/math-heavy, skip writing it
        if keyword_count > 1:
            print(f"Skipping cluster {cid} (keyword_count={keyword_count})")
            continue

        # Otherwise write cluster to TXT
        out.write("\n====================\n")
        out.write(f"Cluster: {cid}\n")
        out.write(f"Total unique prompts: {len(unique_prompts)}\n\n")

        for prompt in unique_prompts:
            out.write(prompt + "\n")

print("Finished. Filtered clusters written to TXT.")