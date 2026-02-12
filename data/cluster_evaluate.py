import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Load API key
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=api_key)

INPUT_FILE = "data/filtered_clusters.txt"
OUTPUT_FILE = "data/cluster_evaluation.jsonl"

# -----------------------------
# Parse clusters from TXT
# -----------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# Split clusters
raw_clusters = content.split("====================")

clusters = []

for block in raw_clusters:
    block = block.strip()
    if not block:
        continue

    lines = block.split("\n")
    cid_line = lines[0]

    match = re.search(r"Cluster:\s*(-?\d+)", cid_line)
    if not match:
        continue

    cid = int(match.group(1))

    # Skip metadata lines (Cluster + Total unique prompts)
    prompts = lines[2:]

    # Remove empty lines
    prompts = [p.strip() for p in prompts if p.strip()]

    clusters.append((cid, prompts))

print(f"Loaded {len(clusters)} clusters for evaluation")

# -----------------------------
# Evaluate with OpenAI
# -----------------------------

def evaluate_cluster(cluster_id, prompts):
    sample = prompts[:30]  # limit to 30 prompts

    prompt_text = (
        "You are evaluating conversation prompts.\n\n"
        "Below are example user prompts from one cluster:\n\n"
    )

    for i, p in enumerate(sample):
        prompt_text += f"{i+1}. {p}\n"

    prompt_text += """
Classify this cluster into one of:
- Technical
- Academic
- Casual Chat
- Creative Writing
- General Knowledge
- Mixed

Then rate its suitability for training a natural conversational assistant (1–5).

Respond ONLY in JSON format:
{
  "cluster_type": "...",
  "suitability_score": 1-5,
  "reason": "short explanation"
}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0
    )

    return response.choices[0].message.content


# -----------------------------
# Process all clusters
# -----------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for cid, prompts in clusters:
        print(f"Evaluating cluster {cid}...")

        try:
            result = evaluate_cluster(cid, prompts)

            # Try parsing JSON
            parsed = json.loads(result)

            output_obj = {
                "cluster_id": cid,
                "evaluation": parsed
            }

            json.dump(output_obj, out, ensure_ascii=False)
            out.write("\n")

        except Exception as e:
            print(f"Error in cluster {cid}: {e}")

print("Finished evaluation.")