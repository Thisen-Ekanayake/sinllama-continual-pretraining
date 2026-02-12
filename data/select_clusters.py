import json

CLUSTER_EVAL_FILE = "data/cluster_evaluation.jsonl"
INPUT_FILE = "data/oasst2_clustered.jsonl"
OUTPUT_FILE = "data/oasst2_selected.jsonl"

SUITABILITY_THRESHOLD = 4  # change if needed

# ----------------------------------
# Step 1: Extract cluster IDs
# ----------------------------------

keep_clusters = set()

with open(CLUSTER_EVAL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        cid = obj["cluster_id"]
        score = obj["evaluation"]["suitability_score"]

        if score >= SUITABILITY_THRESHOLD:
            keep_clusters.add(cid)

print(f"Keeping {len(keep_clusters)} clusters:", keep_clusters)

# ----------------------------------
# Step 2: Filter dataset
# ----------------------------------

kept_count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

    for line in f_in:
        obj = json.loads(line)

        if obj.get("cluster_id") in keep_clusters:
            obj.pop("cluster_id", None)  # remove cluster_id
            json.dump(obj, f_out, ensure_ascii=False)
            f_out.write("\n")
            kept_count += 1

print(f"Saved {kept_count} samples to {OUTPUT_FILE}")