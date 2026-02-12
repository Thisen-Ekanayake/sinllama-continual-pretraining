import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import hdbscan
import umap

INPUT_FILE = "data/oasst2_en_chat.jsonl"
OUTPUT_FILE = "data/oasst2_clustered.jsonl"

print("Loading data...")

texts = []
raw_data = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        raw_data.append(obj)

        # Use first user message for clustering
        for msg in obj["messages"]:
            if msg["role"] == "user":
                texts.append(msg["content"])
                break

print(f"Loaded {len(texts)} samples")

# ----------------------------
# Step 1: Embeddings
# ----------------------------

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding...")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

# ----------------------------
# Step 2: Dimensionality Reduction (optional but improves clustering)
# ----------------------------

print("Reducing dimensions with UMAP...")
reducer = umap.UMAP(
    n_neighbors=15,
    n_components=50,
    metric="cosine",
    random_state=42
)

reduced_embeddings = reducer.fit_transform(embeddings)

# ----------------------------
# Step 3: HDBSCAN Clustering
# ----------------------------

print("Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,
    metric="euclidean",
    cluster_selection_method="eom"
)

cluster_labels = clusterer.fit_predict(reduced_embeddings)

print(f"Found {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters")
print(f"Noise points: {(cluster_labels == -1).sum()}")

# ----------------------------
# Step 4: Save Clustered Data
# ----------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for obj, label in zip(raw_data, cluster_labels):
        obj["cluster_id"] = int(label)
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")

print("Saved clustered dataset.")