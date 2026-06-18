import json
from tqdm import tqdm
import torch
from transformers import BertModel
import sentencepiece as spm
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# --- Paths and settings ---
MODEL_PATH = "HelaBERT/HelaBERT"
TOKENIZER_PATH = "HelaBERT/tokenizer/unigram_32000_0.9995.model"
DOWNSTREAM_FILE = "data/evaluate/processed_eval_dataset/processed_sentiment/train.jsonl"
CPT_FILE = "All-Text_8696658_147190824.normalized.txt"

BATCH_SIZE = 64  # adjust for GPU RAM
EMBED_DIM = 384  # HelaBERT hidden size

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load tokenizer and model ---
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)

model = BertModel.from_pretrained(MODEL_PATH, add_pooling_layer=False, ignore_mismatched_sizes=True)
model.eval()
model.to(device)

# --- Utility functions ---
def embed_texts(texts):
    """Compute mean-pooled embeddings for a batch of texts."""
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        input_ids = [sp.encode(t, out_type=int)[:512] for t in batch]
        max_len = max(len(ids) for ids in input_ids)
        input_ids_padded = [ids + [sp.pad_id()]*(max_len-len(ids)) for ids in input_ids]
        input_ids_tensor = torch.tensor(input_ids_padded).to(device)
        attention_mask = (input_ids_tensor != sp.pad_id()).float().unsqueeze(-1)
        with torch.no_grad():
            outputs = model(input_ids_tensor)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0).numpy()

def downstream_text_generator(file_path, buffer_size=1000):
    buffer = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            buffer.append(data["text"])
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []
        if buffer:
            yield buffer

def cpt_text_generator(file_path, chunk_size=100_000):
    buffer = ""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            buffer += line
            if len(buffer) >= chunk_size:
                yield [buffer]  # wrap in list to match embed_texts
                buffer = ""
        if buffer:
            yield [buffer]

def compute_mean_cov(embedding_gen):
    """Compute mean vector and covariance matrix incrementally."""
    total_count = 0
    mean = np.zeros(EMBED_DIM)
    M2 = np.zeros((EMBED_DIM, EMBED_DIM))  # for covariance

    for chunk in tqdm(embedding_gen, desc="Embedding chunks"):
        emb = embed_texts(chunk)
        n = emb.shape[0]
        new_total = total_count + n
        delta = emb.mean(axis=0) - mean
        mean += delta * n / new_total
        # Incremental covariance formula
        centered = emb - mean
        M2 += centered.T @ centered
        total_count = new_total

    cov = M2 / total_count
    return mean, cov

def gaussian_kl_stable(mu1, sigma1, mu2, sigma2, eps=1e-3):
    """Stable KL(N1||N2) for multivariate Gaussians using SVD + diagonal smoothing."""
    d = mu1.shape[0]
    sigma1 += np.eye(d) * eps
    sigma2 += np.eye(d) * eps

    # Use pseudo-inverse in case sigma2 is near-singular
    try:
        inv_sigma2 = np.linalg.inv(sigma2)
    except np.linalg.LinAlgError:
        inv_sigma2 = np.linalg.pinv(sigma2)

    sign1, logdet1 = np.linalg.slogdet(sigma1)
    sign2, logdet2 = np.linalg.slogdet(sigma2)
    if sign1 <= 0 or sign2 <= 0:
        # fallback to sum of log of singular values
        logdet1 = np.sum(np.log(np.linalg.svd(sigma1, compute_uv=False) + eps))
        logdet2 = np.sum(np.log(np.linalg.svd(sigma2, compute_uv=False) + eps))

    trace_term = np.trace(inv_sigma2 @ sigma1)
    diff = mu2 - mu1
    kl = 0.5 * (logdet2 - logdet1 - d + trace_term + diff.T @ inv_sigma2 @ diff)
    return kl

def gaussian_js_stable(mu1, sigma1, mu2, sigma2):
    mu_m = 0.5 * (mu1 + mu2)
    sigma_m = 0.5 * (sigma1 + sigma2)
    kl1 = gaussian_kl_stable(mu1, sigma1, mu_m, sigma_m)
    kl2 = gaussian_kl_stable(mu2, sigma2, mu_m, sigma_m)
    return 0.5 * (kl1 + kl2)

def gaussian_js(mu1, sigma1, mu2, sigma2):
    """JS divergence via Gaussian approximation"""
    mu_m = 0.5 * (mu1 + mu2)
    sigma_m = 0.5 * (sigma1 + sigma2)
    kl1 = gaussian_kl_stable(mu1, sigma1, mu_m, sigma_m)
    kl2 = gaussian_kl_stable(mu2, sigma2, mu_m, sigma_m)
    return 0.5 * (kl1 + kl2)

# --- Main ---
print("Computing downstream embeddings statistics...")
downstream_gen = downstream_text_generator(DOWNSTREAM_FILE)
mu_down, cov_down = compute_mean_cov(downstream_gen)

print("Computing CPT embeddings statistics...")
cpt_gen = cpt_text_generator(CPT_FILE)
mu_cpt, cov_cpt = compute_mean_cov(cpt_gen)

print("Computing semantic JS divergence...")
js_div = gaussian_js(mu_down, cov_down, mu_cpt, cov_cpt)
print(f"Semantic JS divergence (downstream || CPT) = {js_div:.6f}")

# --- Function to sample embeddings ---
def sample_embeddings(generator, sample_size=50000):
    embeddings_list = []
    count = 0
    for chunk in tqdm(generator, desc="Embedding chunks for sampling"):
        emb = embed_texts(chunk)
        embeddings_list.append(emb)
        count += emb.shape[0]
        if count >= sample_size:
            break
    embeddings_all = np.concatenate(embeddings_list, axis=0)
    if embeddings_all.shape[0] > sample_size:
        indices = np.random.choice(embeddings_all.shape[0], sample_size, replace=False)
        embeddings_all = embeddings_all[indices]
    return embeddings_all

# --- Sample embeddings ---
downstream_emb_sample = sample_embeddings(downstream_text_generator(DOWNSTREAM_FILE), sample_size=5000)
cpt_emb_sample = sample_embeddings(cpt_text_generator(CPT_FILE), sample_size=50000)

# --- Combine and reduce to 3D ---
all_embeddings = np.concatenate([downstream_emb_sample, cpt_emb_sample], axis=0)
pca = PCA(n_components=3)
reduced = pca.fit_transform(all_embeddings)

# Split back for plotting
downstream_reduced = reduced[:downstream_emb_sample.shape[0]]
cpt_reduced = reduced[downstream_emb_sample.shape[0]:]

# --- Create Plotly 3D scatter plot ---
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=cpt_reduced[:,0], y=cpt_reduced[:,1], z=cpt_reduced[:,2],
    mode='markers',
    marker=dict(size=3, color='blue', opacity=0.3),
    name='CPT'
))

fig.add_trace(go.Scatter3d(
    x=downstream_reduced[:,0], y=downstream_reduced[:,1], z=downstream_reduced[:,2],
    mode='markers',
    marker=dict(size=5, color='red', opacity=0.8),
    name='Downstream'
))

fig.update_layout(
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    ),
    title='Interactive 3D PCA of HelaBERT embeddings',
    width=900,
    height=700
)

fig.show()