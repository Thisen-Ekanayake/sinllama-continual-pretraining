import json
from collections import Counter
import math
from tqdm import tqdm
from transformers import AutoTokenizer

MODEL_PATH = "SinLlama_merged_bf16"
BUFFER_SIZE = 1000  # number of lines to process at a time

def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def compute_token_distribution_from_file(file_path, tokenizer, is_jsonl=True):
    """Compute token counts incrementally from file to save memory."""
    token_counts = Counter()
    total_tokens = 0

    if is_jsonl:
        with open(file_path, "r", encoding="utf-8") as f:
            buffer = []
            for line in tqdm(f, desc=f"Processing {file_path}"):
                data = json.loads(line)
                buffer.append(data["text"])
                if len(buffer) >= BUFFER_SIZE:
                    for text in buffer:
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        token_counts.update(tokens)
                        total_tokens += len(tokens)
                    buffer = []
            # Process remaining buffer
            for text in buffer:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_counts.update(tokens)
                total_tokens += len(tokens)
    else:
        # For plain text file, read in chunks
        with open(file_path, "r", encoding="utf-8") as f:
            buffer_text = ""
            for line in tqdm(f, desc=f"Processing {file_path}"):
                buffer_text += line
                if len(buffer_text) > 100_000:  # ~100k chars per chunk
                    tokens = tokenizer.encode(buffer_text, add_special_tokens=False)
                    token_counts.update(tokens)
                    total_tokens += len(tokens)
                    buffer_text = ""
            # Process remaining text
            if buffer_text:
                tokens = tokenizer.encode(buffer_text, add_special_tokens=False)
                token_counts.update(tokens)
                total_tokens += len(tokens)

    # Convert counts to probabilities
    token_probs = {token: count / total_tokens for token, count in token_counts.items()}
    return token_probs

def kl_divergence(p_dist, q_dist, eps=1e-12):
    """Compute KL(P || Q) with smoothing."""
    kl = 0.0
    for token, p_prob in p_dist.items():
        q_prob = q_dist.get(token, eps)
        kl += p_prob * math.log(p_prob / q_prob)
    return kl

def main():
    tokenizer = load_tokenizer()
    print("Tokenizer loaded.")

    downstream_file = "data/evaluate/processed_eval_dataset/processed_news/train.jsonl"
    cpt_file = "All-Text_8696658_147190824.normalized.txt"

    print("Computing token distribution for downstream dataset...")
    p_dist = compute_token_distribution_from_file(downstream_file, tokenizer, is_jsonl=True)

    print("Computing token distribution for CPT dataset...")
    q_dist = compute_token_distribution_from_file(cpt_file, tokenizer, is_jsonl=False)

    print("Computing KL divergence...")
    kl = kl_divergence(p_dist, q_dist)
    print(f"KL(P_downstream || Q_CPT) = {kl:.6f}")

if __name__ == "__main__":
    main()