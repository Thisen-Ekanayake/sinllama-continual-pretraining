# SinLlama: Sinhala Language Model

A comprehensive project for fine-tuning and optimizing the Llama-3-8b model for the Sinhala language using efficient training techniques like **LoRA (Low-Rank Adaptation)** and **QLoRA**.

**Focus:** Continuous Pre-training (CPT) on Sinhala text with advanced optimization strategies, model merging, evaluation, and inference utilities.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Components](#key-components)
3. [Main Training Script: `train_lora_cpt.py`](#main-training-script-train_lora_cptpy)
4. [Directory Structure](#directory-structure)
5. [Setup & Installation](#setup--installation)
6. [Usage](#usage)
7. [Other Important Files](#other-important-files)
8. [References](#references)

---

## Project Overview

**SinLlama** is an initiative to adapt the Llama-3-8b model for Sinhala language processing. The project focuses on:

- **Efficient Fine-tuning**: Using LoRA and QLoRA techniques to reduce memory footprint and training time
- **Data Processing**: Clustering and selecting high-quality Sinhala text and conversational data
- **Model Optimization**: Token packing, gradient accumulation, and mixed precision training
- **Evaluation & Analysis**: Comprehensive tools for perplexity evaluation, weight analysis, and model pruning
- **Production Ready**: Model merging, quantization, and inference utilities

The base model is **Llama-3-8b** (8 billion parameters), and fine-tuning is performed on:
- Sinhala conversational data (OASST2 dataset, translated to Sinhala)
- General Sinhala text corpus for continuous pre-training

---

## Key Components

### 1. **Fine-tuning Strategies** (`finetune/`)
Multiple fine-tuning approaches to handle different hardware constraints:

- **`train_lora_cpt.py`** ⭐ **[PRIMARY SCRIPT]**: Continuous Pre-training with LoRA
- **`qlora_finetune.py`**: QLoRA fine-tuning (4-bit quantization + LoRA)
- **`qlora_finetune_cpu.py`**: CPU-optimized QLoRA fine-tuning
- **`qlora_finetune_cpu_offload.py`**: CPU offloading strategy
- **`qlora_finetune_hybrid.py`**: Hybrid CPU-GPU training
- **`qlora_finetune_model_parallel.py`**: Multi-GPU model parallelism
- **`CPU_GPU_STRATEGY_COMPARISON.md`**: Detailed comparison of different strategies

### 2. **Data Processing** (`data/`)
- **`clustering.py`**: HDBSCAN clustering of conversational data
- **`select_clusters.py`**: Select high-quality clusters post-clustering
- **`translate_en_to_si.py`**: Translate English text to Sinhala (Google Translate API)
- **`cluster_evaluate.py`**: Evaluate cluster quality
- **`view_clusters.py`**: Visualization and inspection of clusters
- **`extract_en.py`**: Extract and filter English language data

### 3. **Model Analysis** (`model_analysis/`)
- **`weights_distribution_visualization.py`**: Visualize weight distributions
- **`wa_importance_visualization.py`**: Weight & activation importance analysis
- **`activation_analysis/`**: Activation patterns and neuron importance
- **`weight_analysis/`**: Per-layer weight analysis
- **`wa_importance_analysis_cpu_fp32/`**: CPU-friendly importance analysis

### 4. **Project Utilities** (`projects/`)
- **`merge_model.py`**: Merge LoRA adapters with base model
- **`merge_model_bfloat16.py`**: Merge in bfloat16 precision
- **`inference_bf16.py`**: Inference with bfloat16 quantization
- **`inference_cpu.py`**: CPU-only inference
- **`inference_merge_model.py`**: Use merged model for inference
- **`eval_ppl.py`**: Evaluate perplexity on test data
- **`check_perplexity.py`**: Quick perplexity checks
- **`prune_magnitude.py`**: Magnitude-based pruning
- **`masked_mlp_prune_and_eval.py`**: Selective layer pruning with evaluation
- **`export_column_importance.py`**: Export importance rankings for pruning
- **`lora_check.py`**: Verify LoRA adapter loading and parameters

### 5. **Pre-trained Models**
- **`llama-3-8b/`**: Base Llama-3 model (untouched)
- **`SinLlama_merged/`**: Merged model (base + LoRA adapters)
- **`SinLlama_merged_bf16/`**: Merged model in bfloat16 precision
- **`sinllama_compact_lora_r16_cpu/`**: Compact LoRA adapter (r=16)
- **`SinLlama_v01/`**: Version 1 checkpoint
- **`SinLlama3_QLoRA/`**: QLoRA training checkpoint

### 6. **Datasets**
- **`All-Text_8696658_147190824.normalized.txt`**: Large Sinhala text corpus
- **`model.tar.zst`**: Compressed model files
- **`dataset.tar.zst`**: Compressed dataset archives
- **`oasst2_en_chat.jsonl`**: OASST2 English conversational data
- **`oasst2_selected_si.jsonl`**: Selected data, translated to Sinhala
- **`oasst2_clustered.jsonl`**: Clustered conversational data

---

## Main Training Script: `train_lora_cpt.py`

### Overview
`train_lora_cpt.py` is the **primary continuous pre-training script** that fine-tunes Llama-3-8b on Sinhala text using LoRA adapters. It's optimized for efficiency with token packing, mixed precision training, and gradient accumulation.

### Key Features

#### 1. **Environment Configuration**
All parameters are configurable via environment variables:

```bash
# Model paths
MODEL_PATH="/workspace/model/SinLlama_merged_bf16"    # Base model
TXT_PATH="/workspace/data/All-Text_*.normalized.txt"  # Training text
OUT_DIR="/workspace/sinllama_lora_cpt"                # Output directory

# Training hyperparameters
SEQ_LEN=1024              # Sequence length
MICRO_BS=2                # Micro batch size (per GPU)
GRAD_ACC=4                # Gradient accumulation steps (effective BS = 2*4 = 8)
LR=1.5e-4                 # Learning rate
EPOCHS=1.0                # Number of epochs
WARMUP_RATIO=0.02         # 2% warmup

# LoRA configuration
LORA_R=32                 # LoRA rank
LORA_ALPHA=64             # LoRA alpha (scaling)
LORA_DROPOUT=0.05         # Dropout in LoRA

# Logging
LOG_STEPS=50              # Log every N steps
SAVE_STEPS=1000           # Save checkpoint every N steps
```

#### 2. **Data Pipeline**

```
Raw Text (Sinhala)
    ↓
Load via 'text' dataset format
    ↓
Shuffle (seed=42)
    ↓
Train/Val Split (98%/2%)
    ↓
Tokenization (with fast tokenizer)
    ↓
Token Packing (efficient sequence assembly)
    ↓
PyTorch DataLoader (ready for training)
```

**Token Packing**: Unlike simple padding, this efficiently packs multiple sequences into fixed-length chunks, avoiding wasted computation on padding tokens.

```python
def pack_tokens(examples):
    # Concatenate all tokens
    all_ids = []
    for ids in examples["input_ids"]:
        all_ids.extend(ids)
    
    # Split into fixed-length sequences
    total_len = (len(all_ids) // SEQ_LEN) * SEQ_LEN
    return {
        "input_ids": [
            all_ids[i:i+SEQ_LEN]
            for i in range(0, total_len, SEQ_LEN)
        ]
    }
```

#### 3. **LoRA Adapter Configuration**

```python
lora_cfg = LoraConfig(
    r=32,                          # Low rank
    lora_alpha=64,                 # Scaling (2x rank)
    lora_dropout=0.05,             # Regularization
    bias="none",                   # No bias adapters
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj",        # Query, Key projections
        "v_proj", "o_proj",        # Value, Output projections
        "gate_proj",               # Gate in MLP
        "up_proj", "down_proj"     # Up/Down in MLP
    ]
)
```

**Total trainable parameters**: ~12.6M (for r=32, 7 target modules)
**Original model**: 8B parameters
**Efficiency gain**: ~630x parameter reduction

#### 4. **Training Configuration**

- **Precision**: bfloat16 (mixed precision with TF32 matmul)
- **Optimizer**: AdamW Fused (fastest PyTorch implementation)
- **LR Schedule**: Cosine annealing with warmup
- **Evaluation**: Every 1000 steps on validation set
- **Metrics Tracked**:
  - Training loss & perplexity
  - Validation loss & perplexity
  - Learning rate schedule
  - Gradient norms

#### 5. **Perplexity Callback**

Automatically computes perplexity from loss:
$$\text{Perplexity} = e^{\text{loss}}$$

Used by W&B for monitoring training quality in real-time.

#### 6. **Weights & Biases Integration**

Automatically logs:
- Training curves (loss, perplexity)
- Model architecture and parameters
- Checkpoints for model versioning
- Hyperparameter configuration

```python
wandb.init(
    project="sinllama-cpt",
    name=f"lora_cpt_seq{SEQ_LEN}_bs{MICRO_BS}x{GRAD_ACC}_lr{LR}",
)
```

#### 7. **Output Artifacts**

After training completes:
```
/workspace/sinllama_lora_cpt/
├── adapters/               # LoRA weights (to merge with base model)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── tokenizer/              # Extended tokenizer
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── checkpoint-*/           # Checkpoints at save intervals
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── trainer_state.json
    └── training_args.bin
```

### Running `train_lora_cpt.py`

```bash
# With defaults
python finetune/train_lora_cpt.py

# Custom configuration
export MODEL_PATH="/path/to/base/model"
export TXT_PATH="/path/to/sinhala/text.txt"
export SEQ_LEN=2048
export MICRO_BS=4
export LR=2e-4
export LORA_R=64
python finetune/train_lora_cpt.py

# Or via environment file
source training_config.env
python finetune/train_lora_cpt.py
```

---

## Directory Structure

```
SInLlama-Backup/
├── finetune/                          # Training scripts
│   ├── train_lora_cpt.py             # ⭐ Main CPT + LoRA script
│   ├── qlora_finetune.py             # QLoRA variant
│   ├── qlora_finetune_cpu.py         # CPU-optimized
│   ├── qlora_finetune_hybrid.py      # CPU-GPU hybrid
│   └── CPU_GPU_STRATEGY_COMPARISON.md
│
├── data/                              # Data processing
│   ├── clustering.py                 # HDBSCAN clustering
│   ├── select_clusters.py            # Quality filtering
│   ├── translate_en_to_si.py         # Translation pipeline
│   ├── cluster_evaluation.jsonl      # Evaluated clusters
│   └── OASST2/                       # Dataset files
│
├── projects/                          # Utilities & inference
│   ├── merge_model.py                # Merge LoRA → base
│   ├── inference_bf16.py             # Fast inference
│   ├── eval_ppl.py                   # Perplexity evaluation
│   ├── prune_magnitude.py            # Magnitude pruning
│   └── masked_mlp_prune_and_eval.py  # Layer pruning
│
├── model_analysis/                    # Analysis & visualization
│   ├── wa_importance_visualization.py
│   ├── weights_distribution_visualization.py
│   ├── activation_analysis/
│   └── weight_analysis/
│
├── llama-3-8b/                        # Base model
│   ├── config.json
│   ├── model-*.safetensors
│   └── tokenizer.json
│
├── SinLlama_merged/                   # Final merged model (fp32)
├── SinLlama_merged_bf16/              # Final merged model (bf16)
├── SinLlama3_QLoRA/                   # QLoRA checkpoint
├── sinllama_compact_lora_r16_cpu/     # Compact adapter
│
├── All-Text_*.normalized.txt          # Training corpus
├── dataset.tar.zst                    # Compressed datasets
├── model.tar.zst                      # Compressed model
├── requirements.txt
├── run.sh
└── README.md                          # This file
```

---

## Setup & Installation

### 1. **Clone or navigate to project**
```bash
cd /ml/SInLlama-Backup
```

### 2. **Install dependencies**
```bash
pip install -r requirements.txt
```

**Key dependencies**:
- `transformers` - Hugging Face models
- `peft` - LoRA and QLoRA implementation
- `datasets` - Data loading and processing
- `torch` - PyTorch (with CUDA support)
- `bitsandbytes` - 4-bit quantization
- `wandb` - Experiment tracking
- `sentence-transformers` - Embeddings for clustering
- `hdbscan` - Density-based clustering

### 3. **Download or prepare data**

```bash
# Extract compressed datasets
tar -xf dataset.tar.zst
tar -xf model.tar.zst

# Or download OASST2 directly
# python data/extract_en.py  # Extract English
# pip install google-cloud-translate
# python data/translate_en_to_si.py  # Translate to Sinhala
```

### 4. **Set up W&B (optional but recommended)**
```bash
wandb login
# Follow prompts to connect your W&B account
```

---

## Usage

### **Option 1: Continuous Pre-training (CPT) with LoRA** ⭐

Primary use case - train on Sinhala text corpus:

```bash
export MODEL_PATH="./SinLlama_merged_bf16"
export TXT_PATH="./All-Text_8696658_147190824.normalized.txt"
export OUT_DIR="./sinllama_lora_cpt_output"
export SEQ_LEN=1024
export MICRO_BS=2
export GRAD_ACC=4
export LR=1.5e-4
export LORA_R=32
export EPOCHS=1

python finetune/train_lora_cpt.py
```

### **Option 2: Supervised Fine-tuning with QLoRA**

For instruction-following on conversational data:

```bash
python finetune/qlora_finetune.py
```

### **Option 3: Data Processing Pipeline**

1. **Cluster conversational data**:
   ```bash
   python data/clustering.py
   ```

2. **Select high-quality clusters**:
   ```bash
   python data/select_clusters.py
   ```

3. **Translate to Sinhala**:
   ```bash
   export GOOGLE_TRANSLATE_API_KEY="your-api-key"
   python data/translate_en_to_si.py
   ```

### **Option 4: Merge LoRA with Base Model**

After training, merge adapters back into base model:

```bash
python projects/merge_model_bfloat16.py
```

Output: `SinLlama_merged_bf16/` (deployment-ready)

### **Option 5: Run Inference**

```python
# Fast inference with bf16 precision
python projects/inference_bf16.py

# Or CPU-only inference
python projects/inference_cpu.py

# Or use merged model
python projects/inference_merge_model.py
```

### **Option 6: Evaluate Perplexity**

```bash
python projects/eval_ppl.py
```

---

## Other Important Files

### **`finetune/qlora_finetune.py`**
QLoRA variant with 4-bit quantization for memory-constrained training. Uses `SFTTrainer` from TRL library for supervised fine-tuning on instruction data.

### **`projects/merge_model.py`**
Merges LoRA adapters with base model using PEFT's `PeftModel.merge_and_unload()`. Essential for creating inference-ready models.

### **`projects/inference_bf16.py`**
High-performance inference using bfloat16 quantization with 4-bit NF4 loading, achieving ~2x speedup over fp32.

### **`projects/prune_magnitude.py`**
Removes low-magnitude weights to reduce model size. Works post-training to compress models without fine-tuning.

### **`projects/masked_mlp_prune_and_eval.py`**
Selective layer pruning - can zero-out entire MLP layers and re-evaluate perplexity to find pruning targets.

### **`model_analysis/wa_importance_visualization.py`**
Analyzes weight and activation importance to identify which layers/heads contribute most to model performance. Used for pruning decisions.

### **`data/clustering.py`**
Uses HDBSCAN for density-based clustering of embeddings:
1. Encodes text with `all-MiniLM-L6-v2`
2. Reduces dimensions with UMAP
3. Clusters with HDBSCAN
4. Outputs `oasst2_clustered.jsonl` with cluster labels

### **`data/translate_en_to_si.py`**
Batch translates Google's OASST2 dataset to Sinhala using Google Cloud Translation API. Preserves conversation structure and message roles.

### **`run.sh`**
Simple launcher script - execute with `bash run.sh` to run default training configuration.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./SinLlama_merged_bf16` | Path to base model |
| `TXT_PATH` | `./All-Text_*.txt` | Path to training text |
| `OUT_DIR` | `./sinllama_lora_cpt` | Output checkpoint directory |
| `SEQ_LEN` | `1024` | Sequence length (tokens) |
| `MICRO_BS` | `2` | Per-device batch size |
| `GRAD_ACC` | `4` | Gradient accumulation steps |
| `LR` | `1.5e-4` | Learning rate |
| `WARMUP_RATIO` | `0.02` | Warmup ratio (0-1) |
| `EPOCHS` | `1.0` | Number of epochs |
| `LORA_R` | `32` | LoRA rank |
| `LORA_ALPHA` | `64` | LoRA alpha (scaling factor) |
| `LORA_DROPOUT` | `0.05` | LoRA dropout probability |
| `LOG_STEPS` | `50` | Logging frequency |
| `SAVE_STEPS` | `1000` | Checkpoint save frequency |
| `WANDB_PROJECT` | `sinllama-cpt` | W&B project name |

---

## References

- **LoRA** (Low-Rank Adaptation): [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **QLoRA** (4-bit LoRA): [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **Llama-3** Model: [Meta AI](https://www.llama.com/)
- **OASST2** Dataset: [Open Assistant](https://github.com/LAION-AI/Open-Assistant)
- **PEFT Library**: [Hugging Face](https://github.com/huggingface/peft)
- **Token Packing**: Best practice for efficient LLM training
- **Continuous Pre-training (CPT)**: Additional training on domain-specific text

---