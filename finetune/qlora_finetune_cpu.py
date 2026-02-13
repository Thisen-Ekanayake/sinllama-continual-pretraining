import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import gc
import multiprocessing
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set CPU threads to use all cores
NUM_CORES = multiprocessing.cpu_count()
torch.set_num_threads(NUM_CORES)
os.environ['OMP_NUM_THREADS'] = str(NUM_CORES)
os.environ['MKL_NUM_THREADS'] = str(NUM_CORES)

# Clear memory
gc.collect()

# ------------------------
# 1) Config
# ------------------------
MODEL_ID = "SinLlama_merged_bf16"
DATA_FILE = "data/oasst2_selected_si.jsonl"
OUTPUT_DIR = "./SinLlama3_QLoRA"

print("=" * 70)
print("CPU Training Configuration")
print("=" * 70)
print(f"Model: {MODEL_ID}")
print(f"Data: {DATA_FILE}")
print(f"Output: {OUTPUT_DIR}")
print(f"CPU Cores: {NUM_CORES}")
print(f"PyTorch Threads: {torch.get_num_threads()}")
print("=" * 70)

# ------------------------
# 2) Load tokenizer
# ------------------------
print("\n[1/7] Loading tokenizer...")
with tqdm(total=1, desc="📥 Tokenizer", ncols=100, bar_format='{l_bar}{bar}| {elapsed}') as pbar:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    pbar.update(1)
print("✓ Tokenizer loaded")

# ------------------------
# 3) Manual Chat Template
# ------------------------
print("\n[2/7] Setting up chat template...")
with tqdm(total=1, desc="💬 Chat Template", ncols=100, bar_format='{l_bar}{bar}| {elapsed}') as pbar:
    chat_template = r"""
<|begin_of_text|>
{% for message in messages %}
<|start_header_id|>{{ message.role }}<|end_header_id|>

{{ message.content }}
<|eot_id|>
{% endfor %}
{% if add_generation_prompt %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}
"""
    tokenizer.chat_template = chat_template
    pbar.update(1)
print("✓ Chat template configured")

# ------------------------
# 4) Load model for CPU
# ------------------------
print("\n[3/7] Loading model for CPU...")
print("⚠️  This may take several minutes on CPU...")
with tqdm(total=1, desc="🤖 Model Loading", ncols=100, bar_format='{l_bar}{bar}| {elapsed}') as pbar:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu",  # Force CPU
        low_cpu_mem_usage=True,
    )
    pbar.update(1)

# Model config
model.config.use_cache = False
model.config.pretraining_tp = 1
print("✓ Model loaded on CPU")

# ------------------------
# 5) LoRA adapter config
# ------------------------
print("\n[4/7] Configuring LoRA adapter...")
with tqdm(total=1, desc="🔧 LoRA Config", ncols=100, bar_format='{l_bar}{bar}| {elapsed}') as pbar:
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    pbar.update(1)
print("✓ LoRA configured")

# ------------------------
# 6) Load & format dataset
# ------------------------
print("\n[5/7] Loading dataset...")
with tqdm(total=1, desc="📊 Dataset Load", ncols=100, bar_format='{l_bar}{bar}| {elapsed}') as pbar:
    dataset = load_dataset("json", data_files=DATA_FILE)["train"]
    pbar.update(1)

print(f"✓ Dataset loaded: {len(dataset)} examples")

def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

print("\n[6/7] Formatting dataset...")
# Use all CPU cores for dataset processing
with tqdm(total=len(dataset), desc="🔄 Formatting", ncols=100, bar_format='{l_bar}{bar}| [{elapsed}<{remaining}]') as pbar:
    dataset = dataset.map(
        format_example, 
        remove_columns=dataset.column_names,
        num_proc=NUM_CORES,  # Use all CPU cores
        desc="Processing examples"
    )
    pbar.update(len(dataset))
print("✓ Dataset formatted")

# ------------------------
# 7) Training args (CPU optimized)
# ------------------------
print("\n[7/7] Setting up training configuration...")
with tqdm(total=1, desc="⚙️  Training Setup", ncols=100, bar_format='{l_bar}{bar}| {elapsed}') as pbar:
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        fp16=False,  # Disable fp16 for CPU
        bf16=False,  # Disable bf16 for CPU
        learning_rate=2e-4,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        logging_steps=20,
        save_strategy="epoch",
        optim="adamw_torch",  # Use standard AdamW for CPU
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        dataloader_num_workers=min(4, NUM_CORES),  # Use multiple workers
        report_to="none",
        disable_tqdm=False,  # Enable tqdm progress bars
    )
    pbar.update(1)
print("✓ Training configuration ready")

# ------------------------
# 8) Trainer
# ------------------------
print("\n" + "=" * 70)
print("Preparing trainer...")
print("=" * 70)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
    args=training_args,
)

# ------------------------
# 9) Clear memory before training
# ------------------------
gc.collect()

# ------------------------
# 10) Train
# ------------------------
print("\n" + "=" * 70)
print("🚀 Starting training...")
print("=" * 70)
print(f"Total examples: {len(dataset)}")
print(f"Epochs: 3")
print(f"Steps per epoch: ~{len(dataset) // training_args.gradient_accumulation_steps}")
print(f"Total steps: ~{3 * len(dataset) // training_args.gradient_accumulation_steps}")
print("=" * 70)
print("\n⏱️  Training progress (this will take a while on CPU):\n")

trainer.train()

# ------------------------
# 11) Save
# ------------------------
print("\n" + "=" * 70)
print("💾 Saving LoRA adapter...")
with tqdm(total=1, desc="Saving Model", ncols=100, bar_format='{l_bar}{bar}| {elapsed}') as pbar:
    trainer.save_model(OUTPUT_DIR)
    pbar.update(1)

print("=" * 70)
print("✅ Training complete!")
print(f"📁 Model saved to: {OUTPUT_DIR}")
print("=" * 70)