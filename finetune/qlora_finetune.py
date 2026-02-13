import torch
import gc
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()

# ------------------------
# 1) Config
# ------------------------
MODEL_ID = "SinLlama_merged_bf16"
DATA_FILE = "data/oasst2_selected_si.jsonl"
OUTPUT_DIR = "./SinLlama3_QLoRA"

# ------------------------
# 2) Load tokenizer
# ------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Recommended for training

# ------------------------
# 3) Manual Chat Template
# ------------------------
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

# ------------------------
# 4) 4-bit config for QLoRA
# ------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ------------------------
# 5) Load model (4-bit)
# ------------------------
print("Loading 4-bit model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "6GB"},  # Limit memory usage
    low_cpu_mem_usage=True,
)

# Don't manually prepare - let SFTTrainer do it
model.config.use_cache = False
model.config.pretraining_tp = 1

# ------------------------
# 6) LoRA adapter config (minimal)
# ------------------------
lora_config = LoraConfig(
    r=16,  # Reduced
    lora_alpha=32,  # Reduced
    target_modules=["q_proj", "v_proj"],  # Only 2 modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ------------------------
# 7) Load & format dataset
# ------------------------
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_FILE)["train"]

def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

print("Formatting dataset...")
dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# ------------------------
# 8) Training args (aggressive memory saving)
# ------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    bf16=True,
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    logging_steps=20,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    ddp_find_unused_parameters=False,
    report_to="none",  # Disable reporting to save memory
)

# ------------------------
# 9) Trainer
# ------------------------
print("Preparing trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
    args=training_args,
    # Removed max_seq_length - will handle via tokenizer or formatting
)

# ------------------------
# 10) Clear cache before training
# ------------------------
torch.cuda.empty_cache()
gc.collect()

# ------------------------
# 11) Train
# ------------------------
print("Starting training...")
trainer.train()

# ------------------------
# 12) Save
# ------------------------
print("Saving LoRA adapter...")
trainer.save_model(OUTPUT_DIR)
print("Done.")