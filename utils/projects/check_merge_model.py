from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./SinLlama_merged_bf16")
tokenizer = AutoTokenizer.from_pretrained("./SinLlama_merged_bf16")

print(model.get_input_embeddings().weight.shape)
print(model.lm_head.weight.shape)