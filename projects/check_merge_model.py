from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./SinLlama_merged")
tokenizer = AutoTokenizer.from_pretrained("./SinLlama_merged")

print(model.get_input_embeddings().weight.shape)
print(model.lm_head.weight.shape)