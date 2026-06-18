from safetensors.torch import load_file

tensors = load_file("SinLlama_v01/adapter_model.safetensors")
print(len(tensors))
print(list(tensors.keys())[:20])