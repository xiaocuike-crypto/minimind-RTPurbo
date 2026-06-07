import torch
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e4m3fn", torch.float32)

from transformers import AutoModelForImageTextToText

model_path = "/mnt/d/minimind-RTPurbo/model/Qwen3.5-4B"
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

layers = model.model.language_model.layers
print("Layer 0 (linear_attention) class:", type(layers[0]))
print("Layer 0 (linear_attention) named children:")
for name, child in layers[0].named_children():
    print(f"  - {name}: {type(child)}")

print("\nLayer 3 (full_attention) class:", type(layers[3]))
print("Layer 3 (full_attention) named children:")
for name, child in layers[3].named_children():
    print(f"  - {name}: {type(child)}")
