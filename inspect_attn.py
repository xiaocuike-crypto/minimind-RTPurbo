import inspect
import torch
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e4m3fn", torch.float32)

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

print("Qwen3_5Attention full source code:")
try:
    src = inspect.getsource(Qwen3_5Attention)
    print(src)
except Exception as e:
    print("Error getting source:", e)
