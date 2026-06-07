import torch
# 猴子补丁：强行注入缺失的 float8_e8m0fnu 属性以绕过 transformers 5.10 在 torch 2.4 下的导入问题
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e4m3fn", torch.float32)

import transformers
print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)

from transformers import AutoTokenizer, AutoModelForImageTextToText

model_path = "/mnt/d/minimind-RTPurbo/model/Qwen3.5-4B"
print("正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Tokenizer 加载成功！")

print("正在加载 Qwen3.5 模型 (使用 AutoModelForImageTextToText)...")
try:
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    print("✅ Qwen3.5 模型成功加载完毕！")
    print("模型结构:", type(model))
except Exception as e:
    print("❌ 模型加载失败:", e)
