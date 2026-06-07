import torch
import json
import sys
import os
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e4m3fn", torch.float32)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer, AutoModelForImageTextToText
from model.model_qwen_rtpurbo import convert_qwen_to_rtpurbo

model_path = "/mnt/d/minimind-RTPurbo/model/Qwen3.5-4B"
head_config_path = "/mnt/d/minimind-RTPurbo/qwen_head_config_2048.json"
weight_path = "/mnt/d/out/rtpurbo_stage2_qwen_2048.pth"

print("Loading tokenizer and config...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
with open(head_config_path, 'r', encoding='utf-8') as f:
    head_config = json.load(f)

print("Loading Teacher model on cuda:3...")
teacher = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="eager")
teacher = teacher.to("cuda:3").eval()

print("Loading Student model on cuda:2...")
student = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="eager")
student = convert_qwen_to_rtpurbo(
    student,
    head_config,
    index_dim=16,
    local_window_size=128,
    retrieval_top_p=0.9,
    sparse_attn=True
)
w = torch.load(weight_path, map_location="cuda:2")
student.load_state_dict(w, strict=False)
student = student.to("cuda:2").eval()

prompt = "地球上海洋和陆地的比例是多少？哪个大洋的面积最大？"
messages = [{"role": "user", "content": prompt}]
template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 1. 教师端推理
input_ids_t = tokenizer(template, return_tensors="pt").input_ids.to("cuda:3")
generated_t = input_ids_t
print("Start autogressive generation for Teacher (cuda:3)...")
for i in range(10):
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = teacher(generated_t)
    logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    generated_t = torch.cat([generated_t, next_token], dim=-1)
    print(f"Teacher Step {i} token_id: {next_token.item()}")

# 2. 学生端推理
input_ids_s = tokenizer(template, return_tensors="pt").input_ids.to("cuda:2")
generated_s = input_ids_s
print("Start autogressive generation for Student (cuda:2)...")
for i in range(10):
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = student(generated_s)
    logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    generated_s = torch.cat([generated_s, next_token], dim=-1)
    print(f"Student Step {i} token_id: {next_token.item()}")

print("Double-device autogressive generation completed successfully!")
