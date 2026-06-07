import torch
import json
import sys
import os
import time

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
prompt_with_instruction = f"请直接、简短地回答用户的问题。严禁输出任何思考过程（思考链、Thinking Process 等），只输出最终答案。\n\n问题：{prompt}"
messages = [
    {"role": "user", "content": prompt_with_instruction}
]
template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
template = template + "直接回答："
input_ids = tokenizer(template, return_tensors="pt").input_ids.to("cuda:2")

# 测试1：不带 KV Cache 的生成
print("\n--- Test 1: Without KV Cache (No Cache) ---")
start_time = time.time()
generated_no_cache = input_ids
for i in range(50):
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = student(generated_no_cache)
    logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    generated_no_cache = torch.cat([generated_no_cache, next_token], dim=-1)
    if next_token.item() == tokenizer.eos_token_id:
        break
time_no_cache = time.time() - start_time
output_no_cache = tokenizer.decode(generated_no_cache[0, input_ids.shape[1]:], skip_special_tokens=True)
print(f"Time taken: {time_no_cache:.4f}s")
print(f"Tokens generated: {i + 1}")
print(f"Output: {output_no_cache}")

# 测试2：带 KV Cache 的生成
print("\n--- Test 2: With KV Cache ---")
start_time = time.time()
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = student(input_ids, use_cache=True)
logits = outputs.logits[:, -1, :]
next_token = torch.argmax(logits, dim=-1, keepdim=True)
past_key_values = outputs.past_key_values

generated_tokens = [next_token.item()]
for i in range(49):
    if next_token.item() == tokenizer.eos_token_id:
        break
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = student(next_token, past_key_values=past_key_values, use_cache=True)
    logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    generated_tokens.append(next_token.item())
time_cache = time.time() - start_time
output_cache = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(f"Time taken: {time_cache:.4f}s")
print(f"Tokens generated: {len(generated_tokens)}")
print(f"Output: {output_cache}")
