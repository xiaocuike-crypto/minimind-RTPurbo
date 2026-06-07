"""
Qwen3.5 RTPurbo 大海捞针 (Needle in a Haystack) 评测脚本
======================================================
在 2048 长度的干扰文本（Haystack）中，于不同深度（10%, 50%, 90%）插入特定的钥匙（Needle），
测试并对比原生全注意力（Teacher）与 RTPurbo 稀疏注意力（Student）模型的检索召回准确率。
"""

import os
import sys
import json
import math
import argparse
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, AutoModelForImageTextToText
from model.model_qwen_rtpurbo import convert_qwen_to_rtpurbo


def load_teacher(args, device):
    """加载原始全精度 Qwen3.5-4B 模型"""
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager"
    )
    return model.to(device).eval()


def load_student(args, device):
    """加载动态转换为 RTPurbo 后的学生模型并读入微调权重"""
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager"
    )
    with open(args.head_config, 'r', encoding='utf-8') as f:
        head_config = json.load(f)

    # 替换为 RTPurbo 稀疏结构
    model = convert_qwen_to_rtpurbo(
        model,
        head_config,
        index_dim=args.index_dim,
        local_window_size=args.local_window_size,
        retrieval_top_p=args.retrieval_top_p,
        sparse_attn=True
    )

    # 加载已微调完毕的权重
    if args.weight_path and os.path.exists(args.weight_path):
        w = torch.load(args.weight_path, map_location=device)
        missing, unexpected = model.load_state_dict(w, strict=False)
        print(f"✅ 加载 RTPurbo 权重: {args.weight_path}")
        print(f"  缺失键数: {len(missing)}, 多余键数: {len(unexpected)}")
    else:
        print("⚠️ 未检测到已训练权重，使用基线权重直接推理！")

    return model.to(device).eval()


def build_context(tokenizer, needle, depth_ratio, target_len=1900):
    """构建包含 Needle 的干扰文本 Context"""
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    
    # 干扰背景文本段落
    background_text = (
        "Large language models (LLMs) are deep learning algorithms that can recognize, summarize, translate, "
        "predict and generate text and other content based on knowledge gained from massive datasets. "
        "LLMs are pretrained on vast amounts of text data, enabling them to understand structural relationships. "
        "Attention mechanisms, specifically self-attention, allow the model to focus on different parts of the "
        "input sequence when generating tokens. However, the quadratic complexity of self-attention poses a "
        "significant challenge for long sequence processing. RTPurbo introduces a sparse attention mechanism that "
        "selects critical retrieval heads to run low-dimension projection index retrieval while keeping local heads "
        "in a sliding window. This preserves key long-range dependencies while saving KV cache memory. "
    )
    bg_ids = tokenizer.encode(background_text, add_special_tokens=False)
    
    # 填充到指定目标长度
    needed_tokens = target_len - len(needle_ids)
    num_reps = math.ceil(needed_tokens / len(bg_ids))
    full_bg_ids = (bg_ids * num_reps)[:needed_tokens]
    
    # 在指定深度比例处插入 Needle
    insert_pos = int(len(full_bg_ids) * depth_ratio)
    context_ids = full_bg_ids[:insert_pos] + needle_ids + full_bg_ids[insert_pos:]
    return tokenizer.decode(context_ids, skip_special_tokens=True)


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt, max_new_tokens=180):
    """带系统提示词与强引导前缀的贪婪解码生成，以防止 Qwen3.5 产生思考链截断"""
    system_prompt = "请直接、简短地回答用户的问题。严禁输出任何思考过程（思考链、Thinking Process 等），只输出最终答案。"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    template = template + "直接回答："
    input_ids = tokenizer(template, return_tensors="pt").input_ids.to(model.device)

    generated = input_ids
    for _ in range(max_new_tokens):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(generated)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    output_ids = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 RTPurbo Needle in a Haystack Test")
    parser.add_argument("--model_path", type=str, default="../model/Qwen3.5-4B")
    parser.add_argument("--weight_path", type=str, default="../out/rtpurbo_stage2_qwen.pth")
    parser.add_argument("--head_config", type=str, default="../qwen_head_config.json")
    parser.add_argument("--index_dim", type=int, default=16)
    parser.add_argument("--local_window_size", type=int, default=128)
    parser.add_argument("--retrieval_top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--teacher_device", type=str, default=None)
    parser.add_argument("--output", type=str, default="../out/needle_in_haystack_results.json",
                        help="评测结果保存路径")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 确定设备分流
    t_device = args.teacher_device
    if t_device is None:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            try:
                dev_idx = int(args.device.split(":")[-1])
                t_device = f"cuda:{(dev_idx + 1) % num_gpus}"
            except:
                t_device = args.device
        else:
            t_device = args.device

    print("=" * 60)
    print("Qwen3.5-4B 大海捞针 (Needle in a Haystack) 2048 长度测试")
    print("=" * 60)
    print(f"  Student 设备: {args.device} | Teacher 设备: {t_device}")

    print("\n[1/3] 加载模型中...")
    teacher = load_teacher(args, t_device)
    student = load_student(args, args.device)
    print("  模型加载完成！")

    needle = "The password to the secure server is: 'antigravity-999'."
    question = "What is the password to the secure server? Answer the question directly with the password itself, do not output any thinking process."
    
    print("\n[2/3] 开始测试不同深度的召回率...")
    depths = [0.1, 0.5, 0.9]  # 分别对应开头、中间、末尾
    
    results = []
    for depth in depths:
        print(f"\n👉 测试深度: {depth*100:.0f}% 位置 ------------------------")
        
        # 1. 构建长上下文
        context = build_context(tokenizer, needle, depth, target_len=1900)
        prompt = f"Long Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
        
        # 验证输入长度是否满足约 2000 token
        input_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        print(f"   输入 Prompt 实际 token 长度: {input_len}")

        # 2. 教师端推理
        print("   教师端 (Full Attention) 推理中...")
        t_res = greedy_generate(teacher, tokenizer, prompt)
        print(f"   [Teacher 结果]: {t_res.strip()}")

        # 3. 学生端推理
        print("   学生端 (RTPurbo) 推理中...")
        s_res = greedy_generate(student, tokenizer, prompt)
        print(f"   [Student 结果]: {s_res.strip()}")

        # 验证是否成功召回 "antigravity-999"
        t_ok = "antigravity-999" in t_res
        s_ok = "antigravity-999" in s_res
        
        results.append({
            "depth": depth,
            "input_len": input_len,
            "teacher_response": t_res.strip(),
            "student_response": s_res.strip(),
            "teacher_success": t_ok,
            "student_success": s_ok
        })

    print("\n[3/3] 汇总测试结果:")
    print("-" * 60)
    for res in results:
        t_status = "✅ 成功" if res["teacher_success"] else "❌ 失败"
        s_status = "✅ 成功" if res["student_success"] else "❌ 失败"
        print(f"深度 {res['depth']*100:.0f}% (长度 {res['input_len']}): "
              f"Teacher -> {t_status} | Student -> {s_status}")
    print("-" * 60)

    if args.output:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({"results": results}, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 大海捞针评测结果成功写入：{args.output}")
        except Exception as e:
            print(f"\n⚠️ 保存大海捞针评测结果失败: {e}")


if __name__ == "__main__":
    main()
