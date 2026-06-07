"""
Qwen3.5 RTPurbo 评估对比脚本
============================
对比 原生全注意力 Qwen3.5-4B vs RTPurbo 稀疏注意力 Qwen3.5-4B 的：
  1. 生成质量（长文本及长问答测试）
  2. 注意力稀疏度统计
  3. 推理速度（前向传播 Latency）
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, AutoModelForImageTextToText
from model.model_qwen_rtpurbo import convert_qwen_to_rtpurbo, get_sparsity_stats


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
        print("⚠️ 未加载已训练权重，使用基线权重直接推理！")

    return model.to(device).eval()


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7, top_p=0.9):
    """自回归文本生成"""
    messages = [{"role": "user", "content": prompt}]
    template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(template, return_tensors="pt").input_ids.to(model.device)

    generated = input_ids
    for _ in range(max_new_tokens):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(generated)
        logits = outputs.logits[:, -1, :] / temperature

        # Top-p 采样
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[sorted_mask] = float('-inf')
        probs = F.softmax(sorted_logits, dim=-1)
        next_token_sorted = torch.multinomial(probs, 1)
        next_token = sorted_indices.gather(-1, next_token_sorted)

        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    output_ids = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


@torch.no_grad()
def measure_latency(model, input_ids, n_runs=15):
    """测量前向传播平均延迟"""
    # Warmup
    for _ in range(3):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            model(input_ids)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            model(input_ids)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        'mean_ms': sum(times) / len(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 RTPurbo Evaluation")
    parser.add_argument("--model_path", type=str, default="../model/Qwen3.5-4B")
    parser.add_argument("--weight_path", type=str, default="../out/rtpurbo_stage2_qwen.pth")
    parser.add_argument("--head_config", type=str, default="../qwen_head_config.json")
    parser.add_argument("--index_dim", type=int, default=16)
    parser.add_argument("--local_window_size", type=int, default=128)
    parser.add_argument("--retrieval_top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="../qwen_eval_results.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("=" * 60)
    print("Qwen3.5-4B RTPurbo 评估对比")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/4] 加载模型中...")
    teacher = load_teacher(args, args.device)
    student = load_student(args, args.device)
    print("  模型加载成功！")

    # 2. 稀疏统计
    print("\n[2/4] 稀疏度与结构统计...")
    stats = get_sparsity_stats(student)
    total_heads = 0
    retrieval_heads = 0
    for layer_info in stats.values():
        total_heads += len(layer_info['retrieval_heads']) + len(layer_info['local_heads'])
        retrieval_heads += len(layer_info['retrieval_heads'])
    
    print(f"  稀疏标定层数: {len(stats)} 层")
    print(f"  总 Attention Heads: {total_heads}")
    print(f"  Retrieval Heads: {retrieval_heads} ({retrieval_heads/total_heads*100:.1f}%)")
    print(f"  Local Heads: {total_heads - retrieval_heads} ({(1-retrieval_heads/total_heads)*100:.1f}%)")
    
    layer_0_key = list(stats.keys())[0]
    print(f"  Local 滑动窗口: {stats[layer_0_key]['window_size']}")
    print(f"  检索 Top-p: {stats[layer_0_key]['top_p']}")

    # 3. 对话与长文本生成测试
    print("\n[3/4] 文本生成生成对比...")
    test_prompts = [
        "你好，请问你是谁？请简短地自我介绍一下。",
        "写一段关于未来科技发展趋势的简短评论。",
        "为什么大语言模型在长文本处理上往往面临算力和显存的巨大瓶颈？有什么优化的思路吗？",
    ]

    results = {'prompts': []}
    for prompt in test_prompts:
        print(f"\n  问: {prompt}")
        teacher_resp = generate_text(teacher, tokenizer, prompt, max_new_tokens=100)
        student_resp = generate_text(student, tokenizer, prompt, max_new_tokens=100)
        print(f"  [Teacher  全精度]: {teacher_resp}")
        print(f"  [RTPurbo  学生端]: {student_resp}")
        results['prompts'].append({
            'prompt': prompt,
            'teacher': teacher_resp,
            'student': student_resp
        })

    # 4. 推理前向 Latency 测速
    print("\n[4/4] 前向延迟对比...")
    # 对比 256, 512, 1024, 2048 Token 长度下的 Latency
    for seq_len in [256, 512, 1024, 2048]:
        input_ids = torch.randint(0, 200000, (1, seq_len), device=args.device)
        t_lat = measure_latency(teacher, input_ids)
        s_lat = measure_latency(student, input_ids)
        speedup = t_lat['mean_ms'] / max(s_lat['mean_ms'], 0.001)
        print(f"  序列长度 = {seq_len}: "
              f"全注意力 = {t_lat['mean_ms']:.2f}ms, "
              f"RTPurbo = {s_lat['mean_ms']:.2f}ms, "
              f"加速比 = {speedup:.2f}x")
        results[f'latency_seq{seq_len}'] = {
            'teacher_ms': t_lat['mean_ms'],
            'student_ms': s_lat['mean_ms'],
            'speedup': speedup
        }

    results['sparsity'] = {
        'total_heads': total_heads,
        'retrieval_heads': retrieval_heads,
        'local_heads': total_heads - retrieval_heads
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 评估结果成功写入: {args.output}")


if __name__ == "__main__":
    main()
