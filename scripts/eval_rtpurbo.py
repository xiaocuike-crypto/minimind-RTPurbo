"""
RTPurbo 评估对比脚本
====================
对比 全注意力模型 vs RTPurbo 稀疏注意力模型 的：
  1. 生成质量（对话测试）
  2. 注意力稀疏度统计
  3. 推理速度（单次前向 latency）
  4. 内存占用

用法:
  python scripts/eval_rtpurbo.py
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_rtpurbo import RTPurboConfig, RTPurboForCausalLM, load_head_config
from transformers import AutoTokenizer


def load_teacher(args, device):
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        flash_attn=True
    )
    model = MiniMindForCausalLM(config)
    moe = '_moe' if config.use_moe else ''
    w = torch.load(f'{args.model_dir}/{args.base_weight}_{args.hidden_size}{moe}.pth', map_location=device)
    model.load_state_dict(w, strict=False)
    return model.to(device).eval()


def load_student(args, device):
    head_config = load_head_config(args.head_config)
    config = RTPurboConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        sparse_attn=True,
        index_dim=args.index_dim,
        local_window_size=args.local_window_size,
        retrieval_top_p=args.retrieval_top_p,
    )
    model = RTPurboForCausalLM(config, head_config)
    stage2_path = f'{args.model_dir}/rtpurbo_stage2_{args.hidden_size}.pth'
    if os.path.exists(stage2_path):
        w = torch.load(stage2_path, map_location=device)
        model.load_state_dict(w, strict=False)
        print(f"加载 Stage2 权重: {stage2_path}")
    else:
        stage1_path = f'{args.model_dir}/rtpurbo_stage1_{args.hidden_size}.pth'
        if os.path.exists(stage1_path):
            w = torch.load(stage1_path, map_location=device)
            model.load_state_dict(w, strict=False)
            print(f"加载 Stage1 权重: {stage1_path}")
    return model.to(device).eval()


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_p=0.9):
    """简单的自回归生成"""
    messages = [{"role": "user", "content": prompt}]
    template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(template, return_tensors="pt").input_ids.to(model.lm_head.weight.device)

    generated = input_ids
    for _ in range(max_new_tokens):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(generated)
        logits = outputs.logits[:, -1, :] / temperature

        # Top-p sampling
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
def measure_latency(model, input_ids, n_runs=20):
    """测量推理延迟"""
    device = input_ids.device
    # Warmup
    for _ in range(5):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            model(input_ids)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            model(input_ids)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        'mean_ms': sum(times) / len(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000
    }


def main():
    parser = argparse.ArgumentParser(description="RTPurbo Evaluation")
    parser.add_argument("--model_dir", type=str, default="../out")
    parser.add_argument("--base_weight", type=str, default="full_sft")
    parser.add_argument("--head_config", type=str, default="../head_config.json")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--index_dim", type=int, default=16)
    parser.add_argument("--local_window_size", type=int, default=128)
    parser.add_argument("--retrieval_top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="../eval_results.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('../model')

    print("=" * 60)
    print("RTPurbo 评估对比")
    print("=" * 60)

    # 加载模型
    print("\n[1/4] 加载模型...")
    teacher = load_teacher(args, args.device)
    student = load_student(args, args.device)
    print(f"  教师模型: {sum(p.numel() for p in teacher.parameters()) / 1e6:.2f}M")
    print(f"  学生模型: {sum(p.numel() for p in student.parameters()) / 1e6:.2f}M")

    # 稀疏度统计
    print("\n[2/4] 稀疏度统计...")
    stats = student.get_sparsity_stats()
    total_heads = 0
    retrieval_heads = 0
    for layer_info in stats.values():
        total_heads += len(layer_info['retrieval_heads']) + len(layer_info['local_heads'])
        retrieval_heads += len(layer_info['retrieval_heads'])
    print(f"  总 heads: {total_heads}")
    print(f"  Retrieval heads: {retrieval_heads} ({retrieval_heads/total_heads*100:.1f}%)")
    print(f"  Local heads: {total_heads - retrieval_heads} ({(1-retrieval_heads/total_heads)*100:.1f}%)")
    print(f"  窗口大小: {stats['layer_0']['window_size']}")
    print(f"  Top-p: {stats['layer_0']['top_p']}")

    # 对话测试
    print("\n[3/4] 对话生成对比...")
    test_prompts = [
        "请介绍一下人工智能的发展历史。",
        "写一首关于春天的诗。",
        "解释什么是深度学习。",
        "1+1等于多少？",
        "如何学习编程？",
    ]

    results = {'prompts': []}
    for prompt in test_prompts:
        print(f"\n  Q: {prompt}")
        teacher_resp = generate_text(teacher, tokenizer, prompt, max_new_tokens=100)
        student_resp = generate_text(student, tokenizer, prompt, max_new_tokens=100)
        print(f"  全注意力: {teacher_resp[:100]}...")
        print(f"  RTPurbo:  {student_resp[:100]}...")
        results['prompts'].append({
            'prompt': prompt,
            'teacher': teacher_resp,
            'student': student_resp
        })

    # 推理速度对比
    print("\n[4/4] 推理速度对比...")
    for seq_len in [128, 256, 512]:
        input_ids = torch.randint(0, 1000, (1, seq_len), device=args.device)
        t_lat = measure_latency(teacher, input_ids)
        s_lat = measure_latency(student, input_ids)
        speedup = t_lat['mean_ms'] / max(s_lat['mean_ms'], 0.001)
        print(f"  seq_len={seq_len}: "
              f"全注意力={t_lat['mean_ms']:.2f}ms, "
              f"RTPurbo={s_lat['mean_ms']:.2f}ms, "
              f"加速比={speedup:.2f}x")
        results[f'latency_seq{seq_len}'] = {
            'teacher_ms': t_lat['mean_ms'],
            'student_ms': s_lat['mean_ms'],
            'speedup': speedup
        }

    # 保存结果
    results['sparsity'] = {
        'total_heads': total_heads,
        'retrieval_heads': retrieval_heads,
        'local_heads': total_heads - retrieval_heads
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 评估结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
