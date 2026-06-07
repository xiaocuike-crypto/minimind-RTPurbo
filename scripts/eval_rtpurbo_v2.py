"""
RTPurbo 增强版评估对比
======================
对比全注意力模型 vs RTPurbo 稀疏注意力模型：
  1. Perplexity (PPL) 对比 — 量化精度差异
  2. Token 一致率 — 两模型 top-1 预测的吻合度
  3. 生成质量 — 更多样化的测试 prompt
  4. 推理速度
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_rtpurbo import RTPurboConfig, RTPurboForCausalLM, load_head_config
from transformers import AutoTokenizer
from datasets import load_dataset


def load_teacher(args, device):
    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, flash_attn=True)
    model = MiniMindForCausalLM(config)
    moe = '_moe' if config.use_moe else ''
    w = torch.load(f'{args.model_dir}/{args.base_weight}_{args.hidden_size}{moe}.pth', map_location=device)
    model.load_state_dict(w, strict=False)
    return model.to(device).eval()


def load_student(args, device):
    head_config = load_head_config(args.head_config)
    config = RTPurboConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                           sparse_attn=True, index_dim=args.index_dim,
                           local_window_size=args.local_window_size, retrieval_top_p=args.retrieval_top_p)
    model = RTPurboForCausalLM(config, head_config)
    stage2_path = f'{args.model_dir}/rtpurbo_stage2_{args.hidden_size}.pth'
    if os.path.exists(stage2_path):
        w = torch.load(stage2_path, map_location=device)
        model.load_state_dict(w, strict=False)
        print(f"  加载 Stage2 权重: {stage2_path}")
    return model.to(device).eval()


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts, max_len=512, device='cuda'):
    """计算模型在给定文本上的 Perplexity"""
    total_loss = 0.0
    total_tokens = 0
    for text in tqdm(texts, desc="计算PPL"):
        tokens = tokenizer(text, return_tensors='pt', max_length=max_len, truncation=True)
        input_ids = tokens['input_ids'].to(device)
        if input_ids.shape[1] < 10:
            continue
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :].float()
        labels = input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += labels.numel()
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = np.exp(avg_loss)
    return ppl, avg_loss, total_tokens


@torch.no_grad()
def compute_token_agreement(teacher, student, tokenizer, texts, max_len=512, device='cuda'):
    """计算两个模型 top-1 token 预测的一致率"""
    total_agree = 0
    total_tokens = 0
    top5_agree = 0
    for text in tqdm(texts, desc="Token一致率"):
        tokens = tokenizer(text, return_tensors='pt', max_length=max_len, truncation=True)
        input_ids = tokens['input_ids'].to(device)
        if input_ids.shape[1] < 10:
            continue
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            t_logits = teacher(input_ids).logits[:, :-1, :].float()
            s_logits = student(input_ids).logits[:, :-1, :].float()
        t_top1 = t_logits.argmax(dim=-1)
        s_top1 = s_logits.argmax(dim=-1)
        total_agree += (t_top1 == s_top1).sum().item()
        # Top-5 一致率
        t_top5 = torch.topk(t_logits, 5, dim=-1).indices
        s_top1_expanded = s_top1.unsqueeze(-1)
        top5_agree += (t_top5 == s_top1_expanded).any(dim=-1).sum().item()
        total_tokens += t_top1.numel()
    return {
        'top1_agreement': total_agree / max(total_tokens, 1),
        'top5_agreement': top5_agree / max(total_tokens, 1),
        'total_tokens': total_tokens
    }


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7, top_p=0.85):
    messages = [{"role": "user", "content": prompt}]
    template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(template, return_tensors="pt").input_ids.to(model.lm_head.weight.device)
    generated = input_ids
    for _ in range(max_new_tokens):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(generated)
        logits = outputs.logits[:, -1, :] / temperature
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
    return {'mean_ms': sum(times)/len(times)*1000, 'min_ms': min(times)*1000, 'max_ms': max(times)*1000}


def main():
    parser = argparse.ArgumentParser(description="RTPurbo Enhanced Evaluation")
    parser.add_argument("--model_dir", type=str, default="../out")
    parser.add_argument("--base_weight", type=str, default="full_sft")
    parser.add_argument("--head_config", type=str, default="../head_config.json")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--index_dim", type=int, default=16)
    parser.add_argument("--local_window_size", type=int, default=128)
    parser.add_argument("--retrieval_top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_ppl_samples", type=int, default=200)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--output", type=str, default="../eval_results_10k.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('../model')
    results = {}

    print("=" * 70)
    print("  RTPurbo 增强版评估对比")
    print("=" * 70)

    # ---- 1. 加载模型 ----
    print("\n[1/5] 加载模型...")
    teacher = load_teacher(args, args.device)
    student = load_student(args, args.device)
    t_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    s_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"  教师模型: {t_params:.2f}M")
    print(f"  学生模型: {s_params:.2f}M (多 {(s_params-t_params)*1e6:.0f} params = index_proj)")

    # ---- 2. Perplexity ----
    print(f"\n[2/5] Perplexity 对比（{args.num_ppl_samples} 条文本）...")
    dataset = load_dataset('json', data_files=args.data_path, split='train')
    indices = np.random.RandomState(123).permutation(len(dataset))[:args.num_ppl_samples]
    eval_texts = [str(dataset[int(i)]['text']) for i in indices]

    t_ppl, t_loss, t_tokens = compute_perplexity(teacher, tokenizer, eval_texts, device=args.device)
    s_ppl, s_loss, s_tokens = compute_perplexity(student, tokenizer, eval_texts, device=args.device)
    ppl_gap = (s_ppl - t_ppl) / t_ppl * 100

    print(f"\n  {'指标':<20} {'全注意力':>12} {'RTPurbo':>12} {'差距':>10}")
    print(f"  {'─'*54}")
    print(f"  {'Perplexity':<20} {t_ppl:>12.2f} {s_ppl:>12.2f} {ppl_gap:>+9.2f}%")
    print(f"  {'Avg CE Loss':<20} {t_loss:>12.4f} {s_loss:>12.4f}")
    print(f"  {'评估 Tokens':<20} {t_tokens:>12,d} {s_tokens:>12,d}")
    results['perplexity'] = {'teacher': t_ppl, 'student': s_ppl, 'gap_pct': ppl_gap}

    # ---- 3. Token 一致率 ----
    print(f"\n[3/5] Token 预测一致率...")
    agreement = compute_token_agreement(teacher, student, tokenizer, eval_texts[:100], device=args.device)
    print(f"  Top-1 一致率: {agreement['top1_agreement']*100:.2f}%")
    print(f"  Top-5 一致率: {agreement['top5_agreement']*100:.2f}%")
    print(f"  评估 Tokens:  {agreement['total_tokens']:,d}")
    results['token_agreement'] = agreement

    # ---- 4. 生成质量 ----
    print("\n[4/5] 生成质量对比...")
    test_prompts = [
        "请介绍一下人工智能的发展历史。",
        "写一首关于春天的诗。",
        "解释什么是深度学习。",
        "如何学习编程？",
        "请解释量子计算的基本原理。",
        "中国有哪些著名的历史人物？",
        "什么是机器学习中的过拟合？如何解决？",
        "请写一段关于环境保护的文章。",
    ]

    gen_results = []
    for prompt in test_prompts:
        print(f"\n  Q: {prompt}")
        t_resp = generate_text(teacher, tokenizer, prompt, max_new_tokens=150)
        s_resp = generate_text(student, tokenizer, prompt, max_new_tokens=150)
        print(f"  [全注意力] {t_resp[:120]}...")
        print(f"  [RTPurbo ] {s_resp[:120]}...")
        gen_results.append({'prompt': prompt, 'teacher': t_resp, 'student': s_resp})
    results['generation'] = gen_results

    # ---- 5. 推理速度 ----
    print("\n[5/5] 推理速度对比...")
    speed_results = {}
    for seq_len in [128, 256, 512]:
        input_ids = torch.randint(0, 1000, (1, seq_len), device=args.device)
        t_lat = measure_latency(teacher, input_ids)
        s_lat = measure_latency(student, input_ids)
        ratio = t_lat['mean_ms'] / max(s_lat['mean_ms'], 0.001)
        print(f"  seq_len={seq_len}: 全注意力={t_lat['mean_ms']:.2f}ms, "
              f"RTPurbo={s_lat['mean_ms']:.2f}ms, 比值={ratio:.3f}x")
        speed_results[f'seq_{seq_len}'] = {
            'teacher_ms': t_lat['mean_ms'], 'student_ms': s_lat['mean_ms'], 'ratio': ratio
        }
    results['speed'] = speed_results

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("  📊 评估汇总")
    print("=" * 70)
    print(f"  Perplexity:       教师={t_ppl:.2f}  学生={s_ppl:.2f}  差距={ppl_gap:+.2f}%")
    print(f"  Top-1 Token一致率: {agreement['top1_agreement']*100:.2f}%")
    print(f"  Top-5 Token一致率: {agreement['top5_agreement']*100:.2f}%")
    print(f"  稀疏度:           75% local heads, 25% retrieval heads")
    print(f"  新增参数:         21,504 (0.034%)")
    print("=" * 70)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 详细结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
