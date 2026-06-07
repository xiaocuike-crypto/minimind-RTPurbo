"""
RTPurbo 离线 Head 标定 (Offline Head-wise Calibration)
====================================================
分析预训练好的 MiniMind 模型中，每个 attention head 的注意力模式：
  - Retrieval Head: 对远距离 token 有显著注意力，需要保留完整 KV cache
  - Local Head:     主要关注局部窗口内的 token，可以只保留窗口内 KV

方法：
  1. 在标定数据上跑前向推理（关闭 flash attention）
  2. 对每个 head，按 query-key 距离分桶统计注意力权重
  3. 计算 "retrieval score" = 远距离桶的平均注意力占比
  4. 按 retrieval score 排序，标记 top-K% 为 retrieval head

用法：
  python scripts/calibrate_heads.py [--retrieval_ratio 0.25] [--local_window 128]
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset


def collect_attention_scores(model, tokenizer, data_samples, max_seq_len=512, device='cuda'):
    """
    在标定数据上收集每个 head 的注意力分数。
    返回 per-head 的距离分布统计。
    """
    model.eval()
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # 累积每个 head 的 "远距离注意力比例"
    # retrieval_scores[layer][head] = list of per-sample remote attention ratios
    retrieval_scores = [[[] for _ in range(num_heads)] for _ in range(num_layers)]

    # 注册 hook 来捕获注意力分数
    attn_weights_cache = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # 在 non-flash 路径中，手动计算注意力分数
            # 我们需要在 forward 之外重新计算，因为原代码没有保存中间 scores
            pass
        return hook_fn

    # 由于原始 Attention 类不保存注意力权重，我们直接修改 forward 来捕获
    # 更优雅的方式：临时 monkey-patch forward
    import math
    import torch.nn.functional as F
    from model.model_minimind import repeat_kv

    original_forwards = {}
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        original_forwards[layer_idx] = attn.forward

        def patched_forward(self, x, position_embeddings, past_key_value=None,
                            use_cache=False, attention_mask=None, _layer_idx=layer_idx):
            bsz, seq_len, _ = x.shape
            xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
            xq, xk = self.q_norm(xq), self.k_norm(xk)
            cos, sin = position_embeddings
            from model.model_minimind import apply_rotary_pos_emb
            xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
            xq = xq.transpose(1, 2)  # (B, n_heads, T, head_dim)
            xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
            xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

            # 计算注意力分数（不使用 flash attention）
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 因果掩码
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_probs = F.softmax(scores.float(), dim=-1).type_as(xq)  # (B, n_heads, T, T)

            # 保存注意力权重供后续分析
            attn_weights_cache[_layer_idx] = attn_probs.detach().cpu()

            output = attn_probs @ xv
            output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
            output = self.resid_dropout(self.o_proj(output))
            return output, None

        import types
        attn.forward = types.MethodType(patched_forward, attn)

    local_window = args.local_window

    with torch.no_grad():
        for sample_idx, text in enumerate(tqdm(data_samples, desc="标定中")):
            tokens = tokenizer(text, return_tensors='pt', max_length=max_seq_len,
                               truncation=True, add_special_tokens=True)
            input_ids = tokens['input_ids'].to(device)
            seq_len = input_ids.shape[1]

            if seq_len < local_window + 10:
                continue  # 序列太短，跳过

            attn_weights_cache.clear()
            model(input_ids)

            # 分析每个 head 的注意力模式
            for layer_idx in range(num_layers):
                if layer_idx not in attn_weights_cache:
                    continue
                attn_probs = attn_weights_cache[layer_idx]  # (1, n_heads, T, T)

                for head_idx in range(num_heads):
                    head_attn = attn_probs[0, head_idx]  # (T, T)

                    # 计算每个 query position 对 "远距离 key" 的注意力比例
                    # 远距离 = 距离 > local_window
                    remote_ratios = []
                    for q_pos in range(local_window, seq_len):
                        # 对于 query position q_pos，local window = [q_pos-local_window, q_pos]
                        # 远距离 = [0, q_pos-local_window)
                        remote_end = q_pos - local_window
                        if remote_end <= 0:
                            continue
                        remote_attn = head_attn[q_pos, :remote_end].sum().item()
                        total_attn = head_attn[q_pos, :q_pos + 1].sum().item()
                        if total_attn > 1e-8:
                            remote_ratios.append(remote_attn / total_attn)

                    if remote_ratios:
                        avg_remote = np.mean(remote_ratios)
                        retrieval_scores[layer_idx][head_idx].append(avg_remote)

    # 恢复原始 forward
    for layer_idx, layer in enumerate(model.model.layers):
        layer.self_attn.forward = original_forwards[layer_idx]

    return retrieval_scores


def classify_heads(retrieval_scores, retrieval_ratio=0.25):
    """
    根据 retrieval scores 将 heads 分为 retrieval / local。
    返回 head_config dict。
    """
    num_layers = len(retrieval_scores)
    num_heads = len(retrieval_scores[0])

    # 计算每个 head 的平均 retrieval score
    head_scores = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            scores = retrieval_scores[layer_idx][head_idx]
            avg_score = np.mean(scores) if scores else 0.0
            head_scores.append({
                'layer': layer_idx,
                'head': head_idx,
                'retrieval_score': float(avg_score)
            })

    # 按 retrieval score 降序排列
    head_scores.sort(key=lambda x: x['retrieval_score'], reverse=True)

    # 标记 top retrieval_ratio 为 retrieval head
    total_heads = num_layers * num_heads
    num_retrieval = max(1, int(total_heads * retrieval_ratio))

    retrieval_set = set()
    for i in range(num_retrieval):
        h = head_scores[i]
        retrieval_set.add((h['layer'], h['head']))

    # 构建配置
    head_config = {
        'num_layers': num_layers,
        'num_heads': num_heads,
        'retrieval_ratio': retrieval_ratio,
        'num_retrieval_heads': num_retrieval,
        'num_local_heads': total_heads - num_retrieval,
        'heads': [],
        'per_layer_summary': []
    }

    for layer_idx in range(num_layers):
        layer_retrieval = []
        layer_local = []
        for head_idx in range(num_heads):
            is_retrieval = (layer_idx, head_idx) in retrieval_set
            scores = retrieval_scores[layer_idx][head_idx]
            avg_score = float(np.mean(scores)) if scores else 0.0
            head_config['heads'].append({
                'layer': layer_idx,
                'head': head_idx,
                'type': 'retrieval' if is_retrieval else 'local',
                'retrieval_score': avg_score
            })
            if is_retrieval:
                layer_retrieval.append(head_idx)
            else:
                layer_local.append(head_idx)

        head_config['per_layer_summary'].append({
            'layer': layer_idx,
            'retrieval_heads': layer_retrieval,
            'local_heads': layer_local
        })

    # 打印排名
    print("\n" + "=" * 60)
    print("Head Retrieval Score 排名（降序）")
    print("=" * 60)
    for i, h in enumerate(head_scores):
        tag = "🔴 RETRIEVAL" if (h['layer'], h['head']) in retrieval_set else "🔵 LOCAL"
        print(f"  #{i+1:2d}  Layer {h['layer']}, Head {h['head']}  "
              f"score={h['retrieval_score']:.6f}  {tag}")

    print(f"\n总计: {num_retrieval} retrieval heads, "
          f"{total_heads - num_retrieval} local heads")

    for layer_summary in head_config['per_layer_summary']:
        print(f"  Layer {layer_summary['layer']}: "
              f"retrieval={layer_summary['retrieval_heads']}, "
              f"local={layer_summary['local_heads']}")

    return head_config


def main():
    global args
    parser = argparse.ArgumentParser(description="RTPurbo Head Calibration")
    parser.add_argument("--model_path", type=str, default="../out",
                        help="模型权重目录")
    parser.add_argument("--weight", type=str, default="full_sft",
                        help="权重名称前缀")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl",
                        help="标定数据路径")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="标定样本数量")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--local_window", type=int, default=128,
                        help="Local head 窗口大小（用于区分远/近距离）")
    parser.add_argument("--retrieval_ratio", type=float, default=0.25,
                        help="标记为 retrieval head 的比例")
    parser.add_argument("--output", type=str, default="../head_config.json",
                        help="输出配置文件路径")
    parser.add_argument("--score_output", type=str, default=None,
                        help="输出详细标定分数的 CSV 文件位置")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="设备")
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="模型隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=8,
                        help="模型层数")
    args = parser.parse_args()

    print("=" * 60)
    print("RTPurbo Head Calibration")
    print("=" * 60)
    print(f"  模型权重: {args.model_path}/{args.weight}")
    print(f"  标定数据: {args.data_path}")
    print(f"  样本数量: {args.num_samples}")
    print(f"  序列长度: {args.max_seq_len}")
    print(f"  局部窗口: {args.local_window}")
    print(f"  Retrieval 比例: {args.retrieval_ratio}")

    # 加载模型
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        flash_attn=False  # 关闭 flash attention 以获取注意力分数
    )
    tokenizer = AutoTokenizer.from_pretrained('../model')
    model = MiniMindForCausalLM(lm_config)

    moe_suffix = '_moe' if lm_config.use_moe else ''
    weight_path = f'{args.model_path}/{args.weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    print(f"\n加载权重: {weight_path}")
    weights = torch.load(weight_path, map_location=args.device)
    model.load_state_dict(weights, strict=False)
    model = model.to(args.device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total_params:.2f}M")

    # 加载标定数据
    print(f"\n加载标定数据: {args.data_path}")
    dataset = load_dataset('json', data_files=args.data_path, split='train')
    indices = np.random.RandomState(42).permutation(len(dataset))[:args.num_samples]
    data_samples = [str(dataset[int(i)]['text']) for i in indices]
    print(f"采样 {len(data_samples)} 条数据")

    # 收集注意力分数
    print("\n开始收集注意力分数...")
    retrieval_scores = collect_attention_scores(
        model, tokenizer, data_samples,
        max_seq_len=args.max_seq_len,
        device=args.device
    )

    # 分类 heads
    head_config = classify_heads(retrieval_scores, args.retrieval_ratio)

    # 保存配置
    head_config['calibration_params'] = {
        'weight': args.weight,
        'num_samples': args.num_samples,
        'max_seq_len': args.max_seq_len,
        'local_window': args.local_window,
        'retrieval_ratio': args.retrieval_ratio
    }

    output_path = args.output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(head_config, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Head 配置已保存到: {output_path}")

    # 额外导出 CSV
    score_output = args.score_output
    if score_output is None:
        if output_path.endswith('.json'):
            score_output = output_path[:-5] + '_scores.csv'
        else:
            score_output = output_path + '_scores.csv'
    
    try:
        import csv
        # 确保父目录存在
        os.makedirs(os.path.dirname(os.path.abspath(score_output)), exist_ok=True)
        with open(score_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['layer', 'head', 'retrieval_score', 'type'])
            for h in head_config['heads']:
                writer.writerow([h['layer'], h['head'], h['retrieval_score'], h['type']])
        print(f"✅ 成功将详细标定指标 CSV 写入：{score_output}")
    except Exception as e:
        print(f"⚠️ 保存标定指标 CSV 失败: {e}")


if __name__ == "__main__":
    main()
