"""
RTPurbo 离线 Head 标定 (Qwen3.5-4B 适配版)
=========================================
分析 Qwen3.5-4B 模型中，每个 attention head 对远距离 token 的注意力占比（Retrieval Score），
并自动将所有 heads 分为：
  - Retrieval Head: 对远距离检索极度重要，需要使用低维投影稀疏注意力并维持完整 KV 缓存
  - Local Head:     主要关注滑动窗口内的局部 token，可以被局部窗口阶段限制
"""

import os
import sys
import json
import argparse
import torch
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e4m3fn", torch.float32)
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, AutoModelForImageTextToText
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
from datasets import load_dataset
from model.model_qwen_rtpurbo import _get_layers


def collect_attention_scores(model, tokenizer, data_samples, max_seq_len=512, device='cuda'):
    """
    通过对前向传播进行 monkey-patch，在标定数据上收集每个 head 的完整注意力权重
    """
    model.eval()
    config = model.config.text_config if hasattr(model.config, "text_config") else model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads

    # 存储每个 head 的 remote attention ratios
    retrieval_scores = [[[] for _ in range(num_heads)] for _ in range(num_layers)]

    attn_weights_cache = {}
    original_forwards = {}

    layers = _get_layers(model)

    # 动态 patch 每一层的 Attention forward（仅针对 full_attention 层）
    for layer_idx, layer in enumerate(layers):
        if hasattr(config, "layer_types") and config.layer_types[layer_idx] != "full_attention":
            continue

        attn = layer.self_attn
        original_forwards[layer_idx] = attn.forward

        # 使用闭包绑定当前的 layer_idx，适配 Qwen3.5 Attention 结构
        def make_patched_forward(l_idx, original_attn):
            def patched_forward(self, hidden_states, position_embeddings, attention_mask=None,
                                past_key_values=None, cache_position=None, **kwargs):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                # 1. 门控划分
                query_states, gate = torch.chunk(
                    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
                )
                gate = gate.reshape(*input_shape, -1)

                # 2. 归一化
                query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
                value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

                # 3. 计算全注意力得分
                scores = torch.matmul(query_states, key_states_expanded.transpose(-2, -1)) * self.scaling
                seq_len = query_states.shape[2]
                T_kv = key_states_expanded.shape[2]

                causal_mask = torch.triu(
                    torch.ones(seq_len, T_kv, device=scores.device), diagonal=T_kv - seq_len + 1
                ).bool()
                scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                attn_probs = F.softmax(scores.float(), dim=-1).type_as(query_states)

                # 缓存离线分析所需的注意力概率
                attn_weights_cache[l_idx] = attn_probs.detach().cpu()

                attn_output = torch.matmul(attn_probs, value_states_expanded)
                attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                
                # 4. 门控相乘后投影输出
                attn_output = attn_output * torch.sigmoid(gate)
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_probs

            import types
            return types.MethodType(patched_forward, original_attn)

        attn.forward = make_patched_forward(layer_idx, attn)

    local_window = args.local_window

    with torch.no_grad():
        for sample_idx, text in enumerate(tqdm(data_samples, desc="标定中")):
            # 对极长文本数据做必要截取以防 OOM
            tokens = tokenizer(text, return_tensors='pt', max_length=max_seq_len,
                               truncation=True, add_special_tokens=True)
            input_ids = tokens['input_ids'].to(device)
            seq_len = input_ids.shape[1]

            if seq_len < local_window + 10:
                continue  # 序列过短无法区分远近，跳过

            attn_weights_cache.clear()
            model(input_ids)

            # 对每一层、每一个 head 提取远距离注意力占比
            for layer_idx in range(num_layers):
                if layer_idx not in attn_weights_cache:
                    continue
                attn_probs = attn_weights_cache[layer_idx]

                for head_idx in range(num_heads):
                    head_attn = attn_probs[0, head_idx]  # (T, T)

                    remote_ratios = []
                    for q_pos in range(local_window, seq_len):
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

    # 标定结束，还原原始 forward 方法
    for layer_idx, layer in enumerate(layers):
        if layer_idx in original_forwards:
            layer.self_attn.forward = original_forwards[layer_idx]

    return retrieval_scores


def classify_heads(retrieval_scores, retrieval_ratio=0.25):
    """根据远距离注意力占比进行 heads 分类"""
    num_layers = len(retrieval_scores)
    num_heads = len(retrieval_scores[0])

    head_scores = []
    calibrated_layers = []

    for layer_idx in range(num_layers):
        # 如果该层所有的 head 都没有收集到任何分数，说明被跳过了
        if any(len(retrieval_scores[layer_idx][h]) > 0 for h in range(num_heads)):
            calibrated_layers.append(layer_idx)
            for head_idx in range(num_heads):
                scores = retrieval_scores[layer_idx][head_idx]
                avg_score = np.mean(scores) if scores else 0.0
                head_scores.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'retrieval_score': float(avg_score)
                })

    # 按 retrieval_score 降序排序
    head_scores.sort(key=lambda x: x['retrieval_score'], reverse=True)

    # 计算 retrieval heads 数量
    total_calibrated_heads = len(head_scores)
    num_retrieval = max(1, int(total_calibrated_heads * retrieval_ratio))

    retrieval_set = set()
    for i in range(num_retrieval):
        h = head_scores[i]
        retrieval_set.add((h['layer'], h['head']))

    head_config = {
        'num_layers': num_layers,
        'num_heads': num_heads,
        'retrieval_ratio': retrieval_ratio,
        'num_retrieval_heads': num_retrieval,
        'num_local_heads': total_calibrated_heads - num_retrieval,
        'heads': [],
        'per_layer_summary': []
    }

    for layer_idx in range(num_layers):
        if layer_idx not in calibrated_layers:
            # linear_attention 层没有 retrieval/local heads
            head_config['per_layer_summary'].append({
                'layer': layer_idx,
                'retrieval_heads': [],
                'local_heads': []
            })
            continue

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

    # 打印标定排名结果
    print("\n" + "=" * 60)
    print("Head Retrieval Score 排名（降序，前 20）")
    print("=" * 60)
    for i, h in enumerate(head_scores[:20]):
        tag = "🔴 RETRIEVAL" if (h['layer'], h['head']) in retrieval_set else "🔵 LOCAL"
        print(f"  #{i+1:2d}  Layer {h['layer']}, Head {h['head']}  "
              f"score={h['retrieval_score']:.6f}  {tag}")

    print(f"\n总计标定了 {len(calibrated_layers)} 层，产生: {num_retrieval} retrieval heads, {total_calibrated_heads - num_retrieval} local heads")
    return head_config


def main():
    global args
    parser = argparse.ArgumentParser(description="Qwen3.5 RTPurbo Head Calibration")
    parser.add_argument("--model_path", type=str, default="../model/Qwen3.5-4B",
                        help="预训练 Qwen 模型文件夹路径")
    parser.add_argument("--data_path", type=str, default="THUDM/LongAlign-10k",
                        help="标定文本数据集文件路径或 HuggingFace 数据集名称")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="用于标定的随机样本数量")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="最大序列长度限制")
    parser.add_argument("--local_window", type=int, default=128,
                        help="局部滑动窗口的大小")
    parser.add_argument("--retrieval_ratio", type=float, default=0.25,
                        help="定义 Retrieval 头在所有头中的百分占比")
    parser.add_argument("--output", type=str, default="../qwen_head_config.json",
                        help="最终导出的配置文件位置")
    parser.add_argument("--score_output", type=str, default=None,
                        help="导出完整标定分数的 CSV 文件位置")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="运行设备")
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3.5 RTPurbo Head Calibration")
    print("=" * 60)
    print(f"  模型路径: {args.model_path}")
    print(f"  数据集: {args.data_path}")
    print(f"  标定样本: {args.num_samples} 条")
    print(f"  局部窗口: {args.local_window}")
    print(f"  Retrieval 占比: {args.retrieval_ratio}")
    score_out_desc = args.score_output if args.score_output else "根据 output 自动生成"
    print(f"  分数 CSV 输出: {score_out_desc}")

    # 1. 载入模型和 Tokenizer
    print("\n载入预训练 Qwen3.5-4B 模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
        attn_implementation="eager"
    )
    print("模型加载完成！")

    # 2. 载入数据集（支持本地 JSONL 或 THUDM/LongAlign-10k 镜像高速流式载入）
    print(f"\n载入标定语料：{args.data_path}")
    if args.data_path == "THUDM/LongAlign-10k":
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("开启国内高速 HuggingFace 镜像源，流式加载清华 LongAlign-10k 中...")
        dataset = load_dataset('THUDM/LongAlign-10k', split='train', streaming=True)
        data_samples = []
        for item in tqdm(dataset, desc="拉取天然长文本数据"):
            messages = item['messages']
            # 防御性合并多轮长对话为连贯长文档进行注意力标定
            text_parts = []
            for msg in messages:
                if 'content' in msg:
                    text_parts.append(msg['content'])
                elif 'value' in msg:
                    text_parts.append(msg['value'])
            text = "\n".join(text_parts)
            data_samples.append(text)
            if len(data_samples) >= args.num_samples:
                break
    else:
        dataset = load_dataset('json', data_files=args.data_path, split='train')
        indices = np.random.RandomState(42).permutation(len(dataset))[:args.num_samples]
        data_samples = [str(dataset[int(i)]['text']) for i in indices]
        
    print(f"成功获取了 {len(data_samples)} 条连贯长文本样本。")

    # 3. 收集注意力统计数据
    retrieval_scores = collect_attention_scores(
        model, tokenizer, data_samples,
        max_seq_len=args.max_seq_len,
        device=args.device
    )

    # 4. 头分类与生成配置
    head_config = classify_heads(retrieval_scores, args.retrieval_ratio)

    head_config['calibration_params'] = {
        'model_path': args.model_path,
        'data_path': args.data_path,
        'num_samples': args.num_samples,
        'max_seq_len': args.max_seq_len,
        'local_window': args.local_window,
        'retrieval_ratio': args.retrieval_ratio
    }

    # 5. 保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(head_config, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 成功将 Qwen3.5 头标定配置写入：{args.output}")

    # 额外导出 CSV
    score_output = args.score_output
    if score_output is None:
        if args.output.endswith('.json'):
            score_output = args.output[:-5] + '_scores.csv'
        else:
            score_output = args.output + '_scores.csv'
    
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
