"""
RTPurbo 多维度消融与验证实验脚本
================================
1. 自动对 index_dim = [4, 8, 16, 32] 运行 Stage 1 投影训练（各 200 步）。
2. 在训练中自动捕获并记录 proj_loss 变化曲线。
3. 对所有训练好的模型进行多维度网格扫描评估 (index_dim x window_size x top_p)。
4. 统计精确的注意力稀疏度、PPL 和与教师模型的一致率。
5. 保存所有过程数据和最终报表为 JSON 文件。
"""

import os
import sys
import json
import re
import subprocess
import time
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_rtpurbo import RTPurboConfig, RTPurboForCausalLM, load_head_config

# 路径定义
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(script_dir, '../model'))
dataset_path = os.path.abspath(os.path.join(script_dir, '../dataset/pretrain_t2t_mini.jsonl'))
head_config_path = os.path.abspath(os.path.join(script_dir, '../head_config.json'))
base_weight_path = os.path.abspath(os.path.join(script_dir, '../out/full_sft_768.pth'))
out_dir = os.path.abspath(os.path.join(script_dir, '../out'))

# 配置
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.makedirs(out_dir, exist_ok=True)

# 评估超参组合
index_dims = [4, 8, 16, 32]
window_sizes = [64, 128, 256]
top_ps = [0.85, 0.90, 0.95, 0.99]

# 1. 自动加载评测数据
print("加载评估数据...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
dataset = load_dataset('json', data_files=dataset_path, split='train')
# 使用固定种子随机挑选 200 条作为评估集，确保对比的公平性
indices = np.random.RandomState(123).permutation(len(dataset))[:200]
eval_texts = [str(dataset[int(i)]['text']) for i in indices]
print(f"评估集加载完成，共 {len(eval_texts)} 条文本")


def run_stage1_train(idx_dim):
    """自动执行 Stage 1 训练，并捕获损失函数曲线"""
    print(f"\n==========================================")
    print(f" 开始 Stage 1 训练：index_dim = {idx_dim}")
    print(f"==========================================")
    
    cmd = [
        sys.executable,
        os.path.abspath(os.path.join(script_dir, "../trainer/train_rtpurbo_stage1.py")),
        "--max_train_steps", "200",
        "--index_dim", str(idx_dim),
        "--save_tag", f"ablation_idx{idx_dim}",
        "--batch_size", "32",
        "--learning_rate", "1e-3",
        "--num_workers", "0",
        "--data_path", dataset_path,
        "--base_weight", "full_sft",
        "--head_config", head_config_path,
        "--save_dir", out_dir
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 启动子进程
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=os.path.abspath(os.path.join(script_dir, ".."))
    )
    
    loss_curve = []
    
    # 实时读取 stdout 提取 loss
    loss_pattern = re.compile(r"step\): (\d+)/\d+.*proj_loss: ([0-9.]+)")
    loss_pattern_alt = re.compile(r"\((\d+)/200\).*proj_loss: ([0-9.]+)")
    
    for line in iter(process.stdout.readline, ''):
        print(line, end='')  # 打印到当前终端
        match = loss_pattern.search(line) or loss_pattern_alt.search(line)
        if match:
            step = int(match.group(1))
            loss_val = float(match.group(2))
            loss_curve.append({"step": step, "loss": loss_val})
            
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        print(f"警告: index_dim = {idx_dim} 训练进程退出码异常 ({return_code})")
        
    return loss_curve


def evaluate_model(model, teacher, name, window_size, top_p):
    """在一轮迭代中计算 PPL、Token 一致率和实际稀疏度"""
    model.eval()
    if teacher is not None:
        teacher.eval()
        
    # 设置超参
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            if hasattr(layer, 'self_attn'):
                if hasattr(layer.self_attn, 'local_window_size'):
                    layer.self_attn.local_window_size = window_size
                if hasattr(layer.self_attn, 'retrieval_top_p'):
                    layer.self_attn.retrieval_top_p = top_p
        
    if hasattr(model, 'reset_sparsity_stats'):
        model.reset_sparsity_stats()
    
    total_ce_loss = 0.0
    total_tokens = 0
    agree1, agree5, total_agree_tokens = 0, 0, 0
    
    with torch.no_grad():
        for text in eval_texts:
            tokens = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
            ids = tokens['input_ids'].to(device)
            if ids.shape[1] < 10: 
                continue
                
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # 教师模型前向
                if teacher is not None:
                    t_out = teacher(ids)
                    t_logits = t_out.logits[:, :-1, :].float()
                
                # 学生模型前向
                s_out = model(ids)
                s_logits = s_out.logits[:, :-1, :].float()
                
            # 计算 PPL 的 Cross Entropy Loss
            labels = ids[:, 1:]
            loss = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)), labels.reshape(-1), reduction='sum')
            total_ce_loss += loss.item()
            total_tokens += labels.numel()
            
            # 计算 Token 一致率（在 200 条上全部计算，保证数据量大）
            if teacher is not None:
                t1 = t_logits.argmax(-1)
                s1 = s_logits.argmax(-1)
                agree1 += (t1 == s1).sum().item()
                
                t5 = torch.topk(t_logits, 5, dim=-1).indices
                agree5 += (t5 == s1.unsqueeze(-1)).any(-1).sum().item()
                total_agree_tokens += t1.numel()
                
    ppl = np.exp(total_ce_loss / max(total_tokens, 1))
    avg_ce = total_ce_loss / max(total_tokens, 1)
    
    agree1_rate = (agree1 / max(total_agree_tokens, 1)) * 100 if teacher is not None else 100.0
    agree5_rate = (agree5 / max(total_agree_tokens, 1)) * 100 if teacher is not None else 100.0
    
    # 收集稀疏度
    if hasattr(model, 'get_sparsity_stats'):
        sparsity_stats = model.get_sparsity_stats()
        overall_sparsity = sparsity_stats['total']['sparsity']
        causal_tokens = int(sparsity_stats['total']['causal_tokens'])
        sparse_tokens = int(sparsity_stats['total']['sparse_tokens'])
    else:
        overall_sparsity = 0.0
        causal_tokens = 0
        sparse_tokens = 0
    
    return {
        "ppl": float(ppl),
        "avg_ce": float(avg_ce),
        "agree_top1": float(agree1_rate),
        "agree_top5": float(agree5_rate),
        "sparsity": float(overall_sparsity),
        "causal_tokens": causal_tokens,
        "sparse_tokens": sparse_tokens
    }


def main():
    # 1. 评估教师模型作为基线
    print("\n--- 加载全注意力教师模型 ---")
    t_cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8, flash_attn=True)
    teacher = MiniMindForCausalLM(t_cfg)
    teacher.load_state_dict(torch.load(base_weight_path, map_location=device), strict=False)
    teacher = teacher.to(device)
    
    print("评估教师基线数据...")
    # 对于教师模型，我们没有 index_dim，没有稀疏注意力，用特殊的评估
    # 我们用一个 dummy 稀疏模型跑 100% 密集作为教师的 ppl 验证
    # 其实可以直接跑 evaluate_model，但传入 None 作为 teacher，返回 100% 一致率
    teacher_stats = evaluate_model(teacher, None, "Teacher", 512, 1.0)
    teacher_stats["sparsity"] = 0.0 # 0% 稀疏
    print(f"教师基线 PPL: {teacher_stats['ppl']:.4f}, 平均 CE Loss: {teacher_stats['avg_ce']:.4f}")
    
    results = {
        "teacher": teacher_stats,
        "ablation_experiments": {}
    }
    
    # 2. 依次训练 4 种投影维度并收集 Loss 曲线
    loss_curves = {}
    for idx_dim in index_dims:
        curve = run_stage1_train(idx_dim)
        loss_curves[str(idx_dim)] = curve
        
    results["training_loss_curves"] = loss_curves
    
    # 保存当前的中间状态
    with open(os.path.join(out_dir, "ablation_results_temp.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    # 3. 网格扫描评估
    head_config = load_head_config(head_config_path)
    
    for idx_dim in index_dims:
        weight_path = os.path.join(out_dir, f"rtpurbo_stage1_ablation_idx{idx_dim}_768.pth")
        if not os.path.exists(weight_path):
            print(f"警告: {weight_path} 不存在，跳过该维度的评估")
            continue
            
        print(f"\n--- 载入并网格评估 index_dim = {idx_dim} 的模型 ---")
        cfg_sparse = RTPurboConfig(
            hidden_size=768, num_hidden_layers=8,
            sparse_attn=True, index_dim=idx_dim,
            local_window_size=128, retrieval_top_p=0.9
        )
        
        model = RTPurboForCausalLM(cfg_sparse, head_config)
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        model = model.to(device)
        
        dim_results = []
        
        # 遍历超参组合
        pbar = tqdm(total=len(window_sizes) * len(top_ps), desc=f"评估 dim={idx_dim}")
        for w in window_sizes:
            for p in top_ps:
                eval_res = evaluate_model(model, teacher, f"dim{idx_dim}", w, p)
                eval_res["window_size"] = w
                eval_res["top_p"] = p
                dim_results.append(eval_res)
                pbar.update(1)
        pbar.close()
        
        results["ablation_experiments"][str(idx_dim)] = dim_results
        
        # 释放显存
        del model
        torch.cuda.empty_cache()
        
    # 保存最终实验结果
    final_path = os.path.join(out_dir, "ablation_results.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\n==========================================")
    print(f" 消融实验全部完成！")
    print(f" 数据已保存至: {final_path}")
    print(f"==========================================")
    
    # 打印简易报表
    print(f"\n--- 经典配置对比快照 (window_size=256, top_p=0.95) ---")
    print(f"{'投影维度':<10} | {'PPL':<8} | {'PPL 差距':<10} | {'Top-1 一致率':<12} | {'实际稀疏度':<10}")
    print("-" * 65)
    t_ppl = teacher_stats["ppl"]
    print(f"{'教师(全注意力)':<10} | {t_ppl:<8.4f} | {'0.00%':<10} | {'100.00%':<12} | {'0.00%':<10}")
    
    for idx_dim in index_dims:
        experiments = results["ablation_experiments"].get(str(idx_dim), [])
        target_exp = next((e for e in experiments if e["window_size"] == 256 and e["top_p"] == 0.95), None)
        if target_exp:
            gap = (target_exp["ppl"] - t_ppl) / t_ppl * 100
            gap_str = f"{gap:+.2f}%"
            agree_str = f"{target_exp['agree_top1']:.2f}%"
            sparsity_str = f"{target_exp['sparsity']*100:.2f}%"
            print(f"{idx_dim:<12} | {target_exp['ppl']:<8.4f} | {gap_str:<10} | {agree_str:<12} | {sparsity_str:<10}")


if __name__ == "__main__":
    main()
