"""诊断 RTPurbo PPL 差距的根因"""
import os, sys, torch, json
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_rtpurbo import RTPurboConfig, RTPurboForCausalLM, load_head_config
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained('../model')

# 加载测试数据
dataset = load_dataset('json', data_files='../dataset/pretrain_t2t_mini.jsonl', split='train')
indices = np.random.RandomState(123).permutation(len(dataset))[:50]
eval_texts = [str(dataset[int(i)]['text']) for i in indices]

def compute_ppl(model, texts, label=""):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in texts[:50]:
            tokens = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
            ids = tokens['input_ids'].to(device)
            if ids.shape[1] < 10: continue
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = model(ids)
            logits = out.logits[:, :-1, :].float()
            labels = ids[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += labels.numel()
    ppl = np.exp(total_loss / max(total_tokens, 1))
    print(f"  [{label}] PPL={ppl:.2f}, tokens={total_tokens}")
    return ppl

# ---- 测试 1: 教师模型 ----
print("=" * 60)
print("诊断 1: 教师模型 PPL")
teacher_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, flash_attn=True)
teacher = MiniMindForCausalLM(teacher_config)
w = torch.load('../out/full_sft_768.pth', map_location=device)
teacher.load_state_dict(w, strict=False)
teacher = teacher.to(device)
compute_ppl(teacher, eval_texts, "教师-FlashAttn")

# ---- 测试 2: RTPurbo 关闭稀疏 (验证权重正确性) ----
print("\n诊断 2: RTPurbo 关闭稀疏注意力 (应该≈教师)")
head_config = load_head_config('../head_config.json')
config_nosparse = RTPurboConfig(hidden_size=768, num_hidden_layers=8,
                                 sparse_attn=False, index_dim=16,
                                 local_window_size=128, retrieval_top_p=0.9)
student_nosparse = RTPurboForCausalLM(config_nosparse, head_config)
w2 = torch.load('../out/rtpurbo_stage2_768.pth', map_location=device)
student_nosparse.load_state_dict(w2, strict=False)
student_nosparse = student_nosparse.to(device)
compute_ppl(student_nosparse, eval_texts, "学生-无稀疏")

# ---- 测试 3: RTPurbo 开启稀疏 ----
print("\n诊断 3: RTPurbo 开启稀疏注意力")
config_sparse = RTPurboConfig(hidden_size=768, num_hidden_layers=8,
                               sparse_attn=True, index_dim=16,
                               local_window_size=128, retrieval_top_p=0.9)
student_sparse = RTPurboForCausalLM(config_sparse, head_config)
student_sparse.load_state_dict(w2, strict=False)
student_sparse = student_sparse.to(device)
compute_ppl(student_sparse, eval_texts, "学生-有稀疏")

# ---- 测试 4: RTPurbo 用基线权重 + 关闭稀疏 ----
print("\n诊断 4: RTPurbo 加载基线权重 + 关闭稀疏 (验证架构等价)")
student_base = RTPurboForCausalLM(config_nosparse, head_config)
student_base.load_state_dict(w, strict=False)
student_base = student_base.to(device)
compute_ppl(student_base, eval_texts, "学生-基线权重-无稀疏")

# ---- 测试 5: 调整 window 和 top_p ----
print("\n诊断 5: 增大 window=256, top_p=0.95")
config_wide = RTPurboConfig(hidden_size=768, num_hidden_layers=8,
                             sparse_attn=True, index_dim=16,
                             local_window_size=256, retrieval_top_p=0.95)
student_wide = RTPurboForCausalLM(config_wide, head_config)
student_wide.load_state_dict(w2, strict=False)
student_wide = student_wide.to(device)
compute_ppl(student_wide, eval_texts, "学生-宽窗口")

print("\n诊断 6: 增大 window=512 (≈全注意力)")
config_full_win = RTPurboConfig(hidden_size=768, num_hidden_layers=8,
                                 sparse_attn=True, index_dim=16,
                                 local_window_size=512, retrieval_top_p=0.99)
student_full_win = RTPurboForCausalLM(config_full_win, head_config)
student_full_win.load_state_dict(w2, strict=False)
student_full_win = student_full_win.to(device)
compute_ppl(student_full_win, eval_texts, "学生-全窗口")

print("\n" + "=" * 60)
print("如果 诊断2 ≈ 诊断1 → 权重没问题, 问题在稀疏掩码")
print("如果 诊断2 >> 诊断1 → 权重被蒸馏训练破坏了")
print("如果 诊断4 ≈ 诊断1 → RTPurbo 架构与原模型等价")
