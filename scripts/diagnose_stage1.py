"""验证 Stage 1 权重 + 稀疏注意力的表现"""
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

dataset = load_dataset('json', data_files='../dataset/pretrain_t2t_mini.jsonl', split='train')
indices = np.random.RandomState(123).permutation(len(dataset))[:50]
eval_texts = [str(dataset[int(i)]['text']) for i in indices]

head_config = load_head_config('../head_config.json')
base_w = torch.load('../out/full_sft_768.pth', map_location=device)
s1_w = torch.load('../out/rtpurbo_stage1_768.pth', map_location=device)

def compute_ppl(model, label=""):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in eval_texts:
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
    print(f"  [{label}] PPL={ppl:.2f}")
    return ppl

def make_model(sparse_attn, weights, label):
    config = RTPurboConfig(hidden_size=768, num_hidden_layers=8,
                           sparse_attn=sparse_attn, index_dim=16,
                           local_window_size=128, retrieval_top_p=0.9)
    model = RTPurboForCausalLM(config, head_config)
    model.load_state_dict(weights, strict=False)
    return model.to(device)

print("=" * 60)
print("方案验证：跳过 Stage 2，用 Stage 1 权重做稀疏推理")
print("=" * 60)

# 1. 基线：教师
print("\n1. 教师模型（基线）")
teacher_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, flash_attn=True)
teacher = MiniMindForCausalLM(teacher_config)
teacher.load_state_dict(base_w, strict=False)
teacher = teacher.to(device)
compute_ppl(teacher, "教师")

# 2. Stage 1 权重 + 无稀疏
print("\n2. Stage 1 权重 + 关闭稀疏")
m = make_model(False, s1_w, "Stage1-无稀疏")
compute_ppl(m, "Stage1-无稀疏")
del m

# 3. Stage 1 权重 + 有稀疏
print("\n3. Stage 1 权重 + 开启稀疏（核心测试）")
m = make_model(True, s1_w, "Stage1-有稀疏")
compute_ppl(m, "Stage1-有稀疏")
del m

# 4. 基线权重 + 有稀疏（无训练投影）
print("\n4. 基线权重 + 开启稀疏（未训练的随机投影）")
m = make_model(True, base_w, "基线-有稀疏-随机投影")
compute_ppl(m, "基线-有稀疏-随机投影")
del m

# 5. Stage 1 权重 + 宽窗口
print("\n5. Stage 1 权重 + 宽窗口 (256) + top_p=0.95")
config = RTPurboConfig(hidden_size=768, num_hidden_layers=8,
                       sparse_attn=True, index_dim=16,
                       local_window_size=256, retrieval_top_p=0.95)
m = RTPurboForCausalLM(config, head_config)
m.load_state_dict(s1_w, strict=False)
m = m.to(device)
compute_ppl(m, "Stage1-宽窗口")
del m

print("\n" + "=" * 60)
print("结论：如果 Stage1-有稀疏 ≈ 教师，则可跳过 Stage 2")
