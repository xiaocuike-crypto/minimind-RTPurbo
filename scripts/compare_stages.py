"""对比 Stage1 vs 修复后 Stage2 的 PPL"""
import os, sys, torch, numpy as np
import torch.nn.functional as F
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_rtpurbo import RTPurboConfig, RTPurboForCausalLM, load_head_config
from transformers import AutoTokenizer
from datasets import load_dataset

device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained('../model')
dataset = load_dataset('json', data_files='../dataset/pretrain_t2t_mini.jsonl', split='train')
indices = np.random.RandomState(123).permutation(len(dataset))[:200]
eval_texts = [str(dataset[int(i)]['text']) for i in indices]

def compute_ppl(model, label):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in tqdm(eval_texts, desc=label):
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
    print(f'  [{label}] PPL={ppl:.2f}')
    return ppl

# Token 一致率
def compute_agreement(teacher, student, label, n=100):
    teacher.eval(); student.eval()
    agree1, agree5, total = 0, 0, 0
    with torch.no_grad():
        for text in tqdm(eval_texts[:n], desc=f'{label}-agree'):
            tokens = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
            ids = tokens['input_ids'].to(device)
            if ids.shape[1] < 10: continue
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                t_l = teacher(ids).logits[:, :-1, :].float()
                s_l = student(ids).logits[:, :-1, :].float()
            t1 = t_l.argmax(-1)
            s1 = s_l.argmax(-1)
            agree1 += (t1 == s1).sum().item()
            t5 = torch.topk(t_l, 5, dim=-1).indices
            agree5 += (t5 == s1.unsqueeze(-1)).any(-1).sum().item()
            total += t1.numel()
    r1 = agree1 / max(total, 1) * 100
    r5 = agree5 / max(total, 1) * 100
    print(f'  [{label}] Top-1: {r1:.2f}%, Top-5: {r5:.2f}%')
    return r1, r5

head_config = load_head_config('../head_config.json')

# 教师
print('=' * 60)
tc = MiniMindConfig(hidden_size=768, num_hidden_layers=8, flash_attn=True)
teacher = MiniMindForCausalLM(tc)
teacher.load_state_dict(torch.load('../out/full_sft_768.pth', map_location=device), strict=False)
teacher = teacher.to(device)
t_ppl = compute_ppl(teacher, '教师')

cfg_sparse = RTPurboConfig(hidden_size=768, num_hidden_layers=8,
                           sparse_attn=True, index_dim=16,
                           local_window_size=256, retrieval_top_p=0.95)

# Stage1 + 稀疏
print('\n--- Stage1 + 稀疏 ---')
m1 = RTPurboForCausalLM(cfg_sparse, head_config)
m1.load_state_dict(torch.load('../out/rtpurbo_stage1_768.pth', map_location=device), strict=False)
m1 = m1.to(device)
s1_ppl = compute_ppl(m1, 'Stage1')
s1_a1, s1_a5 = compute_agreement(teacher, m1, 'Stage1')
del m1; torch.cuda.empty_cache()

# Stage2 修复 + 稀疏
print('\n--- Stage2修复 + 稀疏 ---')
m2 = RTPurboForCausalLM(cfg_sparse, head_config)
m2.load_state_dict(torch.load('../out/rtpurbo_stage2_768.pth', map_location=device), strict=False)
m2 = m2.to(device)
s2_ppl = compute_ppl(m2, 'Stage2修复')
s2_a1, s2_a5 = compute_agreement(teacher, m2, 'Stage2修复')
del m2; torch.cuda.empty_cache()

# Stage2 修复 + 无稀疏 (验证权重完整性)
print('\n--- Stage2修复 + 无稀疏 ---')
cfg_ns = RTPurboConfig(hidden_size=768, num_hidden_layers=8,
                       sparse_attn=False, index_dim=16,
                       local_window_size=256, retrieval_top_p=0.95)
m3 = RTPurboForCausalLM(cfg_ns, head_config)
m3.load_state_dict(torch.load('../out/rtpurbo_stage2_768.pth', map_location=device), strict=False)
m3 = m3.to(device)
s2ns_ppl = compute_ppl(m3, 'Stage2修复-无稀疏')

print('\n' + '=' * 60)
print('  Stage 对比结果')
print('=' * 60)
print(f'  {"配置":<25} {"PPL":>8} {"差距":>10} {"Top-1":>8} {"Top-5":>8}')
print(f'  {"─"*25} {"─"*8} {"─"*10} {"─"*8} {"─"*8}')
print(f'  {"教师(全注意力)":<25} {t_ppl:>8.2f} {"基线":>10} {"─":>8} {"─":>8}')
print(f'  {"Stage1+稀疏":<25} {s1_ppl:>8.2f} {f"+{(s1_ppl-t_ppl)/t_ppl*100:.2f}%":>10} {f"{s1_a1:.2f}%":>8} {f"{s1_a5:.2f}%":>8}')
print(f'  {"Stage2修复+稀疏":<25} {s2_ppl:>8.2f} {f"+{(s2_ppl-t_ppl)/t_ppl*100:.2f}%":>10} {f"{s2_a1:.2f}%":>8} {f"{s2_a5:.2f}%":>8}')
print(f'  {"Stage2修复+无稀疏":<25} {s2ns_ppl:>8.2f} {f"+{(s2ns_ppl-t_ppl)/t_ppl*100:.2f}%":>10} {"─":>8} {"─":>8}')
print('=' * 60)
