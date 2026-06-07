"""对比所有 Stage2 变体的 PPL 和 Token 一致率"""
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
head_config = load_head_config('../head_config.json')

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
    return ppl

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
            t1 = t_l.argmax(-1); s1 = s_l.argmax(-1)
            agree1 += (t1 == s1).sum().item()
            t5 = torch.topk(t_l, 5, dim=-1).indices
            agree5 += (t5 == s1.unsqueeze(-1)).any(-1).sum().item()
            total += t1.numel()
    return agree1 / max(total, 1) * 100, agree5 / max(total, 1) * 100

cfg_sparse = RTPurboConfig(hidden_size=768, num_hidden_layers=8,
                           sparse_attn=True, index_dim=16,
                           local_window_size=256, retrieval_top_p=0.95)

# 教师
print('加载教师...')
tc = MiniMindConfig(hidden_size=768, num_hidden_layers=8, flash_attn=True)
teacher = MiniMindForCausalLM(tc)
teacher.load_state_dict(torch.load('../out/full_sft_768.pth', map_location=device), strict=False)
teacher = teacher.to(device)
t_ppl = compute_ppl(teacher, '教师')

results = [('教师(全注意力)', t_ppl, '-', '-')]

# 测试各变体
variants = [
    ('Stage1(200步)', '../out/rtpurbo_stage1_768.pth'),
    ('Stage1(400步)', '../out/rtpurbo_stage1_400steps_768.pth'),
    ('Stage2-lr5e-5(旧)', '../out/rtpurbo_stage2_768.pth'),
    ('Stage2-lr1e-6(新)', '../out/rtpurbo_stage2_lowlr_768.pth'),
]

for name, path in variants:
    if not os.path.exists(path):
        print(f'  跳过 {name}: {path} 不存在')
        continue
    print(f'\n--- {name} ---')
    m = RTPurboForCausalLM(cfg_sparse, head_config)
    m.load_state_dict(torch.load(path, map_location=device), strict=False)
    m = m.to(device)
    ppl = compute_ppl(m, name)
    a1, a5 = compute_agreement(teacher, m, name)
    results.append((name, ppl, f'{a1:.2f}%', f'{a5:.2f}%'))
    del m; torch.cuda.empty_cache()

# 输出结果
print('\n' + '=' * 70)
print('  全部方案对比')
print('=' * 70)
print(f'  {"方案":<25} {"PPL":>8} {"vs教师":>10} {"Top-1":>10} {"Top-5":>10}')
print(f'  {"─"*25} {"─"*8} {"─"*10} {"─"*10} {"─"*10}')
for name, ppl, a1, a5 in results:
    gap = '基线' if name.startswith('教师') else f'{(ppl-t_ppl)/t_ppl*100:+.2f}%'
    print(f'  {name:<25} {ppl:>8.2f} {gap:>10} {a1:>10} {a5:>10}')
print('=' * 70)
