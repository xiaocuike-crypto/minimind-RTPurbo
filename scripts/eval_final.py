"""
RTPurbo 最终评估（使用 Stage 1 权重 + 宽窗口配置）
"""
import os, sys, json, time, torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_rtpurbo import RTPurboConfig, RTPurboForCausalLM, load_head_config
from transformers import AutoTokenizer
from datasets import load_dataset

device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained('../model')

# ---- 加载模型 ----
print("=" * 70)
print("  RTPurbo 最终评估（Stage 1 权重 + 宽窗口）")
print("=" * 70)

# 教师
print("\n[1/5] 加载模型...")
teacher_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, flash_attn=True)
teacher = MiniMindForCausalLM(teacher_config)
base_w = torch.load('../out/full_sft_768.pth', map_location=device)
teacher.load_state_dict(base_w, strict=False)
teacher = teacher.to(device).eval()

# 学生 (Stage 1 权重 + 宽窗口)
head_config = load_head_config('../head_config.json')
student_config = RTPurboConfig(hidden_size=768, num_hidden_layers=8,
                                sparse_attn=True, index_dim=16,
                                local_window_size=256, retrieval_top_p=0.95)
student = RTPurboForCausalLM(student_config, head_config)
s1_w = torch.load('../out/rtpurbo_stage1_768.pth', map_location=device)
student.load_state_dict(s1_w, strict=False)
student = student.to(device).eval()

t_params = sum(p.numel() for p in teacher.parameters()) / 1e6
s_params = sum(p.numel() for p in student.parameters()) / 1e6
print(f"  教师: {t_params:.2f}M | 学生: {s_params:.2f}M (新增 {(s_params-t_params)*1e6:.0f} params)")

# ---- PPL ----
print(f"\n[2/5] Perplexity 对比...")
dataset = load_dataset('json', data_files='../dataset/pretrain_t2t_mini.jsonl', split='train')
indices = np.random.RandomState(123).permutation(len(dataset))[:200]
eval_texts = [str(dataset[int(i)]['text']) for i in indices]

def compute_ppl(model, texts, label):
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in tqdm(texts, desc=label):
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
    return ppl, total_tokens

t_ppl, t_tok = compute_ppl(teacher, eval_texts, "教师PPL")
s_ppl, s_tok = compute_ppl(student, eval_texts, "学生PPL")
gap = (s_ppl - t_ppl) / t_ppl * 100
print(f"\n  教师 PPL: {t_ppl:.2f}")
print(f"  学生 PPL: {s_ppl:.2f}  (差距: {gap:+.2f}%)")

# ---- Token 一致率 ----
print(f"\n[3/5] Token 预测一致率...")
total_agree, top5_agree, total_tokens = 0, 0, 0
with torch.no_grad():
    for text in tqdm(eval_texts[:100], desc="Token一致"):
        tokens = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        ids = tokens['input_ids'].to(device)
        if ids.shape[1] < 10: continue
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            t_logits = teacher(ids).logits[:, :-1, :].float()
            s_logits = student(ids).logits[:, :-1, :].float()
        t_top1 = t_logits.argmax(dim=-1)
        s_top1 = s_logits.argmax(dim=-1)
        total_agree += (t_top1 == s_top1).sum().item()
        t_top5 = torch.topk(t_logits, 5, dim=-1).indices
        top5_agree += (t_top5 == s_top1.unsqueeze(-1)).any(dim=-1).sum().item()
        total_tokens += t_top1.numel()

top1_rate = total_agree / max(total_tokens, 1) * 100
top5_rate = top5_agree / max(total_tokens, 1) * 100
print(f"  Top-1 一致率: {top1_rate:.2f}%")
print(f"  Top-5 一致率: {top5_rate:.2f}%")

# ---- 生成对比 ----
print(f"\n[4/5] 生成质量对比...")
def generate(model, prompt, max_new=150, temp=0.7, top_p=0.85):
    msgs = [{"role": "user", "content": prompt}]
    tmpl = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(tmpl, return_tensors="pt").input_ids.to(device)
    gen = ids
    with torch.no_grad():
        for _ in range(max_new):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = model(gen)
            logits = out.logits[:, -1, :] / temp
            sl, si = torch.sort(logits, descending=True)
            cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            mask = cp - F.softmax(sl, dim=-1) > top_p
            sl[mask] = float('-inf')
            nt = si.gather(-1, torch.multinomial(F.softmax(sl, dim=-1), 1))
            gen = torch.cat([gen, nt], dim=-1)
            if nt.item() == tokenizer.eos_token_id: break
    return tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)

prompts = [
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
for p in prompts:
    print(f"\n  Q: {p}")
    tr = generate(teacher, p)
    sr = generate(student, p)
    print(f"  [全注意力] {tr[:120]}...")
    print(f"  [RTPurbo ] {sr[:120]}...")
    gen_results.append({'prompt': p, 'teacher': tr, 'student': sr})

# ---- 速度 ----
print(f"\n[5/5] 推理速度对比...")
def latency(model, ids, n=20):
    for _ in range(5):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16): model(ids)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16): model(ids)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return sum(ts)/len(ts)*1000

speed = {}
for sl in [128, 256, 512]:
    ids = torch.randint(0, 1000, (1, sl), device=device)
    tm, sm = latency(teacher, ids), latency(student, ids)
    r = tm / max(sm, 0.001)
    print(f"  seq={sl}: 全注意力={tm:.2f}ms, RTPurbo={sm:.2f}ms, 比值={r:.3f}x")
    speed[f'seq_{sl}'] = {'teacher_ms': tm, 'student_ms': sm, 'ratio': r}

# ---- 汇总 ----
print("\n" + "=" * 70)
print("  📊 最终评估汇总")
print("=" * 70)
print(f"  Perplexity:       教师={t_ppl:.2f}  学生={s_ppl:.2f}  差距={gap:+.2f}%")
print(f"  Top-1 Token一致率: {top1_rate:.2f}%")
print(f"  Top-5 Token一致率: {top5_rate:.2f}%")
print(f"  稀疏配置:         window=256, top_p=0.95")
print(f"  稀疏度:           75% local, 25% retrieval")
print(f"  新增参数:         21,504 (0.034%)")
print("=" * 70)

results = {
    'config': {'window': 256, 'top_p': 0.95, 'weights': 'stage1'},
    'perplexity': {'teacher': t_ppl, 'student': s_ppl, 'gap_pct': gap},
    'token_agreement': {'top1': top1_rate, 'top5': top5_rate},
    'generation': gen_results,
    'speed': speed
}
with open('../eval_results_final.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ 结果已保存: ../eval_results_final.json")
