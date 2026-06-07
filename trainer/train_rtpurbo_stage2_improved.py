"""
RTPurbo Stage 2 Improved: 基于 LoRA 和特征蒸馏的端到端自蒸馏训练
================================================================
教师模型: 原始全注意力 MiniMind（冻结）
学生模型: RTPurbo 稀疏注意力模型（仅训练 LoRA 参数与投影层参数，其余冻结）
损失函数: L = α * CE + (1-α) * T² * KL + β * Feature_MSE
在训练结束后自动融合 LoRA 参数并保存为标准格式权重。

用法:
  # 单卡
  python trainer/train_rtpurbo_stage2_improved.py
  # 多卡 DDP
  torchrun --nproc_per_node 4 trainer/train_rtpurbo_stage2_improved.py
"""

import os
import sys
import math
import time
import argparse
import warnings
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_rtpurbo import RTPurboConfig, RTPurboForCausalLM, load_head_config
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, SkipBatchSampler
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')


# ============================================================
#  原生 LoRA 线性层实现
# ============================================================
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r=16, alpha=32, dropout=0.0):
        super().__init__()
        self.original = original_linear
        # 冻结原始权重
        for p in self.original.parameters():
            p.requires_grad = False
            
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # 旁路权重
        self.lora_A = nn.Parameter(original_linear.weight.new_zeros((r, self.in_features)))
        self.lora_B = nn.Parameter(original_linear.weight.new_zeros((self.out_features, r)))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        orig_out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return orig_out + lora_out * self.scaling

    def merge(self):
        """融合 LoRA 旁路到原始线性层权重中"""
        lora_weight = (self.lora_B @ self.lora_A) * self.scaling
        self.original.weight.data += lora_weight.data


def inject_lora_to_model(model, r=16, alpha=32, dropout=0.0):
    """递归将注意力投影层替换为 LoRALinear"""
    count = 0
    for block in model.model.layers:
        attn = block.self_attn
        attn.q_proj = LoRALinear(attn.q_proj, r=r, alpha=alpha, dropout=dropout)
        attn.k_proj = LoRALinear(attn.k_proj, r=r, alpha=alpha, dropout=dropout)
        attn.v_proj = LoRALinear(attn.v_proj, r=r, alpha=alpha, dropout=dropout)
        attn.o_proj = LoRALinear(attn.o_proj, r=r, alpha=alpha, dropout=dropout)
        count += 4
    return count


def merge_and_save_model(model, save_path):
    """融合 LoRA 权重并存为标准格式"""
    import copy
    model_copy = copy.deepcopy(model)
    
    for block in model_copy.model.layers:
        attn = block.self_attn
        if isinstance(attn.q_proj, LoRALinear):
            attn.q_proj.merge()
            attn.q_proj = attn.q_proj.original
        if isinstance(attn.k_proj, LoRALinear):
            attn.k_proj.merge()
            attn.k_proj = attn.k_proj.original
        if isinstance(attn.v_proj, LoRALinear):
            attn.v_proj.merge()
            attn.v_proj = attn.v_proj.original
        if isinstance(attn.o_proj, LoRALinear):
            attn.o_proj.merge()
            attn.o_proj = attn.o_proj.original
            
    state_dict = {k: v.half().cpu() for k, v in model_copy.state_dict().items()}
    torch.save(state_dict, save_path)
    Logger(f"[Stage2 Improved] 成功合并 LoRA 权重并保存至: {save_path}")


# ============================================================
#  损失函数
# ============================================================
def distillation_loss_improved(student_logits, teacher_logits, labels,
                               student_hidden, teacher_hidden,
                               alpha=0.5, temperature=2.0, beta=10.0, top_k=10):
    """
    改进版损失函数：
      L = α * CE + (1-α) * T² * KL + β * Feature_MSE
    """
    s_logits = student_logits[..., :-1, :].contiguous()
    t_logits = teacher_logits[..., :-1, :].contiguous()
    t_labels = labels[..., 1:].contiguous()
    
    # 1. CE Loss (Hard Labels)
    ce_loss = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), t_labels.view(-1), ignore_index=-100)
    
    valid_mask = (t_labels != -100)
    if valid_mask.sum() == 0:
        return ce_loss, ce_loss.item(), 0.0, 0.0

    # 2. Top-K KL Loss
    _, top_indices = torch.topk(t_logits, top_k, dim=-1)
    t_top = torch.gather(t_logits, -1, top_indices) / temperature
    s_top = torch.gather(s_logits, -1, top_indices) / temperature
    
    t_probs = F.softmax(t_top, dim=-1)
    s_log_probs = F.log_softmax(s_top, dim=-1)
    
    kl_per_pos = F.kl_div(s_log_probs, t_probs.detach(), reduction='none').sum(-1)
    kl_loss = (kl_per_pos * valid_mask.float()).sum() / max(valid_mask.float().sum(), 1)
    kl_loss = kl_loss * (temperature ** 2)
    
    # 3. 中间特征蒸馏损失 (MSE)
    feat_loss = 0.0
    for hs, ht in zip(student_hidden, teacher_hidden):
        # 对齐整个序列在各层隐藏状态上的分布
        feat_loss += F.mse_loss(hs.float(), ht.float().detach())
    feat_loss = feat_loss / max(len(student_hidden), 1)
    
    # 总 Loss
    total_loss = alpha * ce_loss + (1 - alpha) * kl_loss + beta * feat_loss
    return total_loss, ce_loss.item(), kl_loss.item(), feat_loss.item()


# ============================================================
#  训练一轮
# ============================================================
train_loss_history = []

def train_epoch(epoch, loader, iters, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        
        # 学习率调度
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        with autocast_ctx:
            # 教师前向（要求返回隐藏状态，冻结梯度）
            with torch.no_grad():
                teacher_out = teacher_model(input_ids, output_hidden_states=True)
                teacher_logits = teacher_out.logits
                teacher_hidden = teacher_out.hidden_states
                
            # 学生前向（要求返回隐藏状态）
            student_out = student_model(input_ids, labels=labels, output_hidden_states=True)
            student_logits = student_out.logits
            student_hidden = student_out.hidden_states
            
            # 计算改进的蒸馏 Loss
            loss, ce_val, kl_val, feat_val = distillation_loss_improved(
                student_logits, teacher_logits, labels,
                student_hidden, teacher_hidden,
                alpha=args.alpha, temperature=args.temperature, beta=args.beta
            )
            
            # 加上 MOE aux loss (如果有的话)
            total_loss = (loss + student_out.aux_loss) / args.accumulation_steps
            
        scaler.scale(total_loss).backward()
        
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        # 记录日志
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step, 1) * (iters - step) // 60
            loss_val = total_loss.item() * args.accumulation_steps
            
            Logger(f'[Stage2 Improved] Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), '
                   f'loss: {loss_val:.4f}, ce: {ce_val:.4f}, kl: {kl_val:.4f}, feat: {feat_val:.6f}, '
                   f'lr: {current_lr:.8f}, eta: {eta_min:.1f}min')
                   
            if is_main_process():
                train_loss_history.append({
                    "step": epoch * iters + step,
                    "loss": loss_val,
                    "ce": ce_val,
                    "kl": kl_val,
                    "feat": feat_val,
                    "lr": current_lr
                })
                
            if wandb:
                wandb.log({
                    "total_loss": loss_val,
                    "ce_loss": ce_val,
                    "kl_loss": kl_val,
                    "feat_loss": feat_val,
                    "lr": current_lr
                })
                
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            save_path = f'{args.save_dir}/rtpurbo_stage2_improved_{args.save_tag}_{args.hidden_size}.pth'
            merge_and_save_model(student_model, save_path)
            
        del input_ids, labels, teacher_out, student_out, total_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTPurbo Stage 2 Improved: LoRA + Feature Distillation")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4) # LoRA 学习率通常可以设得比全参稍大
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--base_weight", type=str, default="full_sft")
    parser.add_argument("--stage1_weight", type=str, default="rtpurbo_stage1")
    parser.add_argument("--head_config", type=str, default="../head_config.json")
    parser.add_argument("--index_dim", type=int, default=16)
    parser.add_argument("--local_window_size", type=int, default=128)
    parser.add_argument("--retrieval_top_p", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.5, help="CE loss 权重")
    parser.add_argument("--temperature", type=float, default=2.0, help="蒸馏温度")
    parser.add_argument("--beta", type=float, default=20.0, help="特征对齐 MSE Loss 权重")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA 缩放因子")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="最大训练步数 (0=不限制)")
    parser.add_argument("--save_tag", type=str, default="v1", help="保存文件标签")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="RTPurbo-Stage2-Improved")
    args = parser.parse_args()

    # 1. 规范化路径为绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.save_dir = os.path.abspath(os.path.join(script_dir, args.save_dir))
    args.data_path = os.path.abspath(os.path.join(script_dir, args.data_path))
    args.head_config = os.path.abspath(os.path.join(script_dir, args.head_config))
    
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)

    Logger("=" * 60)
    Logger("RTPurbo Stage 2 Improved: LoRA + Feature Distillation")
    Logger("=" * 60)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project,
                   name=f"Stage2-LoRA-r{args.lora_r}-beta{args.beta}-lr{args.learning_rate}")

    # 2. 教师模型 (全注意力，冻结)
    Logger("加载教师模型 (全注意力)...")
    teacher_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        flash_attn=True
    )
    teacher_model = MiniMindForCausalLM(teacher_config)
    moe_suffix = '_moe' if teacher_config.use_moe else ''
    teacher_weight_path = f'{args.save_dir}/{args.base_weight}_{args.hidden_size}{moe_suffix}.pth'
    
    teacher_weights = torch.load(teacher_weight_path, map_location=args.device)
    teacher_model.load_state_dict(teacher_weights, strict=False)
    teacher_model = teacher_model.to(args.device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # 3. 学生模型 (RTPurbo 稀疏注意力)
    Logger("加载学生模型 (RTPurbo)...")
    head_config_data = load_head_config(args.head_config)
    rtpurbo_config = RTPurboConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        sparse_attn=True,
        index_dim=args.index_dim,
        local_window_size=args.local_window_size,
        retrieval_top_p=args.retrieval_top_p,
    )
    student_model = RTPurboForCausalLM(rtpurbo_config, head_config_data)

    # 载入 Stage 1 的投影初始化
    stage1_path = f'{args.save_dir}/{args.stage1_weight}_{args.hidden_size}.pth'
    if os.path.exists(stage1_path):
        Logger(f"加载 Stage 1 投影权重: {stage1_path}")
        stage1_weights = torch.load(stage1_path, map_location=args.device)
        student_model.load_state_dict(stage1_weights, strict=False)
    else:
        Logger(f"⚠️ Stage 1 权重不存在，从教师模型参数初始化")
        student_model.load_state_dict(teacher_weights, strict=False)

    student_model = student_model.to(args.device)

    # 4. 注入 LoRA 到学生模型中
    lora_count = inject_lora_to_model(student_model, r=args.lora_r, alpha=args.lora_alpha, dropout=0.05)
    Logger(f"成功注入 LoRA 到注意力层中，共替换 {lora_count} 个 Linear 层")

    # 5. 冻结除 LoRA 参数和 index_proj 外的所有参数
    for name, param in student_model.named_parameters():
        if 'lora_' in name or 'index_proj' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 统计可训练参数
    total_params = sum(p.numel() for p in student_model.parameters())
    trainable_params_count = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    Logger(f"模型总参数量: {total_params / 1e6:.2f}M")
    Logger(f"可训练参数量 (LoRA + index_proj): {trainable_params_count / 1e6:.4f}M ({trainable_params_count/total_params*100:.3f}%)")

    student_model.train()

    # 6. 数据集与优化器
    tokenizer = AutoTokenizer.from_pretrained(os.path.abspath(os.path.join(script_dir, '../model')))
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    trainable_parameters = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_parameters, lr=args.learning_rate)

    if dist.is_initialized():
        student_model = DistributedDataParallel(
            student_model, device_ids=[local_rank], find_unused_parameters=True
        )
        student_model._set_static_graph()

    # 7. 训练过程
    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, 0)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)

        total_steps = len(loader)
        if args.max_train_steps > 0:
            total_steps = min(total_steps, args.max_train_steps)
            Logger(f"限制最大训练步数: {total_steps}")

            class LimitedLoader:
                def __init__(self, loader, max_steps):
                    self.loader = loader
                    self.max_steps = max_steps
                def __iter__(self):
                    for i, batch in enumerate(self.loader):
                        if i >= self.max_steps:
                            break
                        yield batch
                def __len__(self):
                    return self.max_steps

            loader = LimitedLoader(loader, total_steps)

        train_epoch(epoch, loader, total_steps, wandb)

    # 8. 保存训练 Loss 数据
    if is_main_process():
        save_path = f'{args.save_dir}/rtpurbo_stage2_improved_{args.save_tag}_{args.hidden_size}.pth'
        merge_and_save_model(student_model, save_path)
        
        # 记录 loss 曲线作为数据支撑
        curves_path = f'{args.save_dir}/stage2_improved_curves.json'
        with open(curves_path, 'w', encoding='utf-8') as f:
            json.dump({"loss_history": train_loss_history}, f, indent=4)
        Logger(f"[Stage2 Improved] 过程数据已保存至: {curves_path}")
        Logger("[Stage2 Improved] ✅ 训练全部完成！")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
