"""
RTPurbo Stage 2: 端到端自蒸馏 (End-to-End Self-distillation)
============================================================
教师模型: 原始全注意力 MiniMind（冻结）
学生模型: RTPurbo 稀疏注意力模型（全参数训练）
损失函数: L = α * CE(labels, student) + (1-α) * T² * KL(teacher || student)

用法:
  # 单卡
  python train_rtpurbo_stage2.py
  # 4卡 DDP
  torchrun --nproc_per_node 4 train_rtpurbo_stage2.py
"""

import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401
import argparse
import time
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
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


def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=2.0, top_k=10):
    """
    蒸馏损失（论文做法）:
      = α * CE(student, labels) + (1-α) * T² * KL(teacher_top10 || student_top10)
    只对齐教师模型的 top-10 logits，避免在不重要的 token 上过拟合。
    """
    # CE loss (hard labels)
    s_logits = student_logits[..., :-1, :].contiguous()
    t_logits = teacher_logits[..., :-1, :].contiguous()
    t_labels = labels[..., 1:].contiguous()
    ce_loss = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), t_labels.view(-1), ignore_index=-100)

    # KL loss: 只对齐 top-K logits（论文 Section 3.3）
    valid_mask = (t_labels != -100)  # (B, T-1)
    if valid_mask.sum() == 0:
        return ce_loss, ce_loss.item(), 0.0

    # 取教师 top-K 的 indices
    _, top_indices = torch.topk(t_logits, top_k, dim=-1)  # (B, T-1, K)

    # 在 top-K 维度上计算 KL
    t_top = torch.gather(t_logits, -1, top_indices) / temperature  # (B, T-1, K)
    s_top = torch.gather(s_logits, -1, top_indices) / temperature

    t_probs = F.softmax(t_top, dim=-1)
    s_log_probs = F.log_softmax(s_top, dim=-1)

    # per-position KL, 然后只在有效位置平均
    kl_per_pos = F.kl_div(s_log_probs, t_probs.detach(), reduction='none').sum(-1)  # (B, T-1)
    kl_loss = (kl_per_pos * valid_mask.float()).sum() / max(valid_mask.float().sum(), 1)
    kl_loss = kl_loss * (temperature ** 2)

    total_loss = alpha * ce_loss + (1 - alpha) * kl_loss
    return total_loss, ce_loss.item(), kl_loss.item()


def train_epoch(epoch, loader, iters, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 教师前向（无梯度）
            with torch.no_grad():
                teacher_out = teacher_model(input_ids)
                teacher_logits = teacher_out.logits

            # 学生前向
            student_out = student_model(input_ids, labels=labels)
            student_logits = student_out.logits

            # 蒸馏损失
            total_loss, ce_val, kl_val = distillation_loss(
                student_logits, teacher_logits, labels,
                alpha=args.alpha, temperature=args.temperature
            )
            total_loss = (total_loss + student_out.aux_loss) / args.accumulation_steps

        scaler.scale(total_loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step, 1) * (iters - step) // 60
            Logger(f'[Stage2] Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'total: {total_loss.item() * args.accumulation_steps:.4f}, '
                   f'ce: {ce_val:.4f}, kl: {kl_val:.4f}, '
                   f'lr: {current_lr:.8f}, eta: {eta_min:.1f}min')
            if wandb:
                wandb.log({
                    "total_loss": total_loss.item() * args.accumulation_steps,
                    "ce_loss": ce_val, "kl_loss": kl_val, "lr": current_lr
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            save_model(epoch, step)

        del input_ids, labels, teacher_out, student_out, total_loss

    # 处理最后未对齐的累积步
    if iters % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


def save_model(epoch, step):
    raw_model = student_model.module if isinstance(student_model, DistributedDataParallel) else student_model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    state_dict = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}
    save_path = f'{args.save_dir}/rtpurbo_stage2_{args.save_tag}_{args.hidden_size}.pth'
    torch.save(state_dict, save_path)
    Logger(f'[Stage2] 保存权重: {save_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTPurbo Stage 2: Self-distillation")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
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
    parser.add_argument("--max_train_steps", type=int, default=500, help="最大训练步数（0=不限制）")
    parser.add_argument("--save_tag", type=str, default="v2", help="保存文件标签")
    parser.add_argument("--freeze_mode", type=str, default="attn",
                        choices=["attn", "index_only"],
                        help="attn=训练注意力层, index_only=只训练index_proj")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="RTPurbo-Stage2")
    args = parser.parse_args()

    # 1. 初始化环境
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)

    Logger("=" * 60)
    Logger("RTPurbo Stage 2: End-to-End Self-distillation")
    Logger("=" * 60)

    # 2. 混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # 3. wandb
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project,
                   name=f"Stage2-a{args.alpha}-T{args.temperature}-lr{args.learning_rate}")

    # 4. 教师模型（原始全注意力，冻结）
    Logger("加载教师模型（全注意力）...")
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
    Logger(f"教师模型: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.2f}M params")

    # 5. 学生模型（RTPurbo 稀疏注意力）
    Logger("加载学生模型（RTPurbo）...")
    head_config = load_head_config(args.head_config)
    rtpurbo_config = RTPurboConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        sparse_attn=True,
        index_dim=args.index_dim,
        local_window_size=args.local_window_size,
        retrieval_top_p=args.retrieval_top_p,
    )
    student_model = RTPurboForCausalLM(rtpurbo_config, head_config)

    # 加载 Stage 1 权重（包含训练好的投影层）
    stage1_path = f'{args.save_dir}/{args.stage1_weight}_{args.hidden_size}.pth'
    if os.path.exists(stage1_path):
        Logger(f"加载 Stage 1 权重: {stage1_path}")
        stage1_weights = torch.load(stage1_path, map_location=args.device)
        student_model.load_state_dict(stage1_weights, strict=False)
    else:
        Logger(f"⚠️ Stage 1 权重不存在，从基线权重初始化: {teacher_weight_path}")
        student_model.load_state_dict(teacher_weights, strict=False)

    student_model = student_model.to(args.device)

    # 5.1 选择性冻结
    if args.freeze_mode == 'index_only':
        # 方案2：只训练 index_proj（和 Stage 1 相同范围，但用蒸馏目标）
        for name, param in student_model.named_parameters():
            if 'index_proj' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        Logger(f"冻结模式: index_only (只训练 index_proj)")
    else:
        # 方案1：训练注意力层（Q/K/V/O + index_proj）
        attn_keywords = ['self_attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj',
                         'q_index_proj', 'k_index_proj']
        for name, param in student_model.named_parameters():
            if any(kw in name for kw in attn_keywords):
                param.requires_grad = True
            else:
                param.requires_grad = False
        Logger(f"冻结模式: attn (训练注意力相关层)")

    student_model.train()

    total_params = sum(p.numel() for p in student_model.parameters())
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    Logger(f"学生模型: {total_params / 1e6:.2f}M params")
    Logger(f"  可训练（注意力层）: {trainable_params / 1e6:.2f}M ({trainable_params/total_params*100:.1f}%)")
    Logger(f"  冻结（embedding/FFN/LM_head）: {frozen_params / 1e6:.2f}M ({frozen_params/total_params*100:.1f}%)")

    # 6. 数据集
    tokenizer = AutoTokenizer.from_pretrained('../model')
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 只传可训练参数给 optimizer
    trainable_parameters = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_parameters, lr=args.learning_rate)

    # 7. DDP
    if dist.is_initialized():
        student_model = DistributedDataParallel(
            student_model, device_ids=[local_rank], find_unused_parameters=True
        )
        student_model._set_static_graph()

    # 8. 训练
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

    # 9. 最终保存
    if is_main_process():
        save_model(args.epochs - 1, -1)
        Logger("[Stage2] ✅ 训练完成！")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
