"""
RTPurbo Stage 1: 低维投影训练 (Low-dimension Projection Training)
================================================================
冻结原模型所有参数，只训练 q_index_proj 和 k_index_proj。
损失函数: MSE(低维注意力分数, 全注意力分数)
目标: 让低维空间的注意力分布逼近原始全注意力分布。

用法:
  # 单卡
  python train_rtpurbo_stage1.py
  # 4卡 DDP
  torchrun --nproc_per_node 4 train_rtpurbo_stage1.py
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
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from dataset.lm_dataset import PretrainDataset
from model.model_rtpurbo import RTPurboConfig, RTPurboForCausalLM, load_head_config, create_rtpurbo_model
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, SkipBatchSampler
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 前向传播（启用注意力分数保存）
            res = model(input_ids)

            # Stage 1 核心: 低维投影 MSE loss
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            proj_loss = raw_model.get_projection_loss()
            loss = proj_loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = proj_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step, 1) * (iters - step) // 60
            Logger(f'[Stage1] Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'proj_loss: {current_loss:.6f}, lr: {current_lr:.8f}, '
                   f'eta: {eta_min:.1f}min')
            if wandb:
                wandb.log({"proj_loss": current_loss, "lr": current_lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            save_model(epoch, step)

        del input_ids, res, loss, proj_loss

    # 处理最后未对齐的累积步
    if iters % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


def save_model(epoch, step):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)

    # 只保存 index_proj 参数
    proj_state = {}
    for name, param in raw_model.named_parameters():
        if 'index_proj' in name:
            proj_state[name] = param.half().cpu()

    save_path = f'{args.save_dir}/rtpurbo_stage1_proj_{args.save_tag}_{args.hidden_size}.pth' if args.save_tag else f'{args.save_dir}/rtpurbo_stage1_proj_{args.hidden_size}.pth'
    torch.save(proj_state, save_path)
    Logger(f'[Stage1] 保存投影权重: {save_path} ({len(proj_state)} params)')

    # 也保存完整模型
    full_state = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}
    full_path = f'{args.save_dir}/rtpurbo_stage1_{args.save_tag}_{args.hidden_size}.pth' if args.save_tag else f'{args.save_dir}/rtpurbo_stage1_{args.hidden_size}.pth'
    torch.save(full_state, full_path)
    Logger(f'[Stage1] 保存完整权重: {full_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTPurbo Stage 1: Projection Training")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
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
    parser.add_argument("--head_config", type=str, default="../head_config.json")
    parser.add_argument("--index_dim", type=int, default=16)
    parser.add_argument("--local_window_size", type=int, default=128)
    parser.add_argument("--retrieval_top_p", type=float, default=0.9)
    parser.add_argument("--max_train_steps", type=int, default=200, help="最大训练步数（0=不限制）")
    parser.add_argument("--save_tag", type=str, default="", help="保存文件标签")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="RTPurbo-Stage1")
    args = parser.parse_args()

    # 1. 初始化环境
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)

    Logger("=" * 60)
    Logger("RTPurbo Stage 1: Low-dimension Projection Training")
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
                   name=f"Stage1-lr{args.learning_rate}-bs{args.batch_size}")

    # 4. 创建 RTPurbo 模型并加载基线权重
    head_config = load_head_config(args.head_config)
    moe_suffix = '_moe' if False else ''
    base_weight_path = f'{args.save_dir}/{args.base_weight}_{args.hidden_size}{moe_suffix}.pth'
    Logger(f"基线权重: {base_weight_path}")
    Logger(f"Head 配置: {args.head_config}")

    model, _ = create_rtpurbo_model(
        base_weight_path, args.head_config, device=args.device,
        hidden_size=args.hidden_size,
        index_dim=args.index_dim,
        local_window_size=args.local_window_size,
        retrieval_top_p=args.retrieval_top_p,
    )

    # 5. 冻结除 index_proj 外的所有参数
    for name, param in model.named_parameters():
        if 'index_proj' not in name:
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    Logger(f"模型参数: {total_params / 1e6:.2f}M, 可训练参数: {total_trainable} "
           f"({total_trainable / total_params * 100:.4f}%)")

    # 启用注意力分数保存
    model.set_save_attn_scores(True)

    # 6. 数据集和加载器
    model_path = '../model'
    if not os.path.exists(model_path):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model'))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)

    # 7. DDP
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                         find_unused_parameters=True)
        model._set_static_graph()

    # 8. 训练
    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, 0)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)

        # 限制最大训练步数
        total_steps = len(loader)
        if args.max_train_steps > 0:
            total_steps = min(total_steps, args.max_train_steps)
            Logger(f"限制最大训练步数: {total_steps}")

            # 包装 loader 以限制步数
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
        Logger("[Stage1] ✅ 训练完成！")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
