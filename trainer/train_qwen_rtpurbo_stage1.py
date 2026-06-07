"""
RTPurbo Stage 1: 低维投影训练 (Qwen3.5-4B 适配版)
================================================
冻结 Qwen3.5-4B 原模型的所有参数，只训练 q_index_proj 和 k_index_proj。
损失函数: MSE(低维注意力概率, 全注意力概率)
目标: 极速逼近原本的高维全注意力分布，支持清华 LongAlign-10k 长文本。
"""

import os
import sys
import json
import argparse
import time
import warnings
import torch
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e4m3fn", torch.float32)
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.lm_dataset import PretrainDataset
from model.model_qwen_rtpurbo import convert_qwen_to_rtpurbo, set_save_attn_scores, get_projection_loss
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, SkipBatchSampler
from transformers import AutoTokenizer, AutoModelForImageTextToText
from datasets import load_dataset

warnings.filterwarnings('ignore')


# 封装鲁棒的 PretrainDataset，支持本地 JSONL 数据
class RobustPretrainDataset(PretrainDataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        super().__init__(data_path, tokenizer, max_length)

    def __getitem__(self, index):
        sample = self.samples[index]
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        bos_list = [bos_id] if bos_id is not None else []
        eos_list = [eos_id] if eos_id is not None else []

        text_tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=self.max_length - len(bos_list) - len(eos_list),
            truncation=True
        ).input_ids

        tokens = bos_list + text_tokens + eos_list
        input_ids = tokens + [pad_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        labels = input_ids.clone()
        labels[input_ids == pad_id] = -100
        return input_ids, labels


# 天然长文本 SFT 格式数据集转化为预训练语料的包装类
class LongAlignPretrainDataset(Dataset):
    def __init__(self, dataset_list, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = dataset_list
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        bos_list = [bos_id] if bos_id is not None else []
        eos_list = [eos_id] if eos_id is not None else []

        # 防御性合并多轮对话或长文本为纯长文档进行训练
        text_parts = []
        for msg in sample['messages']:
            if 'content' in msg:
                text_parts.append(msg['content'])
            elif 'value' in msg:
                text_parts.append(msg['value'])
        text = "\n".join(text_parts)

        text_tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length - len(bos_list) - len(eos_list),
            truncation=True
        ).input_ids

        tokens = bos_list + text_tokens + eos_list
        input_ids = tokens + [pad_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        labels = input_ids.clone()
        labels[input_ids == pad_id] = -100
        return input_ids, labels


def train_epoch(epoch, loader, iters, wandb=None, csv_log_path=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 前向传播
            res = model(input_ids)

            # 获取 QwenRTPurbo 稀疏注意力产生的低维与高维 MSE 损失
            proj_loss = get_projection_loss(model)
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

            if csv_log_path and is_main_process():
                try:
                    import csv
                    with open(csv_log_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, step, current_loss, current_lr, eta_min])
                except Exception as e:
                    Logger(f'[Warning] 写入 CSV 日志失败: {e}')

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

    # 1. 仅提取 index_proj 的权重进行保存
    proj_state = {}
    for name, param in raw_model.named_parameters():
        if 'index_proj' in name:
            proj_state[name] = param.half().cpu()

    proj_path = f'{args.save_dir}/rtpurbo_stage1_proj_qwen_{args.save_tag}.pth' if args.save_tag else f'{args.save_dir}/rtpurbo_stage1_proj_qwen.pth'
    torch.save(proj_state, proj_path)
    Logger(f'[Stage1] 保存投影权重: {proj_path} ({len(proj_state)} params)')

    # 2. 保存完整模型权重（包含 RTPurbo Attention 及 Projection 参数，用于 Stage 2 初始化）
    full_state = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}
    full_path = f'{args.save_dir}/rtpurbo_stage1_qwen_{args.save_tag}.pth' if args.save_tag else f'{args.save_dir}/rtpurbo_stage1_qwen.pth'
    torch.save(full_state, full_path)
    Logger(f'[Stage1] 保存完整权重: {full_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.5 RTPurbo Stage 1: Projection Training")
    parser.add_argument("--model_path", type=str, default="../model/Qwen3.5-4B", help="模型文件夹路径")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)  # 4B 模型，防止 OOM
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--data_path", type=str, default="THUDM/LongAlign-10k")
    parser.add_argument("--head_config", type=str, default="../qwen_head_config.json")
    parser.add_argument("--index_dim", type=int, default=16)
    parser.add_argument("--local_window_size", type=int, default=128)
    parser.add_argument("--retrieval_top_p", type=float, default=0.9)
    parser.add_argument("--max_train_steps", type=int, default=100, help="最大训练步数")
    parser.add_argument("--save_tag", type=str, default="", help="额外保存名称标记")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Qwen-RTPurbo-Stage1")
    parser.add_argument("--csv_log", type=str, default=None, help="本地 CSV 日志文件路径")
    args = parser.parse_args()

    # 1. 启动分布式环境
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)

    # 初始化本地 CSV 训练日志
    csv_log_path = args.csv_log
    if csv_log_path is None:
        csv_log_path = f'{args.save_dir}/stage1_training_log.csv'
    if is_main_process():
        os.makedirs(os.path.dirname(os.path.abspath(csv_log_path)), exist_ok=True)
        try:
            import csv
            with open(csv_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'step', 'proj_loss', 'lr', 'eta_min'])
        except Exception as e:
            Logger(f'[Warning] 初始化 CSV 日志文件失败: {e}')

    Logger("=" * 60)
    Logger("Qwen3.5 RTPurbo Stage 1: Low-dimension Projection Training")
    Logger("=" * 60)

    # 2. 混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # 3. 统计
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project,
                   name=f"Qwen-Stage1-lr{args.learning_rate}-bs{args.batch_size}")

    # 4. 加载预训练模型并注册 Tokenizer
    Logger(f"加载 Qwen3.5 预训练模型：{args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="eager"
    ).to(args.device)

    # 5. 读取 Head 标定并执行动态替换
    Logger(f"加载 Head 配置：{args.head_config}")
    with open(args.head_config, 'r', encoding='utf-8') as f:
        head_config = json.load(f)

    model = convert_qwen_to_rtpurbo(
        base_model,
        head_config,
        index_dim=args.index_dim,
        local_window_size=args.local_window_size,
        retrieval_top_p=args.retrieval_top_p,
        sparse_attn=True
    )

    # 6. 冻结除投影层 index_proj 之外的所有参数
    for name, param in model.named_parameters():
        if 'index_proj' not in name:
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    Logger(f"模型总参数量: {total_params / 1e6:.2f}M, 可训练参数量: {total_trainable} "
           f"({total_trainable / total_params * 100:.6f}%)")

    # 激活中间注意力保存功能
    set_save_attn_scores(model, True)

    # 7. 双轨数据集流式/离线加载
    if args.data_path == "THUDM/LongAlign-10k":
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        Logger("开启国内高速 HuggingFace 镜像，拉取 LongAlign-10k 语料...")
        raw_dataset = load_dataset('THUDM/LongAlign-10k', split='train', streaming=True)
        dataset_list = []
        # 按需拉取 1000 条
        limit_samples = 1000
        # 仅在主进程打印拉取进度
        if is_main_process():
            for item in tqdm(raw_dataset, total=limit_samples, desc="缓存天然长文本"):
                dataset_list.append(item)
                if len(dataset_list) >= limit_samples:
                    break
        else:
            for item in raw_dataset:
                dataset_list.append(item)
                if len(dataset_list) >= limit_samples:
                    break
        train_ds = LongAlignPretrainDataset(dataset_list, tokenizer, max_length=args.max_seq_len)
    else:
        train_ds = RobustPretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)

    # 8. DDP
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                         find_unused_parameters=True)
        model._set_static_graph()

    # 9. 训练循环
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
            Logger(f"最大训练步数限制为: {total_steps}")

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

        train_epoch(epoch, loader, total_steps, wandb, csv_log_path)

    # 10. 最终保存
    if is_main_process():
        save_model(args.epochs - 1, -1)
        Logger("[Stage1] ✅ 训练完成！")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
