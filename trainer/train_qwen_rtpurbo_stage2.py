"""
RTPurbo Stage 2: 端到端自蒸馏 (Qwen3.5-4B 适配版)
===============================================
教师模型: 原始全注意力 Qwen3.5-4B 模型（冻结）
学生模型: RTPurbo 稀疏注意力 Qwen3.5-4B 模型（仅训练注意力、归一化及投影相关层）
损失函数: Hard Label CE + KL(Teacher_logits_Top10 || Student_logits_Top10) 蒸馏约束
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
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainer.train_qwen_rtpurbo_stage1 import RobustPretrainDataset
from model.model_qwen_rtpurbo import convert_qwen_to_rtpurbo
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, SkipBatchSampler
from transformers import AutoTokenizer, AutoModelForImageTextToText
from datasets import load_dataset
from dataset.lm_dataset import SFTDataset

warnings.filterwarnings('ignore')


class LongAlignSFTDataset(Dataset):
    """适配 THUDM/LongAlign-10k 天然长文本 SFT 格式的数据集"""
    def __init__(self, dataset_list, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = dataset_list
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 匹配助理回复的边界 Token 以计算 labels 掩码
        if '<|im_start|>' in self.tokenizer.get_vocab():
            self.bos_id = self.tokenizer('<|im_start|>assistant\n', add_special_tokens=False).input_ids
            self.eos_id = self.tokenizer('<|im_end|>\n', add_special_tokens=False).input_ids
        else:
            self.bos_id = self.tokenizer(f'{self.tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
            self.eos_id = self.tokenizer(f'{self.tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        messages = sample['messages']
        
        # 兼容处理不同的键名（如 'from'/'value' 或 'role'/'content'）
        formatted_messages = []
        for msg in messages:
            if 'from' in msg:
                role = 'user' if msg['from'] == 'human' else 'assistant'
                content = msg.get('value', '')
            else:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
            formatted_messages.append({'role': role, 'content': content})

        # 运用 chat_template
        prompt = self.tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=False)
        
        # 完整编码
        input_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        
        # 智能切片以保留助理回复
        if len(input_ids) > self.max_length:
            bos_indices = []
            i = 0
            while i < len(input_ids) - len(self.bos_id) + 1:
                if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                    bos_indices.append(i)
                    i += len(self.bos_id)
                else:
                    i += 1
            
            if bos_indices:
                last_bos = bos_indices[-1]
                # 让 last_bos 位于切片的第 768 个 token 位置
                start_slice = max(0, last_bos - 768)
                end_slice = start_slice + self.max_length
                input_ids = input_ids[start_slice:end_slice]
            else:
                input_ids = input_ids[:self.max_length]

        if len(input_ids) < self.max_length:
            input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        labels = self.generate_labels(input_ids)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels


def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=2.0, top_k=10):
    """
    自蒸馏损失:
      L = α * CE(student, labels) + (1-α) * T² * KL(teacher_top10 || student_top10)
    """
    s_logits = student_logits[..., :-1, :].contiguous()
    t_logits = teacher_logits[..., :-1, :].contiguous()
    t_labels = labels[..., 1:].contiguous()
    ce_loss = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), t_labels.view(-1), ignore_index=-100)

    valid_mask = (t_labels != -100)
    if valid_mask.sum() == 0:
        return ce_loss, ce_loss.item(), 0.0

    _, top_indices = torch.topk(t_logits, top_k, dim=-1)

    t_top = torch.gather(t_logits, -1, top_indices) / temperature
    s_top = torch.gather(s_logits, -1, top_indices) / temperature

    t_probs = F.softmax(t_top, dim=-1)
    s_log_probs = F.log_softmax(s_top, dim=-1)

    kl_per_pos = F.kl_div(s_log_probs, t_probs.detach(), reduction='none').sum(-1)
    kl_loss = (kl_per_pos * valid_mask.float()).sum() / max(valid_mask.float().sum(), 1)
    kl_loss = kl_loss * (temperature ** 2)

    total_loss = alpha * ce_loss + (1 - alpha) * kl_loss
    return total_loss, ce_loss.item(), kl_loss.item()


def train_epoch(epoch, loader, iters, wandb=None, csv_log_path=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 教师前向传播（完全冻结）
            with torch.no_grad():
                teacher_device = next(teacher_model.parameters()).device
                teacher_out = teacher_model(input_ids.cpu().to(teacher_device))
                teacher_logits = teacher_out.logits.cpu().to(args.device)

            # 学生前向传播（稀疏注意力，参与训练）
            student_out = student_model(input_ids, labels=labels)
            student_logits = student_out.logits

            # 蒸馏损失计算
            total_loss, ce_val, kl_val = distillation_loss(
                student_logits, teacher_logits, labels,
                alpha=args.alpha, temperature=args.temperature
            )
            total_loss = total_loss / args.accumulation_steps

        scaler.scale(total_loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step, 1) * (iters - step) // 60
            tot_loss_val = total_loss.item() * args.accumulation_steps
            Logger(f'[Stage2] Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'total: {tot_loss_val:.4f}, '
                   f'ce: {ce_val:.4f}, kl: {kl_val:.4f}, '
                   f'lr: {current_lr:.8f}, eta: {eta_min:.1f}min')
            if wandb:
                wandb.log({
                    "total_loss": tot_loss_val,
                    "ce_loss": ce_val, "kl_loss": kl_val, "lr": current_lr
                })

            if csv_log_path and is_main_process():
                try:
                    import csv
                    with open(csv_log_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, step, tot_loss_val, ce_val, kl_val, current_lr, eta_min])
                except Exception as e:
                    Logger(f'[Warning] 写入 CSV 日志失败: {e}')

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            save_model(epoch, step)

        del input_ids, labels, teacher_out, student_out, total_loss

    # 尾部对齐
    if iters % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


def save_model(epoch, step):
    raw_model = student_model.module if isinstance(student_model, DistributedDataParallel) else student_model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    
    # 仅保存可训练参数以极大缩减写盘开销和避免WSL内存溢出
    trainable_names = {name for name, param in raw_model.named_parameters() if param.requires_grad}
    state_dict = {k: v.half().cpu() for k, v in raw_model.state_dict().items() if k in trainable_names}
    
    save_path = f'{args.save_dir}/rtpurbo_stage2_qwen_{args.save_tag}.pth' if args.save_tag else f'{args.save_dir}/rtpurbo_stage2_qwen.pth'
    torch.save(state_dict, save_path)
    Logger(f'[Stage2] 保存微调参数权重 (包含 {len(state_dict)} 个键): {save_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.5 RTPurbo Stage 2: Self-distillation")
    parser.add_argument("--model_path", type=str, default="../model/Qwen3.5-4B")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)  # 4B 限制 batch_size 避免 OOM
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher_device", type=str, default=None, help="教师模型运行的设备，如果为 None 且存在多卡则自适应选择")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--data_path", type=str, default="THUDM/LongAlign-10k")
    parser.add_argument("--stage1_weight", type=str, default="../out/rtpurbo_stage1_qwen.pth")
    parser.add_argument("--head_config", type=str, default="../qwen_head_config.json")
    parser.add_argument("--index_dim", type=int, default=16)
    parser.add_argument("--local_window_size", type=int, default=128)
    parser.add_argument("--retrieval_top_p", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.5, help="Hard CE 损失比例")
    parser.add_argument("--temperature", type=float, default=2.0, help="蒸馏温度")
    parser.add_argument("--max_train_steps", type=int, default=200)
    parser.add_argument("--save_tag", type=str, default="", help="保存标记")
    parser.add_argument("--freeze_mode", type=str, default="attn",
                        choices=["attn", "index_only"],
                        help="attn=微调注意力及索引投影层, index_only=仅微调投影投影层")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Qwen-RTPurbo-Stage2")
    parser.add_argument("--csv_log", type=str, default=None, help="本地 CSV 日志文件路径")
    args = parser.parse_args()

    # 1. 分布式环境
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)

    # 初始化本地 CSV 训练日志
    csv_log_path = args.csv_log
    if csv_log_path is None:
        csv_log_path = f'{args.save_dir}/stage2_training_log.csv'
    if is_main_process():
        os.makedirs(os.path.dirname(os.path.abspath(csv_log_path)), exist_ok=True)
        try:
            import csv
            with open(csv_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'step', 'total_loss', 'ce_loss', 'kl_loss', 'lr', 'eta_min'])
        except Exception as e:
            Logger(f'[Warning] 初始化 CSV 日志文件失败: {e}')

    # 确定教师模型的运行设备
    teacher_device = args.teacher_device
    if teacher_device is None:
        if "cuda" in args.device:
            try:
                device_idx = int(args.device.split(":")[-1])
                num_gpus = torch.cuda.device_count()
                if num_gpus > 1:
                    teacher_device = f"cuda:{(device_idx + 1) % num_gpus}"
                else:
                    teacher_device = args.device
            except:
                teacher_device = args.device
        else:
            teacher_device = args.device

    Logger("=" * 60)
    Logger("Qwen3.5 RTPurbo Stage 2: End-to-End Self-distillation")
    Logger("=" * 60)

    # 2. 混合精度设置
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # 3. 统计
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project,
                   name=f"Qwen-Stage2-lr{args.learning_rate}")

    # 4. 加载教师模型（全注意力，完全冻结）
    Logger(f"加载教师模型（原生全注意力，保持冻结，运行设备：{teacher_device}）...")
    teacher_model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="eager"
    ).to(teacher_device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # 5. 加载学生模型（RTPurbo，进行参数转换并从 Stage 1 初始化）
    Logger("加载学生模型并将其转换为 RTPurbo 结构...")
    student_model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="eager"
    )
    with open(args.head_config, 'r', encoding='utf-8') as f:
        head_config = json.load(f)

    student_model = convert_qwen_to_rtpurbo(
        student_model,
        head_config,
        index_dim=args.index_dim,
        local_window_size=args.local_window_size,
        retrieval_top_p=args.retrieval_top_p,
        sparse_attn=True
    )

    if os.path.exists(args.stage1_weight):
        Logger(f"加载 Stage 1 投影权重进行初始化：{args.stage1_weight}")
        stage1_weights = torch.load(args.stage1_weight, map_location=args.device)
        missing, unexpected = student_model.load_state_dict(stage1_weights, strict=False)
        Logger(f"加载完毕，缺失键数：{len(missing)}，多余键数：{len(unexpected)}")
    else:
        Logger("⚠️ 未检测到 Stage 1 的模型文件，使用基线权重直接初始化（投影随机初始化）。")

    student_model = student_model.to(args.device)

    # 5.2 开启梯度检查点与输入求导以节省显存并支持反向传播
    if hasattr(student_model, "enable_input_require_grads"):
        try:
            student_model.enable_input_require_grads()
            Logger("学生模型已启用输入梯度要求 (enable_input_require_grads)")
        except Exception as e:
            Logger(f"启用输入梯度要求失败: {e}")

    try:
        student_model.gradient_checkpointing_enable()
        Logger("学生模型成功开启 Gradient Checkpointing")
    except Exception as e:
        Logger(f"开启 Gradient Checkpointing 失败: {e}")
        try:
            if hasattr(student_model, "model") and hasattr(student_model.model, "language_model"):
                student_model.model.language_model.gradient_checkpointing_enable()
                Logger("学生模型的 language_model 成功开启 Gradient Checkpointing")
        except Exception as ex:
            Logger(f"底层 language_model 开启 Gradient Checkpointing 依然失败: {ex}")

    # 5.1 冻结参数策略
    if args.freeze_mode == 'index_only':
        for name, param in student_model.named_parameters():
            if 'index_proj' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        Logger("冻结策略: index_only (仅训练投影投影参数)")
    else:
        # 微调所有 attention 相关线性层、归一化层与投影参数
        # 包含了 q_norm 与 k_norm 以契合 Qwen3.5
        attn_keywords = ['self_attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj',
                         'q_norm', 'k_norm', 'q_index_proj', 'k_index_proj']
        for name, param in student_model.named_parameters():
            if any(kw in name for kw in attn_keywords):
                param.requires_grad = True
            else:
                param.requires_grad = False
        Logger("冻结策略: attn (微调注意力模块的所有线性映射、层归一化与投影层)")

    student_model.train()

    total_params = sum(p.numel() for p in student_model.parameters())
    trainable_parameters = [p for p in student_model.parameters() if p.requires_grad]
    trainable_params_count = sum(p.numel() for p in trainable_parameters)
    Logger(f"学生模型总参数量: {total_params / 1e6:.2f}M, 可训练参数量: {trainable_params_count / 1e6:.2f}M "
           f"({trainable_params_count / total_params * 100:.4f}%)")

    # 6. 数据集加载
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.data_path == "THUDM/LongAlign-10k":
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        Logger("开启国内高速 HuggingFace 镜像，拉取 LongAlign-10k SFT 数据中...")
        raw_dataset = load_dataset('THUDM/LongAlign-10k', split='train', streaming=True)
        dataset_list = []
        limit_samples = 1000
        if is_main_process():
            for item in tqdm(raw_dataset, total=limit_samples, desc="缓存天然长文本 SFT"):
                dataset_list.append(item)
                if len(dataset_list) >= limit_samples:
                    break
        else:
            for item in raw_dataset:
                dataset_list.append(item)
                if len(dataset_list) >= limit_samples:
                    break
        train_ds = LongAlignSFTDataset(dataset_list, tokenizer, max_length=args.max_seq_len)
    else:
        train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(trainable_parameters, lr=args.learning_rate)

    # 7. DDP
    if dist.is_initialized():
        student_model = DistributedDataParallel(
            student_model, device_ids=[local_rank], find_unused_parameters=True
        )
        student_model._set_static_graph()

    # 8. 训练循环
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

    # 9. 最终保存
    if is_main_process():
        save_model(args.epochs - 1, -1)
        Logger("[Stage2] ✅ 端到端自蒸馏 SFT 训练完成！")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
