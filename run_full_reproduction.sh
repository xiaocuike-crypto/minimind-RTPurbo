#!/bin/bash
# ==============================================================================
# Qwen3.5-4B RTPurbo 2048 长度一键全链路复现与数据归档脚本
# ==============================================================================
# 硬件调配说明:
# 鉴于当前物理 GPU 显存占用情况，我们将设备做如下安全隔离分配以防 OOM：
# - 标定、Stage1、及单模型运行设备：统一使用 cuda:3 (空闲 30GB)
# - 自蒸馏双模型及推理评测双侧分流设备：
#   - 学生端 (Student): cuda:3 (空闲 30GB，运行训练与推理)
#   - 教师端 (Teacher): cuda:1 (空闲 23.8GB，运行冻结的前向评估)
# ==============================================================================

set -e

# 创建专用的 2048 实验结果归档目录
OUT_DIR="/mnt/d/out_rtpurbo_2048"
mkdir -p "$OUT_DIR"

PYTHON_EXEC="/home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python"
MODEL_PATH="/mnt/d/minimind-RTPurbo/model/Qwen3.5-4B"
HEAD_CONFIG="$OUT_DIR/qwen_head_config_2048.json"
STAGE1_WEIGHT="$OUT_DIR/rtpurbo_stage1_proj_qwen_2048.pth"
STAGE2_WEIGHT="$OUT_DIR/rtpurbo_stage2_qwen_2048.pth"

echo "=============================================================="
echo "🚀 开始执行 RTPurbo 2048 长度全流程实验与数据落盘流程"
echo "=============================================================="
echo "输出与日志归档目录: $OUT_DIR"
echo "=============================================================="

# ------------------------------------------------------------------------------
# 步骤 1：2048 长度下的注意力头标定
# ------------------------------------------------------------------------------
echo "👉 [1/6] 运行 2048 长度注意力头离线标定..."
$PYTHON_EXEC /mnt/d/minimind-RTPurbo/scripts/calibrate_qwen_heads.py \
  --model_path "$MODEL_PATH" \
  --output "$HEAD_CONFIG" \
  --max_seq_len 2048 \
  --num_samples 20 \
  --device "cuda:3"

# ------------------------------------------------------------------------------
# 步骤 2：Stage 1 投影层特征对齐训练
# ------------------------------------------------------------------------------
echo "👉 [2/6] 运行 Stage 1 投影层训练 (100 步)..."
$PYTHON_EXEC /mnt/d/minimind-RTPurbo/trainer/train_qwen_rtpurbo_stage1.py \
  --model_path "$MODEL_PATH" \
  --head_config "$HEAD_CONFIG" \
  --max_seq_len 2048 \
  --max_train_steps 100 \
  --save_dir "$OUT_DIR" \
  --save_tag "2048" \
  --csv_log "$OUT_DIR/stage1_training_log.csv" \
  --device "cuda:3"

# ------------------------------------------------------------------------------
# 步骤 3：Stage 2 端到端自蒸馏微调
# ------------------------------------------------------------------------------
echo "👉 [3/6] 运行 Stage 2 端到端自蒸馏微调 (200 步)..."
# 使用双卡分流: 教师使用 cuda:1, 学生使用 cuda:3，batch_size 减小为 1 且梯度累积增加到 32 以防 2048 长度 OOM
$PYTHON_EXEC /mnt/d/minimind-RTPurbo/trainer/train_qwen_rtpurbo_stage2.py \
  --model_path "$MODEL_PATH" \
  --head_config "$HEAD_CONFIG" \
  --stage1_weight "$OUT_DIR/rtpurbo_stage1_qwen_2048.pth" \
  --max_seq_len 2048 \
  --max_train_steps 200 \
  --save_dir "$OUT_DIR" \
  --save_tag "2048" \
  --csv_log "$OUT_DIR/stage2_training_log.csv" \
  --device "cuda:3" \
  --teacher_device "cuda:1" \
  --batch_size 1 \
  --accumulation_steps 32

# ------------------------------------------------------------------------------
# 步骤 4：评估 2048 时延性能与短文本生成
# ------------------------------------------------------------------------------
echo "👉 [4/6] 评估 2048 长度下的时延性能与文本生成..."
# 暂用单卡 cuda:3 运行对比
$PYTHON_EXEC /mnt/d/minimind-RTPurbo/scripts/eval_qwen_rtpurbo.py \
  --model_path "$MODEL_PATH" \
  --head_config "$HEAD_CONFIG" \
  --weight_path "$STAGE2_WEIGHT" \
  --output "$OUT_DIR/qwen_eval_results_2048.json" \
  --device "cuda:3"

# ------------------------------------------------------------------------------
# 步骤 5：大海捞针 (Needle in a Haystack) 2048 检索评测
# ------------------------------------------------------------------------------
echo "👉 [5/6] 运行 2048 长度大海捞针测试..."
# 分流运行：学生 cuda:3, 教师 cuda:1
$PYTHON_EXEC /mnt/d/minimind-RTPurbo/scripts/needle_in_haystack.py \
  --model_path "$MODEL_PATH" \
  --head_config "$HEAD_CONFIG" \
  --weight_path "$STAGE2_WEIGHT" \
  --output "$OUT_DIR/needle_in_haystack_results.json" \
  --device "cuda:3" \
  --teacher_device "cuda:1"

# ------------------------------------------------------------------------------
# 步骤 6：100 题回答一致性评测 (长文本不限字数)
# ------------------------------------------------------------------------------
echo "👉 [6/6] 运行双端 100 题回答一致性对比测试..."
# 分流运行：学生 cuda:3, 教师 cuda:1
$PYTHON_EXEC /mnt/d/minimind-RTPurbo/scripts/eval_consistency.py \
  --model_path "$MODEL_PATH" \
  --head_config "$HEAD_CONFIG" \
  --weight_path "$STAGE2_WEIGHT" \
  --output "$OUT_DIR/qwen_consistency_results.json" \
  --device "cuda:3" \
  --teacher_device "cuda:1" \
  --max_new_tokens 2048

echo "=============================================================="
echo "🎉 全链路 2048 长度复现实验已全部执行完毕！"
echo "所有持久化结果文件请前往: $OUT_DIR 目录查看。"
echo "=============================================================="
