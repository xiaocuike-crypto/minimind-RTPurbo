#!/bin/bash
# ==============================================================================
# Qwen3.5-4B RTPurbo 稀疏注意力机制 - 全链路复现与评测操作手册
# ==============================================================================
# 使用说明:
# 1. 本脚本为复现指南，包含自项目启动以来的所有核心命令。
# 2. 请确保您在 WSL2 环境中运行，并激活对应的 conda 环境：
#    conda activate NLP_TT_Attention
# 3. 默认项目路径为: /mnt/d/minimind-RTPurbo
# 4. 默认训练权重与日志输出目录为: /mnt/d/out
# ==============================================================================

# 创建统一输出目录（若不存在）
mkdir -p /mnt/d/out

# ==============================================================================
# 第一阶段：准备工作与 1024 验证长度
# ==============================================================================

echo "==== [1.1] 载入 Qwen3.5-4B 基座权重 ===="
# 若首次运行，请解除下面一行的注释以执行基座权重下载
# python /mnt/d/minimind-RTPurbo/download_model.py

echo "==== [1.2] 注意力头标定 (1024 长度) ===="
# 作用: 分析 Qwen3.5-4B 中每个 attention head 对远距离 token 的注意力占比，生成头标定配置
# 本次升级：已自动在 json 相同路径下导出 qwen_head_config_scores.csv 详细打分数据
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/scripts/calibrate_qwen_heads.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --output /mnt/d/minimind-RTPurbo/qwen_head_config.json \
  --max_seq_len 1024 \
  --num_samples 20

echo "==== [1.3] Stage 1 低维投影层训练 (1024 长度) ===="
# 作用: 冻结基座，仅微调新引入的低维注意力投影矩阵 (Q, K 的投影)
# 本次升级：已自动将每步训练指标（Epoch, Step, Loss, LR）持久化至 /mnt/d/out/stage1_training_log.csv
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/trainer/train_qwen_rtpurbo_stage1.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config.json \
  --max_seq_len 1024 \
  --max_train_steps 100 \
  --save_tag 1024

echo "==== [1.4] Stage 2 端到端自蒸馏微调 (1024 长度) ===="
# 作用: 使用 Full Attention 作为 Teacher，通过 KL 散度约束微调 Student 稀疏注意力全参数
# 本次升级：已自动将自蒸馏各项 Loss 与 LR 持久化至 /mnt/d/out/stage2_training_log.csv
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/trainer/train_qwen_rtpurbo_stage2.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config.json \
  --stage1_weight /mnt/d/out/rtpurbo_stage1_proj_qwen_1024.pth \
  --max_seq_len 1024 \
  --max_train_steps 200 \
  --save_tag 1024

echo "==== [1.5] 性能与短文本生成评测 (1024 长度) ===="
# 作用: 对比基座 Full Attention 模型与 RTPurbo 稀疏注意力的时延表现和短文本生成
# 结果路径: /mnt/d/minimind-RTPurbo/qwen_eval_results.json
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/scripts/eval_qwen_rtpurbo.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config.json \
  --weight_path /mnt/d/out/rtpurbo_stage2_qwen_1024.pth \
  --output /mnt/d/minimind-RTPurbo/qwen_eval_results.json


# ==============================================================================
# 第二阶段：长文本能力扩展 (2048 长度 —— 方案 A 扩展)
# ==============================================================================

echo "==== [2.1] 注意力头标定 (2048 长度) ===="
# 作用: 在 2048 扩展长度下重新标定注意力头
# 本次升级：自动在 /mnt/d/out/ 生成详细打分 CSV：/mnt/d/minimind-RTPurbo/qwen_head_config_2048_scores.csv
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/scripts/calibrate_qwen_heads.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --output /mnt/d/minimind-RTPurbo/qwen_head_config_2048.json \
  --max_seq_len 2048 \
  --num_samples 20

echo "==== [2.2] Stage 1 低维投影层训练 (2048 长度) ===="
# 作用: 训练 2048 对应注意力头的低维投影权重
# 日志路径: /mnt/d/out/stage1_training_log.csv
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/trainer/train_qwen_rtpurbo_stage1.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config_2048.json \
  --max_seq_len 2048 \
  --max_train_steps 100 \
  --save_tag 2048

echo "==== [2.3] Stage 2 端到端自蒸馏微调 (2048 长度) ===="
# 作用: 在 2048 长度下利用双卡分流（Teacher-cuda:3, Student-cuda:2）进行端到端自蒸馏
# 日志路径: /mnt/d/out/stage2_training_log.csv
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/trainer/train_qwen_rtpurbo_stage2.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config_2048.json \
  --stage1_weight /mnt/d/out/rtpurbo_stage1_proj_qwen_2048.pth \
  --max_seq_len 2048 \
  --max_train_steps 200 \
  --save_tag 2048

echo "==== [2.4] 性能与长文本生成评测 (2048 长度) ===="
# 作用: 评测 2048 长度下 Full Attention 与 RTPurbo 延迟表现与生成差异
# 结果路径: /mnt/d/out/qwen_eval_results_2048.json
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/scripts/eval_qwen_rtpurbo.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config_2048.json \
  --weight_path /mnt/d/out/rtpurbo_stage2_qwen_2048.pth \
  --output /mnt/d/out/qwen_eval_results_2048.json

echo "==== [2.5] 2048 长度大海捞针（Needle in a Haystack）测试 ===="
# 作用: 测试稀疏化学生模型在长噪声文本（2000字）中，于 10%、50%、90% 深度下检索特定钥匙的召回率
# 本次升级：推理的原始回答结果及召回成功状态已自动落盘至：/mnt/d/out/needle_in_haystack_results.json
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/scripts/needle_in_haystack.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config_2048.json \
  --weight_path /mnt/d/out/rtpurbo_stage2_qwen_2048.pth \
  --output /mnt/d/out/needle_in_haystack_results.json

echo "==== [2.6] 双端模型 100 题回答一致性评测 (长文本不限字数生成) ===="
# 作用: 加载 Teacher (cuda:3) 与 Student (cuda:2) 进行涵盖 7 大场景 100 道日常问答的生成对齐测试，并计算 ROUGE-L 与 BLEU-1 平均重合度
# 结果路径: /mnt/d/out/qwen_consistency_results.json
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/scripts/eval_consistency.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config_2048.json \
  --weight_path /mnt/d/out/rtpurbo_stage2_qwen_2048.pth \
  --output /mnt/d/out/qwen_consistency_results.json \
  --device cuda:2 \
  --teacher_device cuda:3 \
  --max_new_tokens 2048
