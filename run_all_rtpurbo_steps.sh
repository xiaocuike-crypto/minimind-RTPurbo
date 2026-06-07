#!/bin/bash
# ==============================================================================
# Qwen3.5-4B RTPurbo 稀疏注意力复现项目 - 全流程命令操作手册
# ==============================================================================
# 提示: 本脚本整合了项目从零开始的全部核心操作指令。可作为分步执行指南。
# 执行前请确保您处于 WSL 虚拟环境: NLP_TT_Attention 中。
# 工作区根路径统一设定为: /mnt/d/minimind-RTPurbo
# 训练输出与权重存放路径: /mnt/d/out
# ==============================================================================

# 创建统一输出目录
mkdir -p /mnt/d/out

# ==============================================================================
# 第一阶段：准备工作（1024 验证长度）
# ==============================================================================

echo "==== 阶段 1.1: 下载 Qwen3.5-4B 基座模型权重 ===="
# 作用: 首次运行项目需拉取基座权重到项目内 model 目录下
# python /mnt/d/minimind-RTPurbo/download_model.py

echo "==== 阶段 1.2: 对基座模型在 1024 长度下重新标定 Attention Heads ===="
# 作用: 抽取 20 条样本在 1024 长度下运行，统计每个注意力头在不同 token 上的分布情况，保存头部重要性划分。
# 产出配置文件: /mnt/d/minimind-RTPurbo/qwen_head_config.json
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/scripts/calibrate_qwen_heads.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --output /mnt/d/minimind-RTPurbo/qwen_head_config.json \
  --max_seq_len 1024 \
  --num_samples 20

echo "==== 阶段 1.3: 训练 Stage 1 投影层 (1024 长度) ===="
# 作用: 冻结基座，仅训练新引入的低维检索头投影映射矩阵（Q, K的低维投影），对齐特征空间。
# 训练参数: 100 步
# 产出权重: /mnt/d/out/rtpurbo_stage1_proj_qwen_1024.pth
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/trainer/train_qwen_rtpurbo_stage1.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config.json \
  --max_seq_len 1024 \
  --max_train_steps 100 \
  --save_tag 1024

echo "==== 阶段 1.4: 训练 Stage 2 端到端自蒸馏微调 (1024 长度) ===="
# 作用: 以 Stage 1 的对齐投影权重为初始化，通过计算 Full Attention (Teacher) 与 RTPurbo 稀疏注意力 (Student) 间的 KL 散度进行双模型端到端蒸馏，微调全模型参数以消除因注意力丢弃导致的概率偏差。
# 训练参数: 200 步
# 产出最终微调权重: /mnt/d/out/rtpurbo_stage2_qwen_1024.pth
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/trainer/train_qwen_rtpurbo_stage2.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config.json \
  --stage1_weight /mnt/d/out/rtpurbo_stage1_proj_qwen_1024.pth \
  --max_seq_len 1024 \
  --max_train_steps 200 \
  --save_tag 1024

echo "==== 阶段 1.5: 评测时延与短文本生成 (1024 长度) ===="
# 作用: 对比基座 Full Attention 模型与 RTPurbo 稀疏注意力的时延表现和生成文本相似度。
# 产出报告: /mnt/d/minimind-RTPurbo/qwen_eval_results.json
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/scripts/eval_qwen_rtpurbo.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config.json \
  --weight_path /mnt/d/out/rtpurbo_stage2_qwen_1024.pth \
  --output /mnt/d/minimind-RTPurbo/qwen_eval_results.json


# ==============================================================================
# 第二阶段：长文本能力扩展（2048 扩展长度 —— 方案 A）
# ==============================================================================

echo "==== 阶段 2.1: 对基座模型在 2048 长度下重新标定 Attention Heads ===="
# 作用: 在 2048 长度下统计检索头与局部头的划分，以便捕获更长上下文下的全局键关联分布。
# 产出配置文件: /mnt/d/minimind-RTPurbo/qwen_head_config_2048.json
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/scripts/calibrate_qwen_heads.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --output /mnt/d/minimind-RTPurbo/qwen_head_config_2048.json \
  --max_seq_len 2048 \
  --num_samples 20

echo "==== 阶段 2.2: 训练 Stage 1 投影层 (2048 长度) ===="
# 作用: 训练 2048 对应注意力头的低维投影权重。
# 产出权重: /mnt/d/out/rtpurbo_stage1_proj_qwen_2048.pth
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/trainer/train_qwen_rtpurbo_stage1.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config_2048.json \
  --max_seq_len 2048 \
  --max_train_steps 100 \
  --save_tag 2048

echo "==== 阶段 2.3: 训练 Stage 2 端到端自蒸馏微调 (2048 长度) ===="
# 作用: 在 2048 Token 序列上利用双卡分流（Teacher 卡3，Student 卡2）进行 200 步端到端自蒸馏，消除注意力剪枝丢失的信息偏差。
# 产出最终微调权重: /mnt/d/out/rtpurbo_stage2_qwen_2048.pth
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/trainer/train_qwen_rtpurbo_stage2.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config_2048.json \
  --stage1_weight /mnt/d/out/rtpurbo_stage1_proj_qwen_2048.pth \
  --max_seq_len 2048 \
  --max_train_steps 200 \
  --save_tag 2048

echo "==== 阶段 2.4: 评估 2048 时延性能与生成对比 ===="
# 作用: 评测 2048 长度长序列下 Full Attention 与 RTPurbo 延迟表现。
# 产出报告: /mnt/d/out/qwen_eval_results_2048.json
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/scripts/eval_qwen_rtpurbo.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config_2048.json \
  --weight_path /mnt/d/out/rtpurbo_stage2_qwen_2048.pth \
  --output /mnt/d/out/qwen_eval_results_2048.json

echo "==== 阶段 2.5: 运行 2048 长度长文本 大海捞针（Needle in a Haystack）测试 ===="
# 作用: 测试稀疏化学生模型在长噪声文本（2000字）中，于 10%、50%、90% 深度下检索特异性凭证的召回率，验证长依赖注意力投射机制。
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/scripts/needle_in_haystack.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config_2048.json \
  --weight_path /mnt/d/out/rtpurbo_stage2_qwen_2048.pth

echo "==== 阶段 2.6: 运行双端模型 100 题回答一致性评测 (长文本不限字数生成) ===="
# 作用: 加载 Teacher (cuda:3) 与 Student (cuda:2) 进行涵盖 7 大场景 100 道日常问答的生成对齐测试，并计算 ROUGE-L 与 BLEU-1 平均重合度。
# 产出最终相似度报告: /mnt/d/out/qwen_consistency_results.json
wsl /home/ckiphone/anaconda3/envs/NLP_TT_Attention/bin/python /mnt/d/minimind-RTPurbo/scripts/eval_consistency.py \
  --model_path /mnt/d/minimind-RTPurbo/model/Qwen3.5-4B \
  --head_config /mnt/d/minimind-RTPurbo/qwen_head_config_2048.json \
  --weight_path /mnt/d/out/rtpurbo_stage2_qwen_2048.pth \
  --output /mnt/d/out/qwen_consistency_results.json \
  --device cuda:2 \
  --teacher_device cuda:3 \
  --max_new_tokens 2048
