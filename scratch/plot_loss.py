import os
import pandas as pd
import matplotlib.pyplot as plt

out_dir = '/mnt/d/out_rtpurbo_2048'
os.makedirs(out_dir, exist_ok=True)

# 1. 绘制 Stage 1
csv1_path = os.path.join(out_dir, 'stage1_training_log.csv')
if os.path.exists(csv1_path):
    try:
        df1 = pd.read_csv(csv1_path)
        plt.figure(figsize=(8, 4.5))
        plt.plot(df1['step'], df1['proj_loss'], label='Projection Loss (MSE)', color='#1f77b4', linewidth=2)
        plt.title('Stage 1 Projection Layer Alignment Loss (2048 Sequence)', fontsize=12, fontweight='bold')
        plt.xlabel('Steps', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=9)
        plt.tight_layout()
        img_path = os.path.join(out_dir, 'stage1_loss.png')
        plt.savefig(img_path, dpi=200)
        plt.close()
        print(f"✅ Stage 1 Loss curve generated at: {img_path}")
    except Exception as e:
        print(f"❌ Stage 1 plotting failed: {e}")
else:
    print(f"⚠️ Stage 1 CSV not found: {csv1_path}")

# 2. 绘制 Stage 2
csv2_path = os.path.join(out_dir, 'stage2_training_log.csv')
if os.path.exists(csv2_path):
    try:
        df2 = pd.read_csv(csv2_path)
        # 过滤掉首个可能有异常数值的步数，以便画出精美趋势
        df2_clean = df2.dropna(subset=['total_loss', 'ce_loss', 'kl_loss'])
        plt.figure(figsize=(9, 5))
        plt.plot(df2_clean['step'], df2_clean['total_loss'], label='Total Distill Loss', color='#d62728', linewidth=2)
        plt.plot(df2_clean['step'], df2_clean['ce_loss'], label='Hard Label CE Loss', color='#2ca02c', linewidth=1.5, linestyle='--')
        plt.plot(df2_clean['step'], df2_clean['kl_loss'], label='Soft Target KL Loss', color='#ff7f0e', linewidth=1.5, linestyle=':')
        plt.title('Stage 2 End-to-End Self-Distillation Loss (2048 Sequence)', fontsize=12, fontweight='bold')
        plt.xlabel('Steps', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=9)
        plt.tight_layout()
        img_path = os.path.join(out_dir, 'stage2_loss.png')
        plt.savefig(img_path, dpi=200)
        plt.close()
        print(f"✅ Stage 2 Loss curve generated at: {img_path}")
    except Exception as e:
        print(f"❌ Stage 2 plotting failed: {e}")
else:
    print(f"⚠️ Stage 2 CSV not found: {csv2_path}")
