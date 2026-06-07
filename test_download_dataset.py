import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
print("正在尝试从 HuggingFace 镜像源加载 THUDM/LongAlign-10k 数据集...")
try:
    # 只加载一行看一下
    ds = load_dataset('THUDM/LongAlign-10k', split='train', streaming=True)
    first_sample = next(iter(ds))
    print("加载成功！")
    print("Keys:", first_sample.keys())
    if 'conversations' in first_sample:
        print("conversations len:", len(first_sample['conversations']))
        print("Preview:", str(first_sample['conversations'])[:300])
except Exception as e:
    print("加载失败，错误为:", e)
