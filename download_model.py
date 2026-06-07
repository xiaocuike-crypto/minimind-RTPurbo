import os
import shutil
from modelscope import snapshot_download

# 清除代理环境变量以绕过代理提高下载速度
for var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(var, None)

print("正在从 ModelScope 下载 Qwen/Qwen3.5-4B 模型（临时绕过代理以加速下载）...")
try:
    model_dir = snapshot_download('Qwen/Qwen3.5-4B')
    print(f"下载成功！模型缓存路径：{model_dir}")

    dest_dir = '/mnt/d/minimind-RTPurbo/model/Qwen3.5-4B'
    print(f"正在将模型文件移动至：{dest_dir} ...")
    os.makedirs(dest_dir, exist_ok=True)

    # 移动所有文件到目标目录
    for file_name in os.listdir(model_dir):
        src = os.path.join(model_dir, file_name)
        dst = os.path.join(dest_dir, file_name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    print(f"✅ 模型下载与移动已顺利完成！目标路径：{dest_dir}")
except Exception as e:
    print(f"❌ 下载或文件移动失败：{e}")
