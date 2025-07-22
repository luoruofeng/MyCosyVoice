import os
import subprocess
import zipfile

def download_models():
    # 创建 pretrained_models 目录
    os.makedirs('pretrained_models', exist_ok=True)

    # git clone 各个模型
    clones = [
        ('https://www.modelscope.cn/iic/CosyVoice2-0.5B.git', 'pretrained_models/CosyVoice2-0.5B'),
        ('https://www.modelscope.cn/iic/CosyVoice-300M.git', 'pretrained_models/CosyVoice-300M'),
        ('https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git', 'pretrained_models/CosyVoice-300M-SFT'),
        ('https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git', 'pretrained_models/CosyVoice-300M-Instruct'),
        ('https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git', 'pretrained_models/CosyVoice-ttsfrd')
    ]

    for url, path in clones:
        if not os.path.exists(path):
            subprocess.call(['git', 'clone', url, path])
        else:
            print(f"{path} 已存在，跳过克隆。")

    # 处理 ttsfrd：解压并安装（可选，如果未安装将使用 WeTextProcessing）
    ttsfrd_dir = 'pretrained_models/CosyVoice-ttsfrd'
    zip_path = os.path.join(ttsfrd_dir, 'resource.zip')
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ttsfrd_dir)
        # 安装 whl 文件
        whl_files = [
            os.path.join(ttsfrd_dir, 'ttsfrd_dependency-0.1-py3-none-any.whl'),
            os.path.join(ttsfrd_dir, 'ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl')
        ]
        for whl in whl_files:
            if os.path.exists(whl):
                subprocess.call(['pip', 'install', whl])
            else:
                print(f"警告: {whl} 不存在，跳过安装。")
    else:
        print("警告: resource.zip 不存在，跳过 ttsfrd 安装，将使用 WeTextProcessing 默认。")

if __name__ == '__main__':
    download_models()