import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.unet_resnet import Unet
from utils.utils import cvtColor, preprocess_input

def time_synchronized():
    """
    该函数同步 CUDA 操作，并返回当前时间。
    """
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def load_model(model_path, num_classes, device):
    # 1. 创建原汁原味的 Unet 模型
    net = Unet(num_classes=num_classes)
    
    # 2. 将权重加载到内存（字典格式）
    state_dict = torch.load(model_path, map_location=device)
    
    # 3. 核心修复区：脱壳操作，剔除 _orig_mod. 前缀
    uncompiled_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_key = k.replace("_orig_mod.", "")
            uncompiled_state_dict[clean_key] = v
        else:
            uncompiled_state_dict[k] = v
            
    # 4. 把干净的权重加载进模型
    net.load_state_dict(uncompiled_state_dict)
    
    net.eval()  # 设置为评估模式
    net.to(device)  
    
    return net


def detect_image(file_path, model, exp_folder):
    try:
        image = Image.open(file_path)
    except (FileNotFoundError, IOError) as e:
        print(f"Error opening image: {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将图像转换为 RGB 格式
    image = cvtColor(image)

    # ================== 【原尺寸推理：核心修改区】 ==================
    orininal_w, orininal_h = image.size

    # 自动计算出能完全包裹原图的“最小 32 倍数”尺寸
    target_w = (orininal_w + 31) // 32 * 32
    target_h = (orininal_h + 31) // 32 * 32

    # 创建一个灰色背景画布，把原图贴在左上角
    padded_image = Image.new('RGB', (target_w, target_h), (128, 128, 128))
    padded_image.paste(image, (0, 0))

    # 预处理并转为张量
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(padded_image, np.float32)), (2, 0, 1)), 0)
    # ================================================================

    with torch.no_grad():
        images = torch.from_numpy(image_data).to(device)
        pr = model(images)[0]
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

        # 精确裁剪，摒弃插值带来的误差
        pr = pr[0:orininal_h, 0:orininal_w]
        
        # 获取预测类别索引 (背景为0，目标为1)
        pr = pr.argmax(axis=-1)

    # ================== 【纯净 Mask 生成区】 ==================
    # 将 0 和 1 的矩阵，直接乘以 255。背景(0*255=0)纯黑，目标(1*255=255)纯白
    mask_np = (pr * 255).astype(np.uint8)
    
    # 转换为单通道灰度图 (Mode 'L')
    mask_image = Image.fromarray(mask_np, mode='L')
    # ==========================================================

    # 保存结果
    img_name = os.path.basename(file_path)
    # 强制保存为无损的 .png 格式，防止 jpg 压缩破坏 0/255 的纯净度
    mask_filename = os.path.splitext(img_name)[0] + "_mask.png"
    save_path = os.path.join(exp_folder, mask_filename)
    
    mask_image.save(save_path)
    print(f"纯净 Mask 已保存至: {save_path}")


def predict(args):
    # 自动创建 result3072pure 文件夹
    exp_folder = "result3072pure"
    os.makedirs(exp_folder, exist_ok=True) 

    num_classes = args.num_classes + 1
    assert os.path.exists(args.weights), f"weights {args.weights} not found."
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"正在加载模型: {args.weights} ...")
    model = load_model(args.weights, num_classes, device)

    if os.path.isdir(args.data_path):
        file_paths = [str(p) for p in Path(args.data_path).rglob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    elif os.path.isfile(args.data_path):
        file_paths = [args.data_path]
    else:
        raise ValueError(f"不支持的输入路径: {args.data_path}")

    print(f"找到 {len(file_paths)} 张图片，开始原尺寸预测...")
    t_start = time_synchronized()

    for file_path in file_paths:
        detect_image(file_path, model, exp_folder)

    t_end = time_synchronized()
    print(f"预测完成！所有纯净 Mask 已保存至 {exp_folder} 文件夹。总耗时: {t_end - t_start:.2f} 秒")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet predict (pure mask)")

    parser.add_argument("--data_path", default="test3072", help="输入图片的文件夹路径")
    parser.add_argument("--weights", default="run/train/exp/weights/best_model_1.pth", help="模型权重路径")
    parser.add_argument("--num-classes", default=1, type=int)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args)