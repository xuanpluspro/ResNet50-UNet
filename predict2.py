import colorsys
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.unet_resnet import Unet
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.create_exp_folder import create_val_exp_folder

def time_synchronized():
    """
    该函数同步 CUDA 操作，并返回当前时间。

    如果使用 GPU，`torch.cuda.synchronize()` 将会等待所有 CUDA 操作完成，
    以确保测量时间时没有潜在的异步操作。

    Returns:
        float: 当前时间戳，单位为秒。
    """
    # 如果 CUDA 可用，则等待所有 CUDA 操作完成
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # 返回当前时间戳（单位为秒）
    return time.time()


def load_model(model_path, num_classes, device):
    # 1. 创建原汁原味的 Unet 模型
    net = Unet(num_classes=num_classes)
    
    # 2. 将权重加载到内存（字典格式）
    state_dict = torch.load(model_path, map_location=device)
    
    # 3. 核心修复区：脱壳操作，剔除 _orig_mod. 前缀
    uncompiled_state_dict = {}
    for k, v in state_dict.items():
        # 如果键名以 _orig_mod. 开头，就把这个前缀删掉
        if k.startswith("_orig_mod."):
            clean_key = k.replace("_orig_mod.", "")
            uncompiled_state_dict[clean_key] = v
        else:
            uncompiled_state_dict[k] = v
            
    # 4. 把干净的权重加载进模型
    net.load_state_dict(uncompiled_state_dict)
    
    net.eval()  # 设置为评估模式
    net.to(device)  # 移动到指定设备
    
    return net

def detect_image(file_path, model, num_classes, exp_folder, mix_type=True):
    try:
        image = Image.open(file_path)
    except (FileNotFoundError, IOError) as e:
        print(f"Error opening image: {e}")
        return

    # 自动选择设备（GPU 或 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将图像转换为 RGB 格式
    image = cvtColor(image)
    old_img = image.copy()  # 直接复制原图，用于最后的叠图

    # ================== 【原尺寸推理：核心修改区】 ==================
    # 1. 获取原图真实的宽和高 (PIL 的 size 返回的是 w, h)
    orininal_w, orininal_h = image.size

    # 2. U-Net 要求输入尺寸必须是 32 的整数倍。
    # 我们自动计算出能完全包裹原图的“最小 32 倍数”尺寸
    target_w = (orininal_w + 31) // 32 * 32
    target_h = (orininal_h + 31) // 32 * 32

    # 3. 创建一个灰色背景画布，把原图“原封不动”地贴在左上角 (绝对不缩放！)
    padded_image = Image.new('RGB', (target_w, target_h), (128, 128, 128))
    padded_image.paste(image, (0, 0))

    # 4. 对这个填充后的原尺寸图进行预处理并转为张量
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(padded_image, np.float32)), (2, 0, 1)), 0)
    # ================================================================

    # 生成颜色映射
    if num_classes <= 21:
        colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                  (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                  (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                  (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 128)]
    else:
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    with torch.no_grad():
        # 将图像数据转换为 PyTorch 张量
        images = torch.from_numpy(image_data).to(device)

        # 进行推理
        pr = model(images)[0]

        # Softmax 后处理
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

        # ================== 【精确裁剪，完美复原】 ==================
        # 直接把左上角我们贴上去的“绝对原图区域”切下来，彻底摒弃 cv2.resize 插值带来的误差
        pr = pr[0:orininal_h, 0:orininal_w]
        # ============================================================

        # 获取预测类别
        pr = pr.argmax(axis=-1)

    # 根据不同的混合类型，生成不同的结果
    seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
    
    if mix_type:
        # 将 PIL 图像转换为 NumPy 数组
        old_img_np = np.array(old_img)
        alpha = 0.7
        # 使用 OpenCV 进行加权混合
        blended_img = cv2.addWeighted(old_img_np, 1 - alpha, seg_img, alpha, 0)
        # 将混合后的图像转换回 PIL 图像
        image = Image.fromarray(blended_img)
    else:
        image = Image.fromarray(np.uint8(seg_img))

    # 保存结果
    img_name = os.path.basename(file_path)
    mask_filename = os.path.splitext(img_name)[0] + "_mask.png"
    save_path = os.path.join(exp_folder, mask_filename)
    image.save(save_path)
    print(f"Mask saved at: {save_path}")




def predict(args):
    # ================= 修改点 1：自动创建 result1 文件夹 =================
    exp_folder = "result3072"
    os.makedirs(exp_folder, exist_ok=True)  # 如果不存在则创建，存在则不报错
    # =================================================================

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

    print(f"找到 {len(file_paths)} 张图片，开始预测...")
    t_start = time_synchronized()

    for file_path in file_paths:
        # mix_type=True 会自动把 Mask 和原图叠在一起
        detect_image(file_path, model, num_classes, exp_folder, mix_type=args.mix_type)

    t_end = time_synchronized()
    print(f"预测完成！所有结果已保存至 {exp_folder} 文件夹。总耗时: {t_end - t_start:.2f} 48秒")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet predict")

    # ================= 修改点 2：直接把默认路径改成你要求的路径 =================
    parser.add_argument("--data_path", default="test3072", help="输入图片的文件夹路径")
    parser.add_argument("--weights", default="run/train/exp/weights/best_model_1.pth", help="模型权重路径")
    # ============================================================================
    
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--mix_type", default=True, action='store_true', help="将原图和分割结果叠加")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args)

