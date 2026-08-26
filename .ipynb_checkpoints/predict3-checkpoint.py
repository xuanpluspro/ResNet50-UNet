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
from utils.utils import cvtColor, preprocess_input

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def load_model(model_path, num_classes, device):
    net = Unet(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    
    uncompiled_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_key = k.replace("_orig_mod.", "")
            uncompiled_state_dict[clean_key] = v
        else:
            uncompiled_state_dict[k] = v
            
    net.load_state_dict(uncompiled_state_dict)
    net.eval()
    net.to(device)
    return net

def detect_image(file_path, model, num_classes, exp_folder, mix_type=True, threshold=0.5):
    try:
        image = Image.open(file_path)
    except (FileNotFoundError, IOError) as e:
        print(f"Error opening image: {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = cvtColor(image)
    old_img = image.copy()

    # ================== 【原尺寸推理：核心修改区】 ==================
    orininal_w, orininal_h = image.size
    target_w = (orininal_w + 31) // 32 * 32
    target_h = (orininal_h + 31) // 32 * 32

    padded_image = Image.new('RGB', (target_w, target_h), (128, 128, 128))
    padded_image.paste(image, (0, 0))
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(padded_image, np.float32)), (2, 0, 1)), 0)
    # ================================================================

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
        images = torch.from_numpy(image_data).to(device)
        pr = model(images)[0]
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

        # ================== 【精确裁剪，完美复原】 ==================
        pr = pr[0:orininal_h, 0:orininal_w]
        # ============================================================

        # ================== 【动态阈值判断区】 ==================
        if num_classes == 2:
            # 取出类别 1（前景/植被）的概率，与设定的阈值进行比较
            pr = (pr[..., 1] >= threshold).astype(np.int64)
        else:
            # 如果是多分类，默认回退到 argmax
            pr = pr.argmax(axis=-1)
        # ========================================================

    seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
    
    if mix_type:
        old_img_np = np.array(old_img)
        alpha = 0.7
        blended_img = cv2.addWeighted(old_img_np, 1 - alpha, seg_img, alpha, 0)
        image = Image.fromarray(blended_img)
    else:
        image = Image.fromarray(np.uint8(seg_img))

    # 强制保存为 PNG 格式，保证边缘清晰无损
    img_name = os.path.basename(file_path)
    mask_filename = os.path.splitext(img_name)[0] + "_mask.jpg"
    save_path = os.path.join(exp_folder, mask_filename)
    image.save(save_path, format="JPEG",quality=95)
    print(f"Mask saved at: {save_path}")


def predict(args):
    # 自动创建存放推理结果的文件夹
    exp_folder = "result3072"
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

    print(f"找到 {len(file_paths)} 张图片，开始预测...")
    t_start = time_synchronized()

    for file_path in file_paths:
        # 传入 argparse 解析出的 threshold 参数
        detect_image(file_path, model, num_classes, exp_folder, mix_type=args.mix_type, threshold=args.threshold)

    t_end = time_synchronized()
    print(f"预测完成！所有结果已保存至 {exp_folder} 文件夹。总耗时: {t_end - t_start:.2f} 秒")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet predict")

    # 修改为目标文件路径和权重路径
    parser.add_argument("--data_path", default="test3072", help="输入图片的文件夹路径")
    parser.add_argument("--weights", default="run/train/exp1/weights/best_model_1.pth", help="模型权重路径")
    
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--mix_type", default=True, action='store_true', help="将原图和分割结果叠加")
    
    # 新增 Threshold 参数控制
    parser.add_argument("--threshold", default=0.5, type=float, help="判定为前景的概率阈值 (0.0 到 1.0 之间)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args)