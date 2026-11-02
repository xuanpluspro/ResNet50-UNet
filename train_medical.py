import os
from functools import partial
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import datetime
import subprocess

from model.unet_resnet import Unet
from model.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.dataloader_medical import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, worker_init_fn
from utils.train_and_eval import train_one_epoch, evaluate, LogColor  # 增加了 LogColor 导入

from utils.create_exp_folder import create_exp_folder
from utils.plot_results import plot_training_curves

def get_gpu_usage():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    used, total = map(int, result.strip().split(','))
    return used

def create_model(num_classes, weights):
    model = Unet(num_classes=num_classes)
    weights_init(model)
    if weights:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(weights, map_location='cpu')
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
    return model

def get_optimizer_and_lr(model, batch_size, train_epoch, momentum, weight_decay, lr=1e-4):
    if not 0 < lr < float('inf'):
        raise ValueError('--lr must be finite and greater than zero')
    Init_lr = lr
    Min_lr = Init_lr * 0.01
    lr_decay_type = 'cos'
    nbs = 16
    lr_limit_max = Init_lr
    lr_limit_min = Init_lr

    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999),
                           weight_decay=weight_decay)
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, train_epoch)
    return optimizer, lr_scheduler_func

# ================= 新增：动态搜索最高 mIoU 对应的阈值 =================
def search_best_threshold(model, val_loader, device, num_classes):
    print(f"\n{LogColor.BLUE}🔍 开始动态阈值搜索 (压制泥土/背景噪点)...{LogColor.RESET}")
    model.eval()
    
    # 搜索范围：0.4 到 0.7 之间
    thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    best_threshold = 0.5
    best_miou = 0.0
    
    total_miou = {t: 0.0 for t in thresholds}
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in val_loader:
            imgs, pngs, _ = batch
            imgs, pngs = imgs.to(device), pngs.to(device)
            outputs = model(imgs)
            
            if num_classes == 2:
                probs = torch.softmax(outputs, dim=1)[:, 1] # 提取植被类别的预测概率
            else:
                break
                
            for t in thresholds:
                predicted = (probs >= t).long()
                ious = []
                for i in range(num_classes):
                    target_mask = (pngs == i)
                    pred_mask = (predicted == i)
                    intersection = torch.logical_and(target_mask, pred_mask).sum().item()
                    union = torch.logical_or(target_mask, pred_mask).sum().item()
                    if target_mask.sum().item() > 0:
                        ious.append(intersection / union if union > 0 else 0.0)
                batch_miou = sum(ious) / len(ious) if len(ious) > 0 else 0.0
                total_miou[t] += batch_miou
                
    if num_classes == 2:
        for t in thresholds:
            avg_miou = total_miou[t] / num_batches
            if avg_miou > best_miou:
                best_miou = avg_miou
                best_threshold = t
        print(f"{LogColor.GREEN}✅ 搜索完毕！最佳阈值设定为: {best_threshold:.2f} (此阈值下 mIoU={best_miou:.4f}){LogColor.RESET}\n")
    return best_threshold
# =====================================================================


def train(args):
    seed_everything(11)
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    
    num_classes = args.num_classes + 1
    train_epoch = args.epochs
    batch_size = args.batch_size
    num_workers = args.workers

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    exp_folder, weights_folder = create_exp_folder()
    input_shape = [512, 512]

    train_dataset = UnetDataset(args.data_path, input_shape, num_classes, augmentation=True, txt_name="train.txt")
    val_dataset = UnetDataset(args.data_path, input_shape, num_classes, augmentation=True, txt_name="val.txt")

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=True, drop_last=False, collate_fn=unet_dataset_collate,
                              worker_init_fn=partial(worker_init_fn, rank=0, seed=11))

    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, drop_last=False, collate_fn=unet_dataset_collate,
                            worker_init_fn=partial(worker_init_fn, rank=0, seed=11))

    model = create_model(num_classes=num_classes, weights=args.weights)
    model.to(device)
    model = torch.compile(model)
    
    scaler = torch.amp.GradScaler(device='cuda') if args.amp else None

    optimizer, lr_scheduler_func = get_optimizer_and_lr(model, batch_size, train_epoch, args.momentum, args.weight_decay, lr=args.lr)

    start_time = time.time()

    # --- 核心修改：改为依靠 mIoU 判断最优模型 ---
    best_miou = 0.0
    best_threshold_current = 0.5  # 初始阈值
    # ---------------------------------------------
    
    best_model_path = os.path.join(weights_folder, f"best_model_{args.num_classes}.pth")
    last_model_path = os.path.join(weights_folder, f"last_model_{args.num_classes}.pth")

    train_losses, val_losses, val_metrics_history = [], [], []
    focal_loss, dice_loss = False, False

    for epoch in range(0, train_epoch):
        gpu_used = get_gpu_usage()
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        # 训练过程
        loss = train_one_epoch(model, optimizer, train_loader, device, dice_loss, focal_loss,
                               gpu_used, num_classes, scaler, epoch, train_epoch)
        train_losses.append(loss)

        # ============ 每 5 轮动态搜索一次最优阈值 ============
        if epoch == 0 or (epoch + 1) % 5 == 0:
            best_threshold_current = search_best_threshold(model, val_loader, device, num_classes)
        # ===================================================

        # 使用找到的阈值进行验证评估
        metrics = evaluate(model, val_loader, device, dice_loss, focal_loss, num_classes, threshold=best_threshold_current)

        val_losses.append(metrics["Loss"])
        val_metrics_history.append(metrics)

        # --- 核心修改：依照 mIoU 保存 Best Model ---
        current_miou = float(metrics["Mean IoU"])
        if current_miou > best_miou:
            best_miou = current_miou
            torch.save(model.state_dict(), best_model_path)
            print(f"{LogColor.YELLOW}🌟 发现更优的 mIoU: {best_miou:.4f}，已保存最新 best_model.pth{LogColor.RESET}")
        # ---------------------------------------------

        torch.save(model.state_dict(), last_model_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    plot_training_curves(train_losses, val_losses, val_metrics_history, weights_folder)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--weights", default="weights/unet_resnet_voc.pth", help="Path to the directory containing model weights")
    parser.add_argument("--data-path", default="VOCdevkit-3.2", help="VOCdevkit root")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--batch-size", default=40, type=int)
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument("--workers", default=12, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay', dest='weight_decay')
    parser.add_argument("--amp", default=True, type=bool, help="Use torch.cuda.amp for mixed precision training")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)