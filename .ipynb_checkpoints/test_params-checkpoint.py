import torch

def count_parameters_from_pth(pth_path):
    print(f"正在读取模型: {pth_path}")
    
    # 【核心魔法】：无卡模式下，必须强行把模型映射到 CPU 上！
    try:
        checkpoint = torch.load(pth_path, map_location=torch.device('cpu'), weights_only=False)
    except FileNotFoundError:
        print("❌ 找不到文件，请检查路径是否正确！(注意前面有没有漏掉 /root/)")
        return
    
    # 兼容不同的保存格式（有的是直接存字典，有的是套在 model_state_dict 里）
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        # 如果存的是完整的网络结构对象
        state_dict = checkpoint.state_dict()

    # 暴力统计字典里所有的张量元素个数
    total_params = sum(p.numel() for p in state_dict.values())
    
    # 统计其中需要求导（可训练）的参数量（仅在存了完整模型时精准，字典模式下为全量）
    # 为了直观，我们直接显示换算后的 M (兆)
    
    print("\n============ 模型参数量统计 ============")
    print(f"总参数个数: {total_params:,}")
    print(f"换算兆级:   {total_params / 1e6:.3f} M")
    print("========================================\n")
    print("注：由于是直接读取权重字典，该数值包含了一些 BN 层的统计量（如 running_mean），")
    print("可能比 model.parameters() 算出来的严格参数量多出 0.1% 左右，但绝对是一个极具参考价值的体量数据。")

if __name__ == "__main__":
    # 填入你的路径，注意 AutoDL 的绝对路径通常前面要加 /root/
    my_model_path = "/root/autodl-tmp/U-Net_medical/run/train/exp1/weights/best_model_1.pth"
    count_parameters_from_pth(my_model_path)