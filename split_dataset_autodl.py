import os
import random
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

def stratified_split_voc_dataset():
    # 1. 路径配置 (已更新为 AutoDL 上的新路径)
    base_dir = r"/root/autodl-tmp/U-Net_medical/VOCdevkit-3.2"
    img_dir = os.path.join(base_dir, "JPEGImages")
    mask_dir = os.path.join(base_dir, "SegmentationClass")
    
    txt_dir = os.path.join(base_dir, "ImageSets", "Segmentation")
    os.makedirs(txt_dir, exist_ok=True)
    
    train_file = os.path.join(txt_dir, "train.txt")
    val_file = os.path.join(txt_dir, "val.txt")
    test_file = os.path.join(txt_dir, "test.txt")

    # 2. 获取图片和掩码交集，执行【第一重防抖：强制排序】
    valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.bmp'}
    try:
        img_files = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_exts}
        mask_files = {os.path.splitext(f)[0] for f in os.listdir(mask_dir) if os.path.splitext(f)[1].lower() in {'.png', '.bmp'}}
    except FileNotFoundError as e:
        print(f"❌ 找不到目录: {e}")
        return

    stems = list(img_files & mask_files)
    # 核心：抹平不同操作系统(如 ext4 vs NTFS)底层读取文件的乱序差异
    stems.sort()  
    
    if not stems:
        print("⚠️ 没有找到匹配的图像和标签！")
        return
        
    print(f"🔍 找到 {len(stems)} 对图像和标签，开始提取多尺度特征进行分层...")

    # 3. 提取特征：计算目标面积占比，放入不同阶层 (Strata)
    strata_buckets = defaultdict(list)
    
    for stem in tqdm(stems, desc="计算目标特征"):
        mask_path = os.path.join(mask_dir, stem + ".png")
        try:
            mask_np = np.array(Image.open(mask_path))
            # 计算植被 (像素值为 1) 在全图的面积占比
            veg_ratio = np.sum(mask_np == 1) / mask_np.size
            
            # 根据面积占比，将图片精确分入 4 个阶层的“筐”中
            if veg_ratio <= 0.05:
                strata_buckets["tiny_target"].append(stem)
            elif veg_ratio <= 0.15:
                strata_buckets["small_target"].append(stem)
            elif veg_ratio <= 0.40:
                strata_buckets["medium_target"].append(stem)
            else:
                strata_buckets["large_target"].append(stem)
        except Exception as e:
            print(f"\n⚠️ 读取 {mask_path} 失败: {e}")

    # 4. 各阶层内部独立划分，执行【第二重防抖：全局固定随机种子】
    train_stems, val_stems, test_stems = [], [], []
    random.seed(42)
    
    print("\n📊 阶层分配详情:")
    # 为了保证字典遍历顺序稳定，对 key 也进行排序遍历
    for bucket_name in sorted(strata_buckets.keys()):
        bucket_stems = strata_buckets[bucket_name]
        # 再次确保各筐内部顺序绝对一致
        bucket_stems.sort()  
        random.shuffle(bucket_stems)
        
        total_len = len(bucket_stems)
        train_end = int(total_len * 0.8)
        val_end = train_end + int(total_len * 0.1)
        
        # 将该阶层按 8:1:1 切分，均匀汇入总集合
        train_stems.extend(bucket_stems[:train_end])
        val_stems.extend(bucket_stems[train_end:val_end])
        test_stems.extend(bucket_stems[val_end:])
        
        print(f"  -> [{bucket_name:<13}] 共 {total_len:<4} 张 | Train:{train_end:<4} Val:{val_end-train_end:<4} Test:{total_len-val_end:<4}")

    # 5. 最后对总体再打乱一次（防止 DataLoader 读取时产生阶段性突变偏差）
    random.shuffle(train_stems)
    random.shuffle(val_stems)
    random.shuffle(test_stems)

    # 6. 写入标准 VOC 格式的 txt 文件
    def write_txt(filepath, data):
        with open(filepath, 'w') as f:
            for item in data:
                f.write(item + '\n')

    write_txt(train_file, train_stems)
    write_txt(val_file, val_stems)
    write_txt(test_file, test_stems)

    print("\n" + "="*50)
    print(f"🎉 高级多尺度分层划分完成！(100% 可跨服务器复现)")
    print(f"  ✅ 训练集 (Train): {len(train_stems)} 张")
    print(f"  ✅ 验证集 (Val)  : {len(val_stems)} 张")
    print(f"  ✅ 测试集 (Test) : {len(test_stems)} 张")
    print(f"文件已保存至:\n{txt_dir}")
    print("="*50)

if __name__ == "__main__":
    stratified_split_voc_dataset()