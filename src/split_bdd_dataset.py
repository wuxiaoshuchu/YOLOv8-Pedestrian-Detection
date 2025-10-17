import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    """
    主函数，用于将BDD100K的训练集划分为新的训练集和验证集。
    """
    print("--- 开始划分BDD100K数据集 ---")

    # --- 1. 定义参数和路径 ---
    # 定义验证集所占的比例（例如0.2代表20%）
    VALIDATION_SPLIT = 0.2
    
    # 使用固定的随机种子，确保每次划分结果都一样，便于复现
    RANDOM_SEED = 42

    project_root = Path(__file__).parent.parent
    base_path = project_root / "data" / "raw" / "bdd100k"

    images_base_path = base_path / "images"
    labels_base_path = base_path / "labels"

    # 原始的训练数据路径
    source_images_dir = images_base_path / "train"
    source_labels_dir = labels_base_path / "train"

    # 我们要创建的验证数据路径
    val_images_dir = images_base_path / "val"
    val_labels_dir = labels_base_path / "val"

    # --- 2. 创建验证集文件夹 ---
    print("正在创建验证集文件夹...")
    val_images_dir.mkdir(exist_ok=True)
    val_labels_dir.mkdir(exist_ok=True)

    # --- 3. 随机抽样 ---
    # 获取所有训练图片的列表（假设图片都是.jpg格式）
    all_images = [f for f in os.listdir(source_images_dir) if f.endswith('.jpg')]
    
    # 设置随机种子并打乱列表
    random.seed(RANDOM_SEED)
    random.shuffle(all_images)

    # 计算划分点
    split_point = int(len(all_images) * VALIDATION_SPLIT)
    
    # 获取要移动到验证集的文件列表
    files_to_move = all_images[:split_point]

    print(f"总共有 {len(all_images)} 张图片。")
    print(f"将移动 {len(files_to_move)} 张图片到验证集。")
    print(f"剩余 {len(all_images) - len(files_to_move)} 张图片作为新的训练集。")

    # --- 4. 移动文件 ---
    print("\n正在移动文件...")
    for filename in tqdm(files_to_move, desc="移动文件到验证集"):
        base_name = filename.split('.')[0]
        label_filename = f"{base_name}.txt"

        # 定义源文件路径
        src_image_path = source_images_dir / filename
        src_label_path = source_labels_dir / label_filename

        # 定义目标文件路径
        dest_image_path = val_images_dir / filename
        dest_label_path = val_labels_dir / label_filename

        # 移动图片文件
        if src_image_path.exists():
            shutil.move(str(src_image_path), str(dest_image_path))
        
        # 移动对应的标签文件
        if src_label_path.exists():
            shutil.move(str(src_label_path), str(dest_label_path))

    print("\n✅ 数据集划分完成！")

if __name__ == '__main__':
    main()
