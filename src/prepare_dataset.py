import os
import re
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def convert_pennfudan_txt_to_yolo(txt_file_path: Path, img_width: int, img_height: int) -> str:
    """解析PennFudan的.txt标注文件并转换为YOLO格式的字符串。"""
    with open(txt_file_path, 'r', encoding='latin-1') as f:
        content = f.read()
    
    pattern = r"Bounding box for object \d+ \"PASpersonWalking\" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)"
    matches = re.findall(pattern, content)
    
    yolo_annotations = []
    for match in matches:
        xmin, ymin, xmax, ymax = [int(coord) for coord in match]
        class_id = 0  # 'pedestrian'
        
        # 转换为YOLO格式
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        yolo_annotations.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
        
    return "\n".join(yolo_annotations)

def main():
    """
    主函数，执行数据集的准备、转换和划分。
    """
    print("--- 开始处理Penn-Fudan数据集 ---")

    # 定义路径
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / "data" / "raw" / "PennFudanPed"
    processed_data_path = project_root / "data" / "processed"

    raw_images_path = raw_data_path / "PNGImages"
    raw_labels_path = raw_data_path / "Annotation"

    # 1. 清理并创建处理后的数据目录结构
    if processed_data_path.exists():
        print(f"清理旧的处理数据文件夹: {processed_data_path}")
        shutil.rmtree(processed_data_path)
    
    print("创建新的处理数据目录结构...")
    train_images_dir = processed_data_path / "images" / "train"
    val_images_dir = processed_data_path / "images" / "val"
    train_labels_dir = processed_data_path / "labels" / "train"
    val_labels_dir = processed_data_path / "labels" / "val"
    
    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    # 2. 获取所有文件并进行配对
    all_image_files = sorted([f for f in os.listdir(raw_images_path) if f.endswith('.png')])
    
    # 3. 随机打乱并划分数据集
    random.seed(42) # 使用固定的随机种子，确保每次划分结果都一样
    random.shuffle(all_image_files)

    split_index = int(len(all_image_files) * 0.8) # 80% 作为训练集
    train_files = all_image_files[:split_index]
    val_files = all_image_files[split_index:]

    print(f"数据集划分完成: {len(train_files)} 张训练图片, {len(val_files)} 张验证图片。")

    # 4. 处理并保存训练集
    print("\n正在处理训练集...")
    for filename in tqdm(train_files, desc="处理训练集"):
        base_name = filename.split('.')[0]
        
        # 复制图片
        shutil.copy(raw_images_path / filename, train_images_dir / filename)
        
        # 转换并保存标签
        img = Image.open(raw_images_path / filename)
        yolo_label_content = convert_pennfudan_txt_to_yolo(
            raw_labels_path / f"{base_name}.txt",
            img.width,
            img.height
        )
        with open(train_labels_dir / f"{base_name}.txt", 'w') as f:
            f.write(yolo_label_content)

    # 5. 处理并保存验证集
    print("\n正在处理验证集...")
    for filename in tqdm(val_files, desc="处理验证集"):
        base_name = filename.split('.')[0]
        
        # 复制图片
        shutil.copy(raw_images_path / filename, val_images_dir / filename)
        
        # 转换并保存标签
        img = Image.open(raw_images_path / filename)
        yolo_label_content = convert_pennfudan_txt_to_yolo(
            raw_labels_path / f"{base_name}.txt",
            img.width,
            img.height
        )
        with open(val_labels_dir / f"{base_name}.txt", 'w') as f:
            f.write(yolo_label_content)

    print("\n✅ 数据集准备完成！所有文件已保存到 data/processed 目录。")


if __name__ == '__main__':
    main()
