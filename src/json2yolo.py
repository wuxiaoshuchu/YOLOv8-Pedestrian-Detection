import json
import os
from pathlib import Path
from tqdm import tqdm

def convert_bdd_json_to_yolo(json_path: Path):
    """
    读取单个BDD100K的.json文件，将其内容转换为YOLO格式的字符串。
    """
    with open(json_path) as f:
        data = json.load(f)

    # BDD100K的类别名称到我们在.yaml中定义的ID的映射
    # 这必须和您的 bdd100k.yaml 文件中的 'names' 保持一致
    category_map = {
        'pedestrian': 0,
        'rider': 1,
        'car': 2,
        'truck': 3,
        'bus': 4,
        'train': 5,
        'motorcycle': 6,
        'bicycle': 7,
        'traffic light': 8,
        'traffic sign': 9
    }
    
    # 从JSON中获取图片尺寸
    # BDD100K v1的JSON格式没有直接提供图片尺寸，我们需要从图片本身获取
    # 为了简化，我们假设图片和标签文件名一一对应，并且图片在../images/目录下
    try:
        from PIL import Image
        img_path = json_path.parent.parent.parent / "images" / json_path.parent.name / f"{json_path.stem}.jpg"
        with Image.open(img_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"警告：无法找到或打开图片 {img_path} 来获取尺寸，跳过 {json_path.name}。错误: {e}")
        return None

    yolo_annotations = []
    
    # 遍历JSON中的所有标注对象
    if 'frames' in data and data['frames']:
        for label in data['frames'][0].get('objects', []):
            category = label.get('category')
            if category in category_map and 'box2d' in label:
                class_id = category_map[category]
                
                # 提取边界框坐标
                box = label['box2d']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                
                # 转换为YOLO格式
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1
                
                # 归一化
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                width_norm = width / img_width
                height_norm = height / img_height
                
                yolo_annotations.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

    return "\n".join(yolo_annotations)

def main():
    """
    主函数，遍历所有.json文件并进行转换。
    """
    print("--- 开始将 BDD100K 的 .json 标签转换为 YOLO .txt 格式 ---")
    project_root = Path(__file__).parent.parent
    labels_base_path = project_root / "data" / "raw" / "bdd100k" / "labels"

    # 我们要处理 'train' 和 'val' 两个集合
    for split in ["train", "val"]:
        print(f"\n正在处理 {split} 集合...")
        json_dir = labels_base_path / split
        
        if not json_dir.exists():
            print(f"警告：找不到目录 {json_dir}，跳过。")
            continue

        json_files = list(json_dir.glob("*.json"))
        
        if not json_files:
            print(f"警告：在 {json_dir} 中没有找到任何 .json 文件。")
            continue

        for json_path in tqdm(json_files, desc=f"转换 {split} 标签"):
            yolo_content = convert_bdd_json_to_yolo(json_path)
            
            if yolo_content is not None:
                # 在同一个文件夹下，创建一个同名的.txt文件
                txt_path = json_path.with_suffix('.txt')
                with open(txt_path, 'w') as f:
                    f.write(yolo_content)
                    
    print("\n✅ 所有 .json 文件已成功转换为 .txt 格式！")

if __name__ == '__main__':
    main()
