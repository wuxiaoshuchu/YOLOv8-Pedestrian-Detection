import json
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def convert_bdd_json_to_yolo(json_path: Path, img_width: int, img_height: int) -> str:
    """
    读取单个BDD100K的.json文件，将其内容转换为YOLO格式的字符串。
    【【【V3 修正版】】】
    """
    with open(json_path) as f:
        data = json.load(f)

    # 【【【关键修正 1：修正了类别名称】】】
    # BDD100K的官方名称是 'motor' 和 'bike'
    category_map = {
        'person': 0,       # 将 'person' 映射为 行人
        'pedestrian': 0,   # 将 'pedestrian' 也映射为 行人
        'rider': 1,
        'car': 2,
        'truck': 3,
        'bus': 4,
        'train': 5,
        'motor': 6,        # 修正：'motorcycle' -> 'motor'
        'bike': 7,         # 修正：'bicycle' -> 'bike'
        'traffic light': 8,
        'traffic sign': 9
    }

    yolo_annotations = []

    if 'frames' in data and data['frames']:
        for label in data['frames'][0].get('objects', []):
            category = label.get('category')

            # 【【【关键修正 2：使用 .get() 来安全处理】】】
            class_id = category_map.get(category)

            # 只有当类别在我们关心的地图中时才处理
            if class_id is not None and 'box2d' in label:
                box = label['box2d']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1

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
    print("--- 开始将 BDD100K 的 .json 标签转换为 YOLO .txt 格式 (V3 最终修正版) ---")
    project_root = Path(__file__).parent.parent
    data_base_path = project_root / "data" / "raw" / "bdd100k"
    images_base_path = data_base_path / "images"
    labels_base_path = data_base_path / "labels"

    for split in ["train", "val"]:
        print(f"\n正在处理 {split} 集合...")

        image_dir = images_base_path / split
        if not image_dir.exists():
            print(f"警告：图片目录 {image_dir} 未找到，跳过。")
            continue

        available_images = {p.stem for p in image_dir.glob("*.jpg")}
        print(f"在 {split} 集合中找到了 {len(available_images)} 张可用的图片。")

        json_dir = labels_base_path / split
        if not json_dir.exists():
            print(f"警告：标签目录 {json_dir} 未找到，跳过。")
            continue

        json_files = list(json_dir.glob("*.json"))

        for json_path in tqdm(json_files, desc=f"转换 {split} 标签"):
            base_name = json_path.stem

            if base_name in available_images:
                try:
                    img_path = image_dir / f"{base_name}.jpg"
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size

                    yolo_content = convert_bdd_json_to_yolo(json_path, img_width, img_height)

                    # 即使yolo_content为空（图片中没有我们关心的类别），也要创建一个空的.txt文件
                    # 这对YOLOv8的训练很重要（作为负样本）
                    txt_path = json_path.with_suffix('.txt')
                    with open(txt_path, 'w') as f:
                        f.write(yolo_content)
                except Exception as e:
                    print(f"\n错误：处理文件 {json_path.name} 时发生意外错误: {e}")
            else:
                pass

    print("\n✅ V3 最终版 .txt 文件已生成完毕！")

if __name__ == '__main__':
    main()
