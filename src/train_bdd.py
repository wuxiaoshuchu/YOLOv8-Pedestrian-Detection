# ================================================================= #
#  BDD100K 多类别检测训练脚本
# ================================================================= #

import numpy
from ultralytics import YOLO
import torch

def main():
    """
    主训练函数 - 训练BDD100K多类别模型。
    """
    # --- 1. 环境检查 ---
    print("--- 开始进行环境检查 ---")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ 成功检测到 {gpu_count} 块 GPU！")
        for i in range(gpu_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ 警告：未能检测到 CUDA GPU。")
    print("------------------------\n")

    # --- 2. 加载模型 ---
    # 我们依然从强大的 yoloV8m 预训练模型开始
    print("正在加载 yoloV8m 预训练模型...")
    model = YOLO('yolov8m.pt')
    print("✅ 模型加载成功！\n")

    # --- 3. 开始在BDD100K上进行多GPU训练！ ---
    print("--- 开始在 BDD100K 数据集上进行训练 ---")
    try:
        results = model.train(
            data='config/bdd100k.yaml', # 使用我们为BDD100K准备的新“地图”
            
            # --- 训练参数 ---
            epochs=50,          # BDD100K数据量大，先从50轮开始
            patience=10,        # BDD100K上过拟合会慢一些，patience设为10
            batch=32,           # 双3090可以轻松应对
            imgsz=640,
            device=[0, 1],
            
            # 为这次更宏大的实验起一个清晰的名字
            name='yolov8m_bdd100k_multiclass_v1' 
        )
        print("\n✅ BDD100K 训练成功完成！")
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")

    print("\n所有训练结果已保存在 'runs/' 文件夹下。")
    print("--------------------\n")

if __name__ == '__main__':
    main()
