# ================================================================= #
#  BDD100K 训练脚本 - 【V15 最终修正版】
# ================================================================= #

import numpy
from ultralytics import YOLO
import torch
from pathlib import Path

def main():
    """
    主训练函数 - 训练BDD100K多类别模型（V15 - 数据修正版）
    """
    # --- 1. 环境检查 ---
    print("--- 开始进行环境检查 ---")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ 成功检测到 {gpu_count} 块 GPU！")
    else:
        print("❌ 警告：未能检测到 CUDA GPU。")
    print("------------------------\n")

    # --- 2. 加载模型 ---
    print("正在加载 yoloV8m 预训练模型...")
    model = YOLO('yolov8m.pt')
    print("✅ 模型加载成功！\n")

    # --- 3. 开始在BDD100K上进行多GPU训练！ ---
    print("--- 开始在 BDD100K 数据集上进行训练 (V15 - 数据修正版) ---")
    try:
        results = model.train(
            data='config/bdd100k.yaml', # 使用我们的“地图”

            # --- 训练参数 (使用V13的设置) ---
            epochs=50,
            patience=10,

            batch=32,
            imgsz=640,
            device=[0, 1],

            # 为这次“补考”起一个清晰的名字
            name='yolov8m_bdd100k_FIXED_v15' 
        )
        print("\n✅ BDD100K (数据修正版) 训练成功完成！")

    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")

    print("\n所有训练结果已保存在 'runs/' 文件夹下。")
    print("--------------------\n")

if __name__ == '__main__':
    main()
