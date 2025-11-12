# ================================================================= #
#  【V5 最终完整版】序贯微调：从 BDD100K 通才 -> Penn-Fudan 专才
# ================================================================= #

import numpy
from ultralytics import YOLO
import torch
from pathlib import Path

def main():
    """
    主训练函数 - V5 序贯微调版。
    """
    # --- 1. 环境检查 (不变) ---
    print("--- 开始进行环境检查 ---")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ 成功检测到 {gpu_count} 块 GPU！")
    else:
        print("❌ 警告：未能检测到 CUDA GPU。")
    print("------------------------\n")
    
    project_root = Path(__file__).parent.parent

    # --- 2. 【【【关键修改 1】】】 ---
    # 我们加载的不再是 'yolov8m.pt'
    # 而是我们自己训练好的“交通通才”模型！
    bdd_model_path = project_root / "runs/detect/yolov8m_bdd100k_multiclass_v13/weights/best.pt"
    
    if not bdd_model_path.exists():
        print(f"❌ 错误：找不到 BDD100K 模型: {bdd_model_path}")
        return

    print(f"正在加载我们自己的 BDD100K 预训练模型: {bdd_model_path}")
    model = YOLO(bdd_model_path)
    print("✅ BDD100K 模型加载成功！\n")

    # --- 3. 开始在 Penn-Fudan 上进行“微调”训练 ---
    print("--- 开始在 Penn-Fudan 上进行专项微调 (V5) ---")
    try:
        results = model.train(
            data='config/pennfudan.yaml', # 我们的目标是行人 (单类别)
            
            # 【【【关键修改 2】】】
            # 使用一个极低的学习率，因为我们只是“微调”一个已经很聪明的模型
            lr0=0.0001,
            
            # 我们依然可以使用V4的优化参数
            copy_paste=0.3,
            weight_decay=0.001,
            epochs=30,          # 30轮足够微调了
            patience=10,        
            
            # --- 其他硬件/项目参数 ---
            imgsz=640,
            device=[0, 1], 
            batch=32,
            
            # 为这次“王牌”实验起个名字
            name='yolov8m_bdd_finetuned_on_penn_v5' 
        )
        print("\n✅ V5 序贯微调训练成功完成！")
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")

    print("\n所有训练结果已保存在 'runs/' 文件夹下。")
    print("--------------------\n")

if __name__ == '__main__':
    main()
