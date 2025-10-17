# ================================================================= #
#  【最终完整版 V4】增加耐心，让数据增强充分发挥作用
# ================================================================= #

import numpy
from ultralytics import YOLO
import torch

def main():
    """
    主训练函数 - V4 最终调优版。
    """
    # --- 1. 环境检查 (不变) ---
    print("--- 开始进行环境检查 ---")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ 成功检测到 {gpu_count} 块 GPU！")
        for i in range(gpu_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ 警告：未能检测到 CUDA GPU。")
    print("------------------------\n")

    # --- 2. 加载模型 (不变) ---
    print("正在加载 yoloV8m 预训练模型...")
    model = YOLO('yolov8m.pt')
    print("✅ 模型加载成功！\n")

    # --- 3. 开始最终的多GPU训练！ ---
    print("--- 开始多GPU模型训练 (V4 - 最终调优版) ---")
    try:
        results = model.train(
            data='config/pennfudan.yaml',
            
            # --- V3的参数我们全部保留 ---
            copy_paste=0.3,
            lr0=0.001,
            weight_decay=0.001,
            imgsz=640,
            device=[0, 1], 
            batch=32,
            
            # 【【【关键修改 1】】】
            # 大幅增加耐心值，给模型更多机会去寻找更优解
            patience=50,
            
            # 【【【关键修改 2】】】
            # 配合patience，我们也增加总的训练轮次上限
            epochs=100,
            
            # 为这次最终的实验起一个清晰的名字
            name='yolov8m_final_tuning_v4' 
        )
        print("\n✅ 最终调优版训练成功完成！")
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")

    print("\n所有训练结果已保存在 'runs/' 文件夹下。")
    print("--------------------\n")

if __name__ == '__main__':
    main()
