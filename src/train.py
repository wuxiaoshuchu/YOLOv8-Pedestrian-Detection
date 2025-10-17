import numpy
from ultralytics import YOLO
import torch

def main():
    """
    主训练函数 - 优化版 V2。
    """
    # --- 1. 环境检查 ---
    print("--- 开始进行环境检查 ---")
    if not torch.cuda.is_available():
        print("❌ 警告：未能检测到 CUDA GPU。")
    else:
        gpu_count = torch.cuda.device_count()
        print(f"✅ 成功检测到 {gpu_count} 块 GPU！")
        for i in range(gpu_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    print("------------------------\n")

    # --- 2. 加载模型 ---
    # 【【【这一步至关重要，必须在 .train() 之前】】】
    print("正在加载 yoloV8m 预训练模型...")
    model = YOLO('yolov8m.pt')
    print("✅ 模型加载成功！\n")

    # --- 3. 开始多GPU训练！ ---
    print("--- 开始多GPU模型训练 (V2 - 优化版) ---")
    try:
        # 现在，我们用上面加载好的 'model' 变量来开始训练
        results = model.train(
            data='config/pennfudan.yaml',
            
            # --- 优化后的超参数 ---
            lr0=0.001,          # 大幅降低初始学习率
            epochs=30,          # 减少训练轮次
            weight_decay=0.001, # 稍微增加权重衰减
            
            # --- 其他参数 ---
            imgsz=640,
            device=[0, 1], 
            batch=32,
            patience=20,
            name='yolov8m_3090_low_lr_v2' # 为这次优化实验起一个新名字
        )
        print("\n✅ 优化版训练成功完成！")
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")

    print("\n所有训练结果已保存在 'runs/' 文件夹下。")
    print("--------------------\n")

if __name__ == '__main__':
    main()
