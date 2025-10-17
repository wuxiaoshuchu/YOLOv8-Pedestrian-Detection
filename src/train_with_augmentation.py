# ================================================================= #
#  【最终完整版 V3】引入 Copy-Paste 数据增强以提升召回率
# ================================================================= #

import numpy  # 优先导入numpy，这是一个好习惯，可以避免一些底层库冲突
from ultralytics import YOLO
import torch

def main():
    """
    主训练函数 - V3 Copy-Paste增强版。
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
    print("正在加载 yoloV8m 预训练模型...")
    model = YOLO('yolov8m.pt')
    print("✅ 模型加载成功！\n")

    # --- 3. 开始多GPU训练！ ---
    print("--- 开始多GPU模型训练 (V3 - Copy-Paste增强版) ---")
    try:
        results = model.train(
            data='config/pennfudan.yaml',
            
            # 【【【关键修改】】】
            # 开启Copy-Paste数据增强，0.3表示有30%的概率对每个批次应用此增强
            # 这是提升召回率、解决遮挡和小目标的利器！
            copy_paste=0.3,

            # --- V2中优化好的超参数保持不变 ---
            lr0=0.001,          # 较低的学习率
            epochs=30,          # 30个轮次
            weight_decay=0.001, # 权重衰减
            
            # --- 其他硬件/项目参数 ---
            imgsz=640,
            device=[0, 1], 
            batch=32,
            patience=20,
            
            # 为这次最终的优化实验起一个全新的名字，以便区分
            name='yolov8m_copy_paste_v3' 
        )
        print("\n✅ Copy-Paste增强版训练成功完成！")
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")

    print("\n所有训练结果已保存在 'runs/' 文件夹下。")
    print("--------------------\n")

if __name__ == '__main__':
    main()
