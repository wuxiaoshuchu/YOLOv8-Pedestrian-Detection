# ================================================================= #
#  【V14 最终平衡版】 - 解决数据不平衡问题
# ================================================================= #

import numpy
from ultralytics import YOLO
import torch
from pathlib import Path

def main():
    """
    主训练函数 - V14 最终平衡版
    目标：解决BDD100K的数据不平衡问题，提升稀有类别（如pedestrian）的性能
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

    # --- 2. 加载模型 (不变) ---
    print("正在加载 yoloV8m 预训练模型...")
    model = YOLO('yolov8m.pt')
    print("✅ 模型加载成功！\n")

    # --- 3. 开始在BDD100K上进行“平衡化”训练！ ---
    print("--- 开始在 BDD100K 数据集上进行“平衡化”训练 (V14) ---")
    try:
        results = model.train(
            data='config/bdd100k.yaml', # 依然使用BDD100K的“地图”
            
            # 【【【关键修改 1：延长训练时间】】】
            # 50轮对于这个量级的数据集只是“热身”
            # 我们给它更长的时间来学习和收敛
            epochs=100,
            
            # 【【【关键修改 2：开启数据增强】】】
            # 开启Copy-Paste，这会随机粘贴物体，
            # 极大地帮助“稀有类别”（如pedestrian, train）增加它们的出场率！
            copy_paste=0.3,
            
            # 【【【关键修改 3：调整损失函数权重】】】
            # cls=0.5 (默认)
            # 我们可以尝试稍微提高“分类损失”的权重，让模型更努力地区分10个类别
            # 但我们先保持默认值，优先看Copy-Paste的效果
            
            # --- 其他参数 ---
            patience=20, # 我们需要给它更长的耐心，因为它学得更久
            batch=32,
            imgsz=640,
            device=[0, 1],
            
            # 为这次更宏大的实验起一个清晰的名字
            name='yolov8m_bdd100k_balanced_v14' 
        )
        print("\n✅ BDD100K 平衡化训练成功完成！")
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")

    print("\n所有训练结果已保存在 'runs/' 文件夹下。")
    print("--------------------\n")

if __name__ == '__main__':
    main()
