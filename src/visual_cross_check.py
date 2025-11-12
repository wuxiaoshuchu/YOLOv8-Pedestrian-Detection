import numpy
from ultralytics import YOLO
from pathlib import Path
import glob
import torch

def main():
    """
    主函数，用于【视觉验证】(定性分析)
    BDD100K模型在Penn-Fudan验证集上的真实表现。
    """
    print("--- 开始执行跨数据集的【视觉验证】 ---")
    device = 0 if torch.cuda.is_available() else 'cpu'

    # --- 1. 定义路径 ---
    project_root = Path(__file__).parent.parent
    
    # 定义“通才”模型 (BDD100K V13)
    bdd_model_path = project_root / "runs/detect/yolov8m_bdd100k_multiclass_v13/weights/best.pt"
    
    # 【【【关键】】】
    # 定义我们要测试的图片来源：Penn-Fudan的验证集图片
    penn_val_images_dir = project_root / "data/processed/images/val"
    
    # 定义结果保存的文件夹
    output_dir = project_root / "results/cross_check_bdd_on_penn/"
    output_dir.mkdir(exist_ok=True)

    # --- 2. 加载模型 ---
    print(f"正在加载“交通通才”模型...")
    model_bdd = YOLO(bdd_model_path)
    print("✅ 模型加载成功！")

    # --- 3. 遍历Penn-Fudan图片并进行预测 ---
    if not penn_val_images_dir.exists():
        print(f"❌ 错误：找不到Penn-Fudan的验证图片: {penn_val_images_dir}")
        return
            
    print(f"\n--- 正在对Penn-Fudan验证集图片进行预测... ---")
    
    # 【【【关键修改】】】
    # 我们使用 model.predict() 而不是 model.val()
    # 同时，我们把置信度阈值(conf)设置得非常低，比如0.1，
    # 看看模型到底有没有识别出来！
    results = model_bdd.predict(
        source=str(penn_val_images_dir),
        save=True,      # 自动保存结果
        conf=0.1,       # 使用一个很低的“及格线”
        project=str(output_dir), # 指定保存的项目目录
        name="inference_results",
        device=device,
        exist_ok=True   # 允许覆盖旧结果
    )
    
    print(f"\n✅ 视觉验证完成！")
    print(f"请下载并查看 'results/cross_check_bdd_on_penn/inference_results/' 目录下的图片。")

if __name__ == '__main__':
    main()
