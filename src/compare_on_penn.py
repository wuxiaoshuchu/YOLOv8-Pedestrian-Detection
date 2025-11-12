import numpy
from ultralytics import YOLO
from pathlib import Path
import torch

def main():
    """
    主函数，用于【同台竞技】
    比较“行人专才”模型和“交通通才”模型
    在Penn-Fudan验证集上的真实表现。
    """
    print("--- 开始执行模型“同台竞技”对比 ---")
    device = 0 if torch.cuda.is_available() else 'cpu'

    # --- 1. 定义路径 ---
    project_root = Path(__file__).parent.parent
    
    # 定义“专才”模型 (Penn-Fudan V4)
    penn_model_path = project_root / "runs/detect/yolov8m_final_tuning_v4/weights/best.pt"
    # 定义“通才”模型 (BDD100K V13)
    bdd_model_path = project_root / "runs/detect/yolov8m_bdd100k_multiclass_v13/weights/best.pt"
    
    # 定义我们要测试的图片来源：Penn-Fudan的验证集图片
    penn_val_images_dir = project_root / "data/processed/images/val"
    
    # 【【【重要】】】我们为两个模型创建各自的结果文件夹
    output_dir_penn = project_root / "results/COMPARE_penn_model_on_penn/"
    output_dir_bdd = project_root / "results/COMPARE_bdd_model_on_penn/"
    output_dir_penn.mkdir(exist_ok=True)
    output_dir_bdd.mkdir(exist_ok=True)

    # --- 2. 加载模型 ---
    print(f"正在加载“行人专才”模型 (V4)...")
    model_penn = YOLO(penn_model_path)
    print(f"正在加载“交通通才”模型 (V13)...")
    model_bdd = YOLO(bdd_model_path)
    print("✅ 所有模型加载成功！")

    # --- 3. “行人专才”进行预测 ---
    if not penn_val_images_dir.exists():
        print(f"❌ 错误：找不到Penn-Fudan的验证图片: {penn_val_images_dir}")
        return
            
    print(f"\n--- 正在使用“行人专才”模型进行预测... ---")
    model_penn.predict(
        source=str(penn_val_images_dir),
        save=True,
        conf=0.25, # 使用一个标准的置信度
        project=str(output_dir_penn),
        name="inference_results",
        device=device,
        exist_ok=True
    )
    print(f"✅ “行人专才”模型预测完成，结果已保存到: {output_dir_penn}")

    # --- 4. “交通通才”进行预测 ---
    print(f"\n--- 正在使用“交通通才”模型进行预测... ---")
    model_bdd.predict(
        source=str(penn_val_images_dir),
        save=True,
        conf=0.1, # 我们仍然使用一个较低的置信度，给它一个机会
        project=str(output_dir_bdd),
        name="inference_results",
        device=device,
        exist_ok=True
    )
    print(f"✅ “交通通才”模型预测完成，结果已保存到: {output_dir_bdd}")

    print("\n\n✅ 所有对比预测已完成！请下载并对比 'results/' 目录下的两个新文件夹。")

if __name__ == '__main__':
    main()
