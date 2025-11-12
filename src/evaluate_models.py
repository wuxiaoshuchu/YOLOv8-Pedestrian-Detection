import numpy  # 优先导入，避免MKL库冲突
from ultralytics import YOLO
from pathlib import Path
import torch

def main():
    """
    主函数，用于评估我们所有训练好的模型，
    并使用导师要求的 IoU=0.5 标准。
    """
    # --- 0. 环境和路径定义 ---
    print("--- 开始执行评估脚本 ---")
    device = 0 if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"✅ 成功检测到 GPU: {torch.cuda.get_device_name(device)}")
    
    project_root = Path(__file__).parent.parent
    
    # 定义两个冠军模型的路径
    pennfudan_model_path = project_root / "runs/detect/yolov8m_final_tuning_v4/weights/best.pt"
    bdd100k_model_path = project_root / "runs/detect/yolov8m_bdd100k_multiclass_v13/weights/best.pt"
    
    # 定义两个数据集配置文件的路径
    pennfudan_yaml = project_root / "config/pennfudan.yaml"
    bdd100k_yaml = project_root / "config/bdd100k.yaml"

    # --- 1. 任务 2.1 (A): 评估行人模型 (V4) ---
    print("\n\n" + "="*50)
    print("任务 1: 评估 Penn-Fudan (行人) V4 模型")
    print("标准: IoU=0.5, 验证集: Penn-Fudan")
    print("="*50)
    if not pennfudan_model_path.exists():
        print(f"❌ 错误：找不到行人模型: {pennfudan_model_path}")
    else:
        model_penn = YOLO(pennfudan_model_path)
        metrics_penn = model_penn.val(
            data=str(pennfudan_yaml), 
            iou=0.5,  # 按照导师要求，设置IoU阈值为0.5
            split='val',
            device=device
        )
        print("\n--- 行人模型 (V4) 评估结果 (IoU=0.5): ---")
        print(f"Precision (精确率): {metrics_penn.box.mp:.3f}")
        print(f"Recall (召回率):    {metrics_penn.box.mr:.3f}")
        print(f"mAP@50:             {metrics_penn.box.map50:.3f}")

    # --- 2. 任务 2.1 (B): 评估BDD100K模型 (V13) ---
    print("\n\n" + "="*50)
    print("任务 2: 评估 BDD100K (多类别) V13 模型")
    print("标准: IoU=0.5, 验证集: BDD100K")
    print("="*50)
    if not bdd100k_model_path.exists():
        print(f"❌ 错误：找不到BDD100K模型: {bdd100k_model_path}")
    else:
        model_bdd = YOLO(bdd100k_model_path)
        metrics_bdd = model_bdd.val(
            data=str(bdd100k_yaml), 
            iou=0.5,  # 按照导师要求，设置IoU阈值为0.5
            split='val',
            device=device
        )
        print("\n--- BDD100K模型 (V13) 评估结果 (IoU=0.5): ---")
        print(f"Precision (精确率): {metrics_bdd.box.mp:.3f}")
        print(f"Recall (召回率):    {metrics_bdd.box.mr:.3f}")
        print(f"mAP@50 (All):       {metrics_bdd.box.map50:.3f}")
        # 我们可以打印出每个类别的详细mAP50
        print("\n按类别分的 mAP@50:")
        for i, name in model_bdd.names.items():
            print(f"  - {name}: {metrics_bdd.box.maps[i]:.3f}")

    # --- 3. 旁支实验: 跨数据集测试 ---
    print("\n\n" + "="*50)
    print("任务 3: 跨数据集测试 (BDD100K模型 -> Penn-Fudan数据)")
    print("标准: IoU=0.5, 验证集: Penn-Fudan")
    print("="*50)
    if 'model_bdd' in locals():
        # 我们重用上面已经加载的 BDD100K 模型
        metrics_cross = model_bdd.val(
            data=str(pennfudan_yaml), # 使用 Penn-Fudan 的“地图”
            iou=0.5,
	    conf=0.1,
            split='val',
            device=device
        )
        print("\n--- 跨数据集测试结果 (BDD模型 vs Penn数据): ---")
        print(f"Precision (精确率): {metrics_cross.box.mp:.3f}")
        print(f"Recall (召回率):    {metrics_cross.box.mr:.3f}")
        print(f"mAP@50:             {metrics_cross.box.map50:.3f}")
    else:
        print("❌ BDD100K模型未加载，跳过跨数据集测试。")

    print("\n\n✅ 所有评估任务已完成！")

if __name__ == '__main__':
    main()
