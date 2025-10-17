import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy # 最好导入一下，以防万一

def main():
    """
    主函数，使用在BDD100K上训练的模型进行视频推理。
    """
    print("--- 开始使用BDD100K模型进行视频推理 ---")

    # --- 1. 定义路径 ---
    project_root = Path(__file__).parent.parent
    
    # 【重要】指定您新的“冠军模型”的路径
    # 使用 v13 的结果
    model_path = project_root / "runs/detect/yolov8m_bdd100k_multiclass_v13/weights/best.pt"
    
    # 【重要】指定您想要处理的输入视频的路径
    # 建议提前将测试视频放到 data/raw/ 目录下
    # 例如: data/raw/test_video_traffic.mp4
    input_video_path = project_root / "data/raw/test_video_traffic.mp4" 
    
    # 定义保存结果的输出视频路径
    output_video_path = project_root / "results/bdd_inference_output.mp4"
    output_video_path.parent.mkdir(exist_ok=True) # 如果results文件夹不存在就创建

    # --- 2. 加载模型 ---
    if not model_path.exists():
        print(f"❌ 错误：找不到模型文件: {model_path}")
        return
        
    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)
    print("✅ 模型加载成功！")

    # --- 3. 处理视频 ---
    if not input_video_path.exists():
        print(f"❌ 错误：找不到输入视频文件: {input_video_path}")
        return

    print(f"正在处理视频文件: {input_video_path}")
    # 使用 stream=True 可以更高效地处理视频流
    results_generator = model.predict(source=str(input_video_path), stream=True)
    
    # 准备使用OpenCV写入视频
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"❌ 错误：无法打开视频文件: {input_video_path}")
        return
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或者用 'avc1'
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_count = 0
    # 逐帧处理结果
    for results in results_generator:
        annotated_frame = results.plot() # 获取画好框的帧
        out.write(annotated_frame) # 写入新的视频文件
        frame_count += 1
        # 打印进度 (例如每100帧)
        if frame_count % 100 == 0:
            print(f"   ... 已处理 {frame_count} 帧 ...")
    
    # 清理资源
    cap.release()
    out.release()

    print(f"\n✅ 视频推理完成！ (共 {frame_count} 帧)")
    print(f"结果已保存到文件: {output_video_path}")

if __name__ == '__main__':
    main()
