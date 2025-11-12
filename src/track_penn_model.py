import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy # 最好导入一下

def main():
    """
    主函数，使用在BDD100K上训练的模型进行视频目标追踪。
    """
    print("--- 开始使用BDD100K模型进行视频目标追踪 ---")

    # --- 1. 定义路径 ---
    project_root = Path(__file__).parent.parent
    
    # 使用我们训练好的多类别模型
    model_path = project_root / "runs/detect/yolov8m_final_tuning_v4/weights/best.pt" # 请确保这是您正确的模型路径！
    
    # 输入视频路径 (和之前一样)
    input_video_path = project_root / "data/raw/13142111_2160_3840_30fps.mp4" 
    
    # 定义保存追踪结果的输出视频路径
    output_video_path = project_root / "results/china_traffic_tracking_PENN_MODEL_output.mp4"
    output_video_path.parent.mkdir(exist_ok=True)

    # --- 2. 加载模型 ---
    if not model_path.exists():
        print(f"❌ 错误：找不到模型文件: {model_path}")
        return
        
    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)
    print("✅ 模型加载成功！")

    # --- 3. 处理视频并进行追踪 ---
    if not input_video_path.exists():
        print(f"❌ 错误：找不到输入视频文件: {input_video_path}")
        return

    print(f"正在处理视频文件并进行追踪: {input_video_path}")
    
    # 【【【关键修改！开启追踪功能】】】
    # 我们调用 model.track() 而不是 model.predict()
    # tracker='bytetrack.yaml' 指定使用ByteTrack算法
    # persist=True 让追踪器记住跨帧的对象
    results_generator = model.track(source=str(input_video_path), tracker='bytetrack.yaml', persist=True, stream=True)
    
    # 准备写入视频 (和之前一样)
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"❌ 错误：无法打开视频文件: {input_video_path}")
        return
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_count = 0
    # 逐帧处理追踪结果
    for results in results_generator:
        # results.plot() 会自动画出带有ID的追踪框！
        annotated_frame = results.plot() 
        out.write(annotated_frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"   ... 已处理 {frame_count} 帧 ...")
    
    # 清理资源
    cap.release()
    out.release()

    print(f"\n✅ 视频追踪完成！ (共 {frame_count} 帧)")
    print(f"结果已保存到文件: {output_video_path}")

if __name__ == '__main__':
    main()
