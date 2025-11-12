from pytube import YouTube
from pathlib import Path

# 【【【已为您填好视频链接】】】
video_url = "https://www.youtube.com/watch?v=hJJL127rpZ4"

# 定义保存的文件夹和文件名
save_dir = Path(__file__).parent.parent / "data" / "raw" / "tokyo_youtube_video"
save_dir.mkdir(exist_ok=True)

try:
    print(f"正在连接到: {video_url}")
    yt = YouTube(video_url)

    print(f"正在获取视频流: {yt.title}")
    # 获取720p的.mp4格式，这对于分析足够清晰，且文件不会过大
    stream = yt.streams.filter(progressive=True, file_extension='mp4', res="720p").first()

    if not stream:
        # 如果没有720p，就获取它能找到的第一个mp4流
        print("未找到720p，正在尝试获取其他mp4流...")
        stream = yt.streams.filter(progressive=True, file_extension='mp4').first()

    if stream:
        print(f"正在下载视频，保存到: {save_dir}")
        # 我们给它起一个固定的名字，方便后续脚本调用
        stream.download(output_path=str(save_dir), filename="tokyo_drive_test.mp4")
        print("✅ 视频下载完成！")
    else:
        print("❌ 错误：找不到合适的 .mp4 视频流。")

except Exception as e:
    print(f"❌ 下载过程中发生错误: {e}")
