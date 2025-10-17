import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    """
    主函数，用于读取YOLOv8训练结果并绘制学习曲线。
    """
    print("--- 开始绘制V4版本的学习曲线 ---")

    # --- 1. 定义路径 ---
    project_root = Path(__file__).parent.parent
    # 我们要分析的是V4版本的结果
    results_path = project_root / "runs" / "detect" / "yolov8m_final_tuning_v4" / "results.csv"
    
    # 定义图表保存的位置
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True) # 如果results文件夹不存在，就创建一个
    output_path = output_dir / "learning_curves_v4.png"

    # --- 2. 检查并读取数据 ---
    if not results_path.exists():
        print(f"❌ 错误：在路径 {results_path} 未找到 results.csv 文件。")
        print("请确认您的V4版本训练是否已成功完成，并且结果文件夹名称正确。")
        return

    print(f"正在从 {results_path} 读取数据...")
    df = pd.read_csv(results_path)
    
    # 清理列名中可能存在的多余空格
    df.columns = df.columns.str.strip()

    # --- 3. 绘制图表 ---
    print("正在生成学习曲线图...")
    
    # 设置图表风格
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # 绘制左侧Y轴：mAP 和 Recall
    color = 'tab:blue'
    ax1.set_xlabel('Epoch (轮次)', fontsize=14)
    ax1.set_ylabel('Performance (性能)', color=color, fontsize=14)
    sns.lineplot(data=df, x='epoch', y='metrics/mAP50-95(B)', ax=ax1, color=color, label='mAP@.50-.95 (综合性能)')
    sns.lineplot(data=df, x='epoch', y='metrics/recall(B)', ax=ax1, color='tab:green', label='Recall (召回率)')
    ax1.tick_params(axis='y', labelcolor=color)

    # 找到最佳轮次并标记
    best_epoch = df['metrics/mAP50-95(B)'].idxmax()
    best_mAP = df.loc[best_epoch, 'metrics/mAP50-95(B)']
    
    # 在图上画一条垂直线标记最佳轮次
    ax1.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch} (mAP={best_mAP:.3f})')
    
    # 创建右侧Y轴，用于绘制损失
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Loss (损失)', color=color, fontsize=14)
    sns.lineplot(data=df, x='epoch', y='train/box_loss', ax=ax2, color=color, linestyle=':', label='Train Box Loss (定位损失)')
    ax2.tick_params(axis='y', labelcolor=color)

    # --- 4. 美化并保存图表 ---
    fig.suptitle('YOLOv8m Final Tuning (V4) - Learning Curves', fontsize=20)
    # 合并两个Y轴的图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')
    
    # 移除ax1的独立图例
    ax1.get_legend().remove()
    
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局防止标题重叠
    
    plt.savefig(output_path)
    print(f"\n✅ 学习曲线图已成功保存到: {output_path}")

if __name__ == '__main__':
    main()
