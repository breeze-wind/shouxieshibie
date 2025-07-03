import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(data_dir):
    """
    从按类别分文件夹的结构中加载数据

    参数:
    data_dir: 数据集根目录，包含多个子文件夹，每个子文件夹名代表一个类别

    返回:
    signals: 特征数组
    labels: 标签数组
    label_names: 标签名称映射表
    """
    signals = []
    labels = []
    label_map = {}

    # 获取所有子文件夹（类别）
    class_folders = [f for f in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, f))]

    # 为每个类别分配数字编码
    for label_id, class_folder in enumerate(class_folders):
        class_path = os.path.join(data_dir, class_folder)
        label_map[label_id] = class_folder  # 保存标签映射关系

        # 遍历当前类别文件夹中的所有文件
        for file_name in os.listdir(class_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(class_path, file_name)

                # 读取CSV文件（假设电阻值在第3列）
                df = pd.read_csv(file_path)
                signal = df.iloc[:, 2].values  # 获取电阻值列

                signals.append(signal)
                labels.append(label_id)

    # 转换为NumPy数组并调整形状
    signals = np.array(signals)
    if signals.ndim == 2:
        signals = signals.reshape(-1, 1)  # 如果只有一个特征

    labels = np.array(labels)

    return signals, labels, label_map


# 使用示例
data_dir = "path/to/your/dataset"  # 数据集根目录
signals, labels, label_map = load_data(data_dir)

print(f"加载了 {len(signals)} 个样本")
print("标签映射:", label_map)
__main__: