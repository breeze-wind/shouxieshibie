import os
import pandas as pd
import numpy as np
import chardet  # 需要安装：pip install chardet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(data_dir, column_index=2, file_extension='.csv',
              normalize=False, handle_missing=True, test_size=None, random_state=42):
    """
    从按类别分文件夹的结构中加载数据

    参数:
    data_dir: 数据集根目录，包含多个子文件夹，每个子文件夹名代表一个类别
    column_index: 需要提取的列索引，默认为第3列(索引2)
    file_extension: 需要加载的文件扩展名，默认为.csv
    normalize: 是否对数据进行标准化，默认为False
    handle_missing: 是否处理缺失值，默认为True
    test_size: 若提供，则将数据划分为训练集和测试集
    random_state: 随机种子，用于数据划分

    返回:
    如果test_size为None:
        signals: 特征数组
        labels: 标签数组
        label_names: 标签名称映射表
    否则:
        X_train, X_test, y_train, y_test, label_names
    """
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据集目录 '{data_dir}' 不存在")

    signals = []
    labels = []
    label_map = {}
    max_length = 0

    # 获取所有子文件夹（类别）
    class_folders = [f for f in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, f))]

    if not class_folders:
        raise ValueError(f"目录 '{data_dir}' 不包含任何子文件夹")

    # 为每个类别分配数字编码
    for label_id, class_folder in enumerate(class_folders):
        class_path = os.path.join(data_dir, class_folder)
        label_map[label_id] = class_folder

        # 遍历当前类别文件夹中的所有文件
        file_count = 0
        for file_name in os.listdir(class_path):
            if file_name.endswith(file_extension):
                file_path = os.path.join(class_path, file_name)

                try:
                    # 尝试检测文件编码
                    with open(file_path, 'rb') as f:
                        result = chardet.detect(f.read(10000))

                    # 使用检测到的编码读取文件
                    df = pd.read_csv(file_path, encoding=result['encoding'])

                    # 检查列索引是否有效
                    if column_index >= df.shape[1]:
                        raise IndexError(f"文件 '{file_path}' 不包含索引为 {column_index} 的列")

                    # 获取指定列
                    signal = df.iloc[:, column_index].values

                    # 更新最大长度
                    if len(signal) > max_length:
                        max_length = len(signal)

                    signals.append(signal)
                    labels.append(label_id)
                    file_count += 1
                    print(f"成功加载: {file_path} (编码: {result['encoding']})")

                except Exception as e:
                    print(f"警告: 无法加载文件 '{file_path}': {str(e)}")

        if file_count == 0:
            print(f"警告: 类别文件夹 '{class_folder}' 不包含任何{file_extension}文件")

    # 检查是否加载了任何数据
    if not signals:
        raise ValueError("没有找到可加载的数据文件")

    # 处理变长序列：填充到最大长度
    if any(len(s) != max_length for s in signals):
        print(f"警告: 发现变长序列，将填充到最大长度 {max_length}")
        signals = [np.pad(s, (0, max_length - len(s)), 'constant') for s in signals]

    # 转换为NumPy数组
    signals = np.array(signals)
    labels = np.array(labels)

    # 处理缺失值
    if handle_missing:
        signals = np.nan_to_num(signals)

    # 标准化数据
    if normalize:
        scaler = StandardScaler()
        signals = scaler.fit_transform(signals)

    # 划分数据集
    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            signals, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        return X_train, X_test, y_train, y_test, label_map
    else:
        return signals, labels, label_map


# 使用示例
if __name__ == "__main__":
    data_dir = "dataset"  # 数据集根目录

    try:
        # 简单加载
        signals, labels, label_map = load_data(data_dir)
        print(f"加载了 {len(signals)} 个样本")
        print("标签映射:", label_map)

    except Exception as e:
        print(f"错误: {str(e)}")
