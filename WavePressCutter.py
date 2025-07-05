import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm
import chardet

# 配置参数
CONFIG = {
    "input_dir": "dataset/a_bk",  # 原始数据目录
    "output_dir": "processed_data/a_bk",  # 处理后数据保存目录
    "interruption_threshold": 0.1,  # 连续性中断的最小回升比例（用于旧方法）
    "descent_threshold": 0.05,  # 书写下降的最小幅度比例（用于旧方法）
    "min_writing_length": 100,  # 最小有效书写长度
    "seq_length": 800,  # 标准化序列长度
    "visualize_samples": True,  # 是否可视化处理结果
    "filter_window": 7,  # 滤波窗口大小
    "filter_polyorder": 3,  # 滤波多项式阶数
    "resistance_column": "DeviceInfo_[23082300]",  # 电阻值所在列名
    "normalize_data": True,  # 是否对数据进行归一化
    "use_dynamic_thresholds": False,  # 是否使用动态阈值
    "window_size": 50,  # 局部阈值计算窗口大小
    "keep_press_segment": False,  # 是否保留按纸过程
    "peak_threshold_ratio": 0.7,  # 极大值阈值比例（相对于平均极大值）
    "min_segment_length": 50,  # 最小分割段长度
    "use_derivative_method": True,  # 是否使用导数拐点法
}


def load_data(file_path):
    """加载CSV数据，处理复合表头，从第3行开始加载"""
    try:
        # 1. 检测文件编码
        with open(file_path, 'rb') as f:
            raw_data = f.read(1024)
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']

        encodings = [detected_encoding, 'utf-8', 'gbk', 'gb2312', 'latin-1']

        # 2. 尝试每种编码加载文件
        for encoding in encodings:
            try:
                # 读取前5行用于分析
                preview = pd.read_csv(file_path, encoding=encoding, nrows=5)

                # 3. 重新设置列名并从第3行(index=2)开始加载数据
                df = pd.read_csv(file_path, encoding=encoding, header=None)
                df.columns = ['DeviceNo', 'Unnamed: 1', 'DeviceInfo_[23082300]', 'DeviceRange_[--------]', 'Unnamed: 4']
                df = df[2:]
                df = df.reset_index(drop=True)

                print(f"成功使用 {encoding} 编码加载文件，跳过2行表头")

                # 确保电阻值列存在
                if CONFIG["resistance_column"] not in df.columns:
                    print(f"错误：文件中不存在列 {CONFIG['resistance_column']}")
                    return None

                # 尝试将电阻值列转换为数值类型
                try:
                    resistance = pd.to_numeric(df[CONFIG["resistance_column"]], errors='coerce')
                    # 检查是否有大量非数值数据
                    if resistance.isna().sum() > len(resistance) * 0.1:
                        print(f"警告：电阻值列包含过多非数值数据（{resistance.isna().sum()}个无效值）")

                    # 删除无效值
                    resistance = resistance.dropna().values
                    if len(resistance) == 0:
                        print("错误：处理后电阻值列为空")
                        return None

                    # 打印数据统计特征
                    print(
                        f"数据统计: min={np.min(resistance):.2f}, max={np.max(resistance):.2f}, mean={np.mean(resistance):.2f}, std={np.std(resistance):.2f}")

                    return resistance
                except Exception as e:
                    print(f"转换电阻值列为数值类型时出错: {e}")
                    return None

            except UnicodeDecodeError:
                continue

        print(f"无法使用任何编码加载文件 {file_path}")
        return None

    except Exception as e:
        print(f"加载文件 {file_path} 失败: {e}")
        return None


def normalize(resistance):
    """将电阻值归一化到[0,1]区间"""
    if len(resistance) == 0:
        return resistance
    min_val = np.min(resistance)
    max_val = np.max(resistance)
    if max_val == min_val:
        return np.zeros_like(resistance)
    return (resistance - min_val) / (max_val - min_val)


def calculate_dynamic_thresholds(resistance, config):
    """根据电阻值的统计特性计算动态阈值"""
    if len(resistance) < 10:
        return config["interruption_threshold"], config["descent_threshold"]  # 使用默认值

    # 计算全局统计特征
    std_dev = np.std(resistance)
    mean_val = np.mean(resistance)

    # 动态调整阈值，范围限制在合理区间
    interruption_threshold = max(0.05, min(0.2, 0.1 * (std_dev / mean_val)))
    descent_threshold = max(0.02, min(0.1, 0.05 * (std_dev / mean_val)))

    return interruption_threshold, descent_threshold


def calculate_local_thresholds(resistance, window_size=50):
    """计算局部区域的动态阈值"""
    thresholds = np.zeros_like(resistance, dtype=float)

    for i in range(len(resistance)):
        # 定义局部窗口
        start = max(0, i - window_size // 2)
        end = min(len(resistance), i + window_size // 2)
        local_data = resistance[start:end]

        # 计算局部标准差
        local_std = np.std(local_data)
        local_mean = np.mean(local_data)

        # 基于局部统计特性计算阈值
        thresholds[i] = max(0.02, min(0.1, 0.05 * (local_std / local_mean)))

    return thresholds


def preprocess(resistance, config):
    """数据预处理：应用Savitzky-Golay滤波平滑数据"""
    if len(resistance) < config["filter_window"]:
        return resistance
    return savgol_filter(resistance, config["filter_window"], config["filter_polyorder"])


def split_by_continuity(resistance, config):
    """基于拐点检测分割按纸和书写过程"""
    if len(resistance) < 10:
        return [0, len(resistance)]  # 数据过短，直接返回

    # 归一化数据（如果配置启用）
    processed_resistance = normalize(resistance) if config["normalize_data"] else resistance

    # 1. 检测初始电压结束位置
    initial_voltage = np.mean(processed_resistance[:10])
    initial_end = np.where(processed_resistance < 0.9 * initial_voltage)[0][0] if len(
        np.where(processed_resistance < 0.9 * initial_voltage)[0]) > 0 else 10

    # 如果不使用导数方法，回退到旧的连续性中断方法
    if not config["use_derivative_method"]:
        # 旧方法代码（保持不变）
        # ... 此处省略旧方法代码 ...
        return old_split_method(processed_resistance, config, initial_end)

    # 2. 使用一阶导数检测拐点（极大值和极小值）
    # 计算一阶导数（差分近似）
    derivative = np.diff(processed_resistance)
    # 计算二阶导数（差分的差分）
    second_derivative = np.diff(derivative)

    # 寻找拐点（二阶导数符号变化的点）
    inflection_points = []
    for i in range(len(second_derivative) - 1):
        # 二阶导数符号变化，表示存在拐点
        if second_derivative[i] * second_derivative[i + 1] <= 0:
            inflection_points.append(i + 1)  # +1 是因为导数数组长度减1

    # 3. 过滤拐点：只保留高于阈值的极大值点作为分割点
    # 计算极小值和极大值点
    minima_points = []
    maxima_points = []

    for i in inflection_points:
        # 如果点i是极小值（左侧导数为负，右侧为正）
        if derivative[i - 1] < 0 and derivative[i] > 0:
            minima_points.append(i)
        # 如果点i是极大值（左侧导数为正，右侧为负）
        elif derivative[i - 1] > 0 and derivative[i] < 0:
            maxima_points.append(i)

    # 4. 应用阈值过滤极大值点
    # 使用平均极大值作为阈值基准
    if len(maxima_points) > 0:
        peak_values = [processed_resistance[i] for i in maxima_points]
        threshold = np.mean(peak_values) * config["peak_threshold_ratio"]  # 配置参数

        # 过滤低于阈值的极大值点
        filtered_maxima = [i for i in maxima_points if processed_resistance[i] >= threshold]

        # 添加初始结束点和数据结束点
        split_points = [initial_end] + sorted(filtered_maxima) + [len(processed_resistance) - 1]

        # 确保分割点之间有足够距离
        final_split_points = [split_points[0]]
        for i in range(1, len(split_points)):
            if split_points[i] - final_split_points[-1] >= config["min_segment_length"]:
                final_split_points.append(split_points[i])

        # 调试输出
        print(f"检测到 {len(maxima_points)} 个极大值点，过滤后保留 {len(filtered_maxima)} 个分割点")
        return final_split_points
    else:
        # 如果没有找到极大值点，使用简单分割
        print("警告：未找到符合条件的极大值点，使用简单分割")
        return [initial_end, len(processed_resistance) - 1]


def old_split_method(resistance, config, initial_end):
    """旧的分割方法（保持不变，用于对比）"""
    # 2. 寻找按纸过程的连续性中断点（按纸结束，书写开始）
    press_end = initial_end
    max_descent = resistance[initial_end]

    # 计算动态阈值（如果配置启用）
    if config["use_dynamic_thresholds"]:
        inter_threshold, desc_threshold = calculate_dynamic_thresholds(resistance, config)
    else:
        inter_threshold = config["interruption_threshold"]
        desc_threshold = config["descent_threshold"]

    for i in range(initial_end + 1, len(resistance) - 1):
        current_value = resistance[i]
        prev_value = resistance[i - 1]
        next_value = resistance[i + 1]

        # 更新最大下降值
        if current_value < max_descent:
            max_descent = current_value

        # 连续性中断条件
        if (current_value < prev_value and next_value > current_value and
                (next_value - current_value) > inter_threshold * abs(np.mean(resistance[:10]) - max_descent)):
            press_end = i
            break

    # 3. 分割后续书写周期
    writing_segment = resistance[press_end:]
    split_points = [0]  # 书写段的起始点

    in_descent = False
    peak_value = writing_segment[0]

    for i in range(1, len(writing_segment) - 1):
        current_value = writing_segment[i]
        prev_value = writing_segment[i - 1]
        next_value = writing_segment[i + 1]

        # 更新峰值
        if current_value > peak_value:
            peak_value = current_value

        # 检测书写周期的开始
        if (not in_descent and current_value > prev_value and next_value < current_value and
                abs(next_value - current_value) > desc_threshold * peak_value):
            split_points.append(i)
            in_descent = True
            peak_value = current_value

        # 检测书写周期的结束
        if in_descent and current_value < prev_value and next_value > current_value:
            in_descent = False

    split_points.append(len(writing_segment))  # 书写段的结束点

    # 转换为原始数据的索引
    return [initial_end, press_end] + [press_end + p for p in split_points]


def visualize_splits(resistance, splits, file_name, config):
    """可视化分割结果"""
    plt.figure(figsize=(12, 6))

    # 绘制原始数据或归一化数据
    plot_data = normalize(resistance) if config["normalize_data"] else resistance
    plt.plot(plot_data, 'b-', alpha=0.7, label='Resistance')

    # 绘制导数（用于调试）
    if config["use_derivative_method"]:
        derivative = np.diff(plot_data)
        derivative = np.concatenate([[0], derivative])  # 补全长度
        plt.plot(derivative * 0.2 + 0.5, 'g-', alpha=0.5, label='Derivative (scaled)')

    # 标记分割点
    colors = ['r', 'g', 'm', 'c', 'y', 'k']
    start_segment = 0 if config["keep_press_segment"] else 1  # 根据配置决定从哪个段开始标记

    for i, split in enumerate(splits):
        if i < start_segment:
            continue  # 跳过按纸过程的标记

        color = colors[(i - start_segment) % len(colors)]  # 重新计算颜色索引
        plt.axvline(x=split, color=color, linestyle='--', alpha=0.7)

        if i < len(splits) - 1:
            mid_point = (split + splits[i + 1]) // 2
            plt.text(mid_point, max(plot_data) * 0.9,
                     f'Segment {i - start_segment}',  # 段号从0开始重新编号
                     horizontalalignment='center', color=color)

    plt.title(f'Splits for {file_name}')
    plt.xlabel('Time Steps')
    plt.ylabel('Resistance (Normalized)' if config["normalize_data"] else 'Resistance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存可视化结果
    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'{os.path.splitext(file_name)[0]}.png'))
    plt.close()


def process_file(file_path, config):
    """处理单个文件"""
    file_name = os.path.basename(file_path)
    print(f"Processing {file_name}...")

    # 加载数据
    resistance = load_data(file_path)
    if resistance is None:
        return 0, []

    # 数据预处理
    filtered_resistance = preprocess(resistance, config)

    # 分割数据
    split_points = split_by_continuity(filtered_resistance, config)

    # 可视化分割结果
    if config["visualize_samples"]:
        visualize_splits(resistance, split_points, file_name, config)

    # 生成样本
    samples = []
    start_idx = 0 if config["keep_press_segment"] else 1  # 根据配置决定是否跳过按纸过程

    for i in range(start_idx, len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]

        # 跳过过短的段
        if end - start < config["min_writing_length"]:
            continue

        sample = resistance[start:end]

        # 标准化序列长度
        if len(sample) < config["seq_length"]:
            # 填充零
            sample = np.pad(sample, (0, config["seq_length"] - len(sample)), 'constant')
        else:
            # 截断
            sample = sample[:config["seq_length"]]

        samples.append(sample)

    # 保存处理后的数据
    if samples and not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

    output_path = os.path.join(config["output_dir"], f'{os.path.splitext(file_name)[0]}_processed.npy')
    if samples:
        np.save(output_path, np.array(samples))

    return len(samples), split_points


def process_dataset(config):
    """处理整个数据集"""
    # 确保输入目录存在
    if not os.path.exists(config["input_dir"]):
        os.makedirs(config["input_dir"])
        print(f"创建输入目录: {config['input_dir']}")
        print("请将CSV文件放入此目录后重新运行")
        return 0, 0

    total_samples = 0
    processed_files = 0

    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(config["input_dir"])
                 if f.lower().endswith('.csv')]

    print(f"发现 {len(csv_files)} 个CSV文件")

    # 处理每个文件
    for file in tqdm(csv_files, desc="Processing files"):
        file_path = os.path.join(config["input_dir"], file)
        num_samples, splits = process_file(file_path, config)

        if num_samples > 0:
            total_samples += num_samples
            processed_files += 1

    print(f"\n处理完成!")
    print(f"成功处理 {processed_files} 个文件")
    print(f"生成 {total_samples} 个样本")
    print(f"处理后的数据保存在: {config['output_dir']}")
    if config["visualize_samples"]:
        print(f"可视化结果保存在: visualizations/")

    return processed_files, total_samples


if __name__ == "__main__":
    process_dataset(CONFIG)
