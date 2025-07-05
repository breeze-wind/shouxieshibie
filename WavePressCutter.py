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
    "visualization_dir": "visualizations/a_bk",  # 可视化保存目录
    "min_writing_length": 28,  # 最小有效书写长度
    "seq_length": 800,  # 标准化序列长度
    "visualize_samples": True,  # 是否可视化处理结果
    "filter_window": 7,  # 滤波窗口大小
    "filter_polyorder": 3,  # 滤波多项式阶数
    "resistance_column": "DeviceInfo_[23082300]",  # 电阻值所在列名
    "normalize_data": True,  # 是否对数据进行归一化
    "fixed_peak_threshold": 0.5,  # 极大值固定阈值（归一化后）
    "min_segment_length": 20,  # 最小分割段长度
    "keep_press_segment": False,  # 是否保留按纸过程
    "press_threshold": 0.8,  # 按纸过程检测阈值（相对于初始电压）
    "press_min_length": 30,  # 最小按纸长度
    "writing_descent_threshold": 0.1,  # 书写段必须下降的最小阈值（归一化后）
    "debug_segment": True,  # 是否输出段详细信息
}


def load_data(file_path):
    """加载CSV数据，处理复合表头，从第3行开始加载"""
    try:
        # 检测文件编码
        with open(file_path, 'rb') as f:
            raw_data = f.read(1024)
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']

        encodings = [detected_encoding, 'utf-8', 'gbk', 'gb2312', 'latin-1']

        # 尝试每种编码加载文件
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, header=None)
                df.columns = ['DeviceNo', 'Unnamed: 1', 'DeviceInfo_[23082300]', 'DeviceRange_[--------]', 'Unnamed: 4']
                df = df[2:]
                df = df.reset_index(drop=True)

                print(f"成功使用 {encoding} 编码加载文件，跳过2行表头")

                if CONFIG["resistance_column"] not in df.columns:
                    print(f"错误：文件中不存在列 {CONFIG['resistance_column']}")
                    return None

                resistance = pd.to_numeric(df[CONFIG["resistance_column"]], errors='coerce')
                resistance = resistance.dropna().values
                if len(resistance) == 0:
                    print("错误：处理后电阻值列为空")
                    return None

                print(
                    f"数据统计: min={np.min(resistance):.2f}, max={np.max(resistance):.2f}, mean={np.mean(resistance):.2f}, std={np.std(resistance):.2f}")
                return resistance

            except UnicodeDecodeError:
                continue

        print(f"无法使用任何编码加载文件 {file_path}")
        return None

    except Exception as e:
        print(f"加载文件 {file_path} 失败: {e}")
        return None


def normalize(resistance, global_min=None, global_max=None):
    """将电阻值归一化到[0,1]区间（支持全局或局部归一化）"""
    if len(resistance) == 0:
        return resistance
    if global_min is None or global_max is None:
        min_val = np.min(resistance)
        max_val = np.max(resistance)
    else:
        min_val, max_val = global_min, global_max
    if max_val == min_val:
        return np.zeros_like(resistance)
    return (resistance - min_val) / (max_val - min_val)


def preprocess(resistance, config):
    """数据预处理：应用Savitzky-Golay滤波平滑数据"""
    if len(resistance) < config["filter_window"]:
        return resistance
    return savgol_filter(resistance, config["filter_window"], config["filter_polyorder"])


def split_by_continuity(resistance, config):
    """基于拐点检测分割按纸和书写过程"""
    if len(resistance) < 10:
        return [0, len(resistance)]

    processed_resistance = normalize(resistance) if config["normalize_data"] else resistance
    initial_voltage = np.mean(processed_resistance[:10])

    # 检测按纸开始和结束
    press_start = np.where(processed_resistance < config["press_threshold"] * initial_voltage)[0][0] if len(
        np.where(processed_resistance < config["press_threshold"] * initial_voltage)[0]) > 0 else 10
    press_min_idx = press_start + np.argmin(
        processed_resistance[press_start:press_start + config["press_min_length"] * 2])
    press_end = press_min_idx
    for i in range(press_min_idx + 1,
                   min(press_min_idx + config["press_min_length"] * 3, len(processed_resistance) - 1)):
        if processed_resistance[i] > processed_resistance[i - 1] and processed_resistance[i] > config[
            "press_threshold"] * processed_resistance[press_min_idx]:
            press_end = i
            break

    # 检测书写周期
    writing_segment = processed_resistance[press_end:]
    if len(writing_segment) < 10:
        return [0, press_start, press_end, len(processed_resistance)]

    derivative = np.diff(writing_segment)
    maxima_points = []
    for i in range(1, len(derivative) - 1):
        if derivative[i - 1] > 0 and derivative[i] <= 0 and derivative[i + 1] < 0:
            maxima_points.append(i)

    fixed_threshold = config["fixed_peak_threshold"]
    filtered_maxima = [i for i in maxima_points if writing_segment[i] >= fixed_threshold]

    writing_split_points = []
    for point in filtered_maxima:
        if not writing_split_points or point - writing_split_points[-1] >= config["min_segment_length"]:
            writing_split_points.append(point)

    split_points = [0, press_start, press_end] + [press_end + p for p in writing_split_points] + [
        len(processed_resistance) - 1]
    print(f"检测到 {len(maxima_points)} 个极大值点，过滤后保留 {len(filtered_maxima)} 个")
    print(f"按纸过程: {press_start} → {press_end} (长度: {press_end - press_start})")
    return split_points


def visualize_splits(resistance, split_points, file_name, config, segment_status, start_idx):
    """可视化分割结果（使用预计算的段状态）"""
    plt.figure(figsize=(12, 6))

    # 全局归一化数据
    if config["normalize_data"]:
        global_min = np.min(resistance)
        global_max = np.max(resistance)
        plot_data = normalize(resistance, global_min, global_max)
    else:
        plot_data = resistance
    plt.plot(plot_data, 'b-', alpha=0.7, label='Resistance')

    # 绘制导数
    if len(plot_data) > 10:
        derivative = np.diff(plot_data)
        derivative = np.concatenate([[0], derivative])
        plt.plot(derivative * 0.2 + 0.5, 'g-', alpha=0.5, label='Derivative (scaled)')

    # 标记固定阈值
    plt.axhline(y=config["fixed_peak_threshold"], color='r', linestyle='--', alpha=0.3, label='Peak Threshold')

    # 标记按纸过程
    if len(split_points) >= 3:
        plt.axvspan(split_points[1], split_points[2], color='lightgray', alpha=0.3, label='Press Phase')

    # 标记分割点和段状态
    colors = ['r', 'g', 'm', 'c', 'y', 'k']
    status_markers = ['✓', '✗']
    for i, split in enumerate(split_points):
        if i < 1:
            continue
        color = colors[(i - 1) % len(colors)]
        if i == 1 or i == 2:
            plt.axvline(x=split, color='r', linestyle='--', alpha=0.7)
        else:
            plt.axvline(x=split, color=color, linestyle='--', alpha=0.7)

        if i < len(split_points) - 1:
            mid_point = (split + split_points[i + 1]) // 2
            segment_idx = i - start_idx
            if i == 1:
                plt.text(mid_point, max(plot_data) * 0.85, 'Press Start', horizontalalignment='center', color='r')
            elif i == 2:
                plt.text(mid_point, max(plot_data) * 0.85, 'Press End', horizontalalignment='center', color='r')
            else:
                status = segment_status[segment_idx] if segment_idx < len(segment_status) else False
                plt.text(mid_point, max(plot_data) * 0.9,
                         f'Segment {segment_idx} {status_markers[0] if status else status_markers[1]}',
                         horizontalalignment='center', color=color)

    plt.legend()
    plt.title(f'Splits for {file_name}')
    plt.xlabel('Time Steps')
    plt.ylabel('Resistance (Normalized)' if config["normalize_data"] else 'Resistance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_dir = config["visualization_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'{os.path.splitext(file_name)[0]}.png'))
    plt.close()


def process_file(file_path, config):
    """处理单个文件（优化极值检测和样本生成）"""
    file_name = os.path.basename(file_path)
    print(f"\nProcessing {file_name}...")

    # 加载数据
    resistance = load_data(file_path)
    if resistance is None:
        return 0, [], []

    # 数据预处理
    filtered_resistance = preprocess(resistance, config)

    # 分割数据
    split_points = split_by_continuity(filtered_resistance, config)

    # 生成样本
    samples = []
    start_idx = 1 if config["keep_press_segment"] else 2
    segment_status = []
    segment_details = []

    for i in range(start_idx, len(split_points) - 1):
        start, end = split_points[i], split_points[i + 1]
        raw_segment = resistance[start:end]

        if len(raw_segment) == 0:
            details = f"段 {i - start_idx}: 位置={start}-{end}, 数据为空"
            segment_details.append(details)
            segment_status.append(False)
            continue

        # 预处理段数据
        filtered_segment = preprocess(raw_segment, config)

        # 全局归一化（使用整段数据范围）
        if config["normalize_data"]:
            global_min = np.min(filtered_resistance)
            global_max = np.max(filtered_resistance)
            if global_max == global_min:
                normalized_segment = np.zeros_like(filtered_segment)
            else:
                normalized_segment = (filtered_segment - global_min) / (global_max - global_min)
        else:
            normalized_segment = filtered_segment

        # 检查段长度是否足够
        if len(normalized_segment) < 3:
            details = f"段 {i - start_idx}: 位置={start}-{end}, 长度={end - start}, 数据点过少"
            segment_details.append(details)
            segment_status.append(False)
            continue

        # 优化局部极值检测（增加灵敏度和抗噪性）
        local_maxima, local_minima = [], []
        amplitude_threshold = 0.05  # 幅度阈值，可调整

        for j in range(1, len(normalized_segment) - 1):
            # 检测局部极大值（幅度需超过阈值）
            if (normalized_segment[j] > normalized_segment[j - 1] and
                    normalized_segment[j] > normalized_segment[j + 1] and
                    normalized_segment[j] - min(normalized_segment[j - 1],
                                                normalized_segment[j + 1]) > amplitude_threshold):
                local_maxima.append(normalized_segment[j])

            # 检测局部极小值（幅度需超过阈值）
            if (normalized_segment[j] < normalized_segment[j - 1] and
                    normalized_segment[j] < normalized_segment[j + 1] and
                    max(normalized_segment[j - 1], normalized_segment[j + 1]) - normalized_segment[
                        j] > amplitude_threshold):
                local_minima.append(normalized_segment[j])

        # 计算峰值和谷值（优先使用局部极值，若无则使用整体极值）
        if local_maxima and local_minima:
            # 取前3个最大的极大值和最小的极小值，避免噪声影响
            top_maxima = sorted(local_maxima, reverse=True)[:3]
            top_minima = sorted(local_minima)[:3]
            peak_value = max(top_maxima)
            valley_value = min(top_minima)
        else:
            # 若没有找到局部极值，使用整段的最大最小值
            peak_value = np.max(normalized_segment)
            valley_value = np.min(normalized_segment)

        # 计算下降幅度
        descent = peak_value - valley_value
        length = end - start

        # 判断段是否有效
        length_valid = length >= config["min_writing_length"]
        descent_valid = descent >= config["writing_descent_threshold"]
        is_valid = length_valid and descent_valid

        # 记录段详细信息
        details = f"段 {i - start_idx}: 位置={start}-{end}, 长度={length}"
        details += f", 极值点=(max:{len(local_maxima)}, min:{len(local_minima)})"
        details += f", 峰值={peak_value:.3f}, 谷值={valley_value:.3f}, 下降幅度={descent:.3f}"

        if not is_valid:
            reason = "长度不足" if not length_valid else "下降不足"
            details += f", 忽略：{reason}"
        else:
            details += ", 保存为样本"

        segment_details.append(details)
        segment_status.append(is_valid)

        # 保存有效样本
        if is_valid:
            sample = normalized_segment
            # 标准化序列长度
            if len(sample) < config["seq_length"]:
                sample = np.pad(sample, (0, config["seq_length"] - len(sample)), 'constant')
            else:
                sample = sample[:config["seq_length"]]
            samples.append(sample)

    # 可视化分割结果
    if config["visualize_samples"] and split_points:
        visualize_splits(resistance, split_points, file_name, config, segment_status, start_idx)

    # 保存处理后的数据
    if samples and not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

    output_path = os.path.join(config["output_dir"], f'{os.path.splitext(file_name)[0]}_processed.npy')
    if samples:
        np.save(output_path, np.array(samples))

    # 输出段详细信息
    for detail in segment_details:
        print(detail)

    return len(samples), split_points, segment_status


def process_dataset(config):
    """处理整个数据集"""
    if not os.path.exists(config["input_dir"]):
        os.makedirs(config["input_dir"])
        print(f"创建输入目录: {config['input_dir']}")
        print("请将CSV文件放入此目录后重新运行")
        return 0, 0

    csv_files = [f for f in os.listdir(config["input_dir"]) if f.lower().endswith('.csv')]
    print(f"发现 {len(csv_files)} 个CSV文件")

    total_samples = 0
    processed_files = 0
    for file in tqdm(csv_files, desc="Processing files"):
        file_path = os.path.join(config["input_dir"], file)
        num_samples, _, _ = process_file(file_path, config)
        if num_samples > 0:
            total_samples += num_samples
            processed_files += 1

    print(f"\n处理完成!")
    print(f"成功处理 {processed_files} 个文件")
    print(f"生成 {total_samples} 个样本")
    print(f"处理后的数据保存在: {config['output_dir']}")
    if config["visualize_samples"]:
        print(f"可视化结果保存在: {config['visualization_dir']}")
    return processed_files, total_samples


if __name__ == "__main__":
    process_dataset(CONFIG)
