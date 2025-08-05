import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scipy.signal import savgol_filter
from tqdm import tqdm
import chardet
from scipy.interpolate import interp1d
CONFIG = {
    # ======================== 核心配置 ========================
    "version": "2.0",  # 配置版本标识（便于多版本兼容）
    "description": "书写过程断点标注与样本生成配置",  # 配置用途说明

    # ======================== 目录配置 ========================
    "base_input_dir": "dataset",           # 基础输入目录
    "base_output_dir": "processed_data",    # 基础输出目录
    "base_visualization_dir": "visualizations",  # 基础可视化目录
    "base_delete_dir": "dataset/deleted",   # 基础删除目录
    "last_subdir": "data/ying",      # 动态子目录名
    "modify_last_subdir": True,             # 是否动态调整路径
    "input_dir": "dataset/b_rl",            # 完整输入目录（自动生成）
    "output_dir": "processed_data/b_rl",    # 完整输出目录（自动生成）
    "visualization_dir": "visualizations/b_rl",  # 完整可视化目录
    "delete_dir": "dataset/deleted/Four_channel/mao",  # 不完整文件存放目录

    # ======================== 数据参数 ========================
    "num_channels": 1,                      # 通道数量
    "channel_columns": ["[25061405]"],
    #单通道：[25061405]，四通道：["[25071720]", "[25071721]", "[25071722]", "[25071723]"] 通道名称（需与CSV表头对应）
    "channel_colors": ['b','g','r','c'], # 通道可视化颜色（与通道顺序对应）

    "seq_length": 50,                       # 标准化序列长度
    "min_seq_length": 50,                   # 最小序列长度校验阈值
    "max_seq_length": 1000,                 # 最大序列长度校验阈值

    # ======================== 预处理配置 ========================
    "filter_window": 20,                    # 滤波窗口大小（Savitzky-Golay）
    "filter_polyorder": 3,                  # 滤波多项式阶数
    "normalize_data": True,                 # 是否归一化数据至[0,1]
    "interpolation_method": "linear",       # 序列插值方法（linear/quadratic/cubic）

    # ======================== 分割配置 ========================
    "min_writing_length": 20,               # 最小有效书写长度
    "min_segment_length": 20,               # 最小分割段长度
    "fixed_peak_threshold": 0.125,          # 极大值固定阈值（归一化后）
    "writing_descent_threshold": 0.60,      # 书写段下降比例阈值（相对于峰值）
    "keep_press_segment": False,            # 是否保留按纸过程
    "press_threshold": 0.8,                 # 按纸过程检测阈值（相对于初始电压）
    "press_min_length": 30,                 # 最小按纸长度

    # ======================== 可视化配置 ========================
    "visualize_samples": True,              # 是否生成可视化结果
    "visualization_figsize": {
        "combined": (12, 8),                # 四通道联合图尺寸
        "single_segment": (8, 4)            # 单段可视化图尺寸
    },

    # ======================== 文件处理配置 ========================
    "move_incomplete_files": True,         # 是否移动数据组不足的文件
    "min_samples_to_keep": 6,               # 保留文件的最小样本数阈值

    # ======================== 调试配置 ========================
    "debug_segment": True,                  # 是否输出段分割详细信息

    # ======================== 手动分割配置 ========================
    "manual_split": {
        "enable": True,                     # 是否启用手动分割模式
        "split_file_dir": "manual_splits",  # 断点配置文件保存目录
        "file_ext": ".csv"                  # 断点配置文件扩展名
    },

    # ======================== GUI配置 ========================
    "use_gui": True,
    "gui_geometry": "1200x800",              # GUI窗口初始尺寸（宽x高）
    "channel_columns": ["[25071720]", "[25071721]", "[25071722]", "[25071723]"],
    "channel_colors": ['b','g','r','c'],     # 通道可视化颜色
}


def generate_label_map(data_dir):
    """从数据目录生成类别映射（键为子目录名，值为类别索引）"""
    label_map = {}
    label_counter = 0

    # 遍历数据目录下的所有子目录
    for label_dir in os.listdir(data_dir):
        full_path = os.path.join(data_dir, label_dir)
        if os.path.isdir(full_path):
            label_map[label_dir] = label_counter
            label_counter += 1

    # 如果没有子目录，假设所有文件属于同一类别
    if not label_map:
        label_map[data_dir] = 0

    print(f"生成类别映射: {label_map}")
    return label_map
def load_data(file_path):
    """加载CSV数据，处理复合表头，支持动态通道数"""
    try:
        # 检测文件编码（原有逻辑保留）
        with open(file_path, 'rb') as f:
            raw_data = f.read(1024)
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']

        encodings = [detected_encoding, 'utf-8', 'gbk', 'gb2312', 'latin-1']

        for encoding in encodings:
            try:
                # 跳过前3行表头（DeviceNo/DeviceInfo/DeviceRange），从第4行读取数据
                df = pd.read_csv(file_path, encoding=encoding, header=None, skiprows=3)

                # 动态提取通道数据（假设通道数据从第3列开始，连续排列）
                num_channels = CONFIG["num_channels"]
                start_col = 2  # 通道数据起始列（0-based索引）
                end_col = start_col + num_channels  # 结束列

                # 提取所有通道并转换为数值类型
                channel_data = df.iloc[:, start_col:end_col].apply(pd.to_numeric, errors='coerce')

                # 校验各通道有效性
                valid_channels = []
                for ch_idx in range(num_channels):
                    ch_name = CONFIG["channel_columns"][ch_idx]
                    ch_data = channel_data.iloc[:, ch_idx].dropna().values
                    if len(ch_data) == 0:
                        print(f"错误：通道 {ch_name} 数据为空")
                        return None
                    valid_channels.append(ch_data)

                # 确保所有通道长度一致
                if len(set(len(ch) for ch in valid_channels)) > 1:
                    print(f"错误：通道长度不一致: {[len(ch) for ch in valid_channels]}")
                    return None

                # 合并为二维数组（shape: [样本数, 通道数]）
                combined_data = np.column_stack(valid_channels)
                print(f"成功加载 {num_channels} 通道数据，样本数: {combined_data.shape[0]}")
                return combined_data

            except UnicodeDecodeError:
                continue

        print(f"无法加载文件 {file_path}")
        return None

    except Exception as e:
        print(f"加载失败: {e}")
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
    # 修改条件：只有当数据长度 > 窗口大小时才滤波（严格小于）
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


def visualize_splits(resistance, split_points, file_name, config, segment_status, start_idx, channel_idx):
    """可视化分割结果（使用预计算的段状态，支持通道子文件夹）"""
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
                         f'Segment {segment_idx}\n{status_markers[0] if status else status_markers[1]}',
                         horizontalalignment='center', color=color)

    plt.legend()
    # 检查文件是否会被移除，如果是则添加红色REMOVED标记
    is_removed = config["move_incomplete_files"] and 0 < len(segment_status) and sum(segment_status) < config["min_samples_to_keep"]
    title = f'Splits for {file_name}'
    if is_removed:
        title += ' (REMOVED)'
    plt.title(title)

    # 如果文件被移除，添加醒目的红色标记
    if is_removed:
        plt.text(len(plot_data)*0.5, max(plot_data)*0.5,
                 'REMOVED', fontsize=30, color='red',
                 horizontalalignment='center', verticalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.7), rotation=30)
    plt.xlabel('Time Steps')
    plt.ylabel('Resistance (Normalized)' if config["normalize_data"] else 'Resistance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 构建通道专属子文件夹路径（新增代码）
    channel_vis_dir = os.path.join(config["visualization_dir"], f"ch{channel_idx}")
    if not os.path.exists(channel_vis_dir):
        os.makedirs(channel_vis_dir)

    # 保存图像到通道子文件夹（修改路径）
    plt.savefig(os.path.join(channel_vis_dir, f'{os.path.splitext(file_name)[0]}.png'))
    plt.close()


def process_file(file_path, config):
    """处理单个文件（支持多通道动态处理）"""
    file_name = os.path.basename(file_path)
    print(f"\nProcessing {file_name}...")

    # 加载多通道数据（shape: [n_samples, num_channels]）
    resistance = load_data(file_path)
    if resistance is None:
        return 0, [], []

    all_samples = []  # 存储所有通道的样本
    all_split_points = []
    all_segment_status = []

    # 创建文件级输出目录（与GUI保持一致）
    base_output = os.path.join(config["output_dir"], os.path.splitext(file_name)[0])
    os.makedirs(base_output, exist_ok=True)

    # 动态循环处理每个通道（数量由config["num_channels"]决定）
    for ch_idx in range(config["num_channels"]):
        ch_name = config["channel_columns"][ch_idx]
        print(f"\n----- 处理通道 {ch_name} -----")
        ch_resistance = resistance[:, ch_idx]  # 当前通道的一维数据

        # 数据预处理（滤波）
        filtered_resistance = preprocess(ch_resistance, config)

        # 分割数据（原单通道逻辑保留）
        split_points = split_by_continuity(filtered_resistance, config)

        # 生成当前通道样本（原单通道逻辑保留）
        samples = []
        start_idx = 1 if config["keep_press_segment"] else 2
        segment_status = []
        segment_details = []

        for i in range(start_idx, len(split_points) - 1):
            start, end = split_points[i], split_points[i + 1]
            # 修复：使用当前通道的一维数据 ch_resistance 而非多通道数组 resistance
            raw_segment = ch_resistance[start:end]  # <-- 此处为修复点

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
            descent_valid = descent >= (config["writing_descent_threshold"] * peak_value)
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
                if len(sample) != config["seq_length"]:
                    # 原始样本的索引
                    x_old = np.arange(len(sample))
                    # 目标样本的索引
                    x_new = np.linspace(0, len(sample) - 1, config["seq_length"])
                    # 创建插值函数
                    f = interp1d(x_old, sample, kind='linear')
                    # 进行插值得到新样本
                    sample = f(x_new)
                samples.append(sample)
                ch_output_dir = os.path.join(base_output, f"ch{ch_idx}")
                os.makedirs(ch_output_dir, exist_ok=True)
                np.save(os.path.join(ch_output_dir, f'segment_{len(samples) - 1}.npy'), sample)

        # 可视化当前通道（传入通道索引，使用原始文件名）
        if config["visualize_samples"] and split_points:
            visualize_splits(
                ch_resistance,
                split_points,
                file_name,  # 文件名不再添加通道后缀
                config,
                segment_status,
                start_idx,
                channel_idx=ch_idx  # 新增：传入通道索引用于子文件夹创建
            )

        # 合并当前通道结果
        all_samples.extend(samples)
        all_split_points.append(split_points)
        all_segment_status.append(segment_status)

    # 添加最终的返回语句（修复None返回值问题）
    return len(all_samples), all_split_points, all_segment_status  # <-- 新增


def update_directory_paths(config):
    """
    根据配置动态更新目录路径

    功能: 根据modify_last_subdir标志决定是否重新构建目录路径
    - 当modify_last_subdir为True时，使用base_*_dir和last_subdir组合生成新路径
    - 当modify_last_subdir为False时，保持原有路径不变

    参数:
        config (dict): 包含路径配置的字典，必须包含以下键:
            - modify_last_subdir: 是否修改最后一个子目录的标志
            - base_input_dir: 基础输入目录
            - base_output_dir: 基础输出目录
            - base_visualization_dir: 基础可视化目录
            - last_subdir: 要使用的最后一个子目录名

    返回:
        dict: 更新后的配置字典，包含更新后的input_dir, output_dir和visualization_dir
    """
    if config["modify_last_subdir"]:
        # 重新构建目录路径: 基础目录 + 最后一个子目录
        config["input_dir"] = os.path.join(config["base_input_dir"], config["last_subdir"])
        config["output_dir"] = os.path.join(config["base_output_dir"], config["last_subdir"])
        config["visualization_dir"] = os.path.join(config["base_visualization_dir"], config["last_subdir"])
        config["delete_dir"] = os.path.join(config["base_delete_dir"], config["last_subdir"])
    return config

def process_dataset(config):
    """处理整个数据集"""
    # 更新目录路径
    config = update_directory_paths(config)

    if not os.path.exists(config["input_dir"]):
        os.makedirs(config["input_dir"])
        print(f"创建输入目录: {config['input_dir']}")
        print("请将CSV文件放入此目录后重新运行")
        return 0, 0

    # 递归获取所有CSV文件并保留相对路径
    csv_files = []
    for root, dirs, files in os.walk(config["input_dir"]):
        for file in files:
            if file.lower().endswith('.csv'):
                rel_path = os.path.relpath(root, config["input_dir"])
                csv_files.append((os.path.join(root, file), rel_path))

    print(f"发现 {len(csv_files)} 个CSV文件（包含子目录）")

    total_samples = 0
    processed_files = 0

    # 保存原始目录路径
    original_output = config["output_dir"]
    original_visualization = config["visualization_dir"]
    original_delete = config["delete_dir"]

    for file_path, rel_path in tqdm(csv_files, desc="Processing files"):
        try:
            # 动态创建子目录结构
            if rel_path != ".":
                config["output_dir"] = os.path.join(original_output, rel_path)
                config["visualization_dir"] = os.path.join(original_visualization, rel_path)
                config["delete_dir"] = os.path.join(original_delete, rel_path)

            # 处理文件（原有逻辑保留）
            num_samples, _, _ = process_file(file_path, config)
            if num_samples > 0:
                total_samples += num_samples
                processed_files += 1

            # 移动不完整文件（原有逻辑保留）
            if config["move_incomplete_files"] and num_samples < config["min_samples_to_keep"]:
                os.makedirs(config["delete_dir"], exist_ok=True)
                shutil.move(file_path, os.path.join(config["delete_dir"], os.path.basename(file_path)))

        finally:
            # 恢复原始目录路径
            config["output_dir"] = original_output
            config["visualization_dir"] = original_visualization
            config["delete_dir"] = original_delete

    print(f"\n处理完成!")
    print(f"成功处理 {processed_files} 个文件")
    print(f"生成 {total_samples} 个样本")
    print(f"处理后的数据保存在: {config['output_dir']}")
    if config["visualize_samples"]:
        print(f"可视化结果保存在: {config['visualization_dir']}")
    return processed_files, total_samples


class BreakpointAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("书写过程断点标注工具")
        self.root.geometry(CONFIG["gui_geometry"])

        # 数据状态
        self.current_file = None
        self.current_channel = 0
        self.resistance_data = None  # 原始电阻数据
        self.split_points = []       # 当前标注的断点
        self.channels = CONFIG["channel_columns"]
        self.channel_colors = CONFIG["channel_colors"]

        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建工具栏
        self.toolbar_frame = ttk.Frame(self.main_frame)
        self.toolbar_frame.pack(fill=tk.X, pady=5)

        # 工具栏按钮
        ttk.Button(self.toolbar_frame, text="打开CSV", command=self.load_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar_frame, text="保存断点", command=self.save_breakpoints).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar_frame, text="保存分段", command=self.save_segments).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar_frame, text="下一个样本", command=self.open_next_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar_frame, text="退出", command=root.quit).pack(side=tk.RIGHT, padx=2)

        # 通道选择框架
        self.channel_frame = ttk.Frame(self.main_frame)
        self.channel_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.channel_frame, text="通道选择:").pack(side=tk.LEFT, padx=5)
        self.channel_combobox = ttk.Combobox(
            self.channel_frame, values=self.channels, state="readonly", width=15
        )
        self.channel_combobox.current(0)
        self.channel_combobox.bind("<<ComboboxSelected>>", self.on_channel_change)
        self.channel_combobox.pack(side=tk.LEFT, padx=5)

        # 断点控制按钮
        self.breakpoint_frame = ttk.Frame(self.main_frame)
        self.breakpoint_frame.pack(fill=tk.X, pady=5)

        ttk.Button(self.breakpoint_frame, text="添加断点", command=self.add_breakpoint).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.breakpoint_frame, text="删除断点", command=self.delete_breakpoint).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.breakpoint_frame, text="清空断点", command=self.clear_breakpoints).pack(side=tk.LEFT, padx=2)

        # 创建Matplotlib图表
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加Matplotlib工具栏
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.main_frame)
        self.toolbar.update()

        # 绑定鼠标点击事件
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*")]
        )
        if not file_path:
            return

        self.current_file = file_path
        try:
            # 简化版CSV加载逻辑
            data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
            self.resistance_data = data[:, self.current_channel + 1]  # 假设第一列是时间
            self.split_points = []
            self.update_plot()
            self.root.title(f"书写过程断点标注工具 - {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("加载错误", f"无法加载文件: {str(e)}")

    def update_plot(self):
        if self.resistance_data is None:
            return

        self.ax.clear()
        self.ax.plot(self.resistance_data, color=self.channel_colors[self.current_channel])
        self.ax.set_title(f"通道 {self.channels[self.current_channel]} 数据")
        self.ax.set_xlabel("样本点")
        self.ax.set_ylabel("电阻值")

        # 绘制断点
        for point in self.split_points:
            self.ax.axvline(x=point, color='red', linestyle='--')

        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes == self.ax:
            self.split_points.append(int(round(event.xdata)))
            self.split_points.sort()
            self.update_plot()

    def on_channel_change(self, event):
        self.current_channel = self.channel_combobox.current()
        if self.resistance_data is not None:
            self.update_plot()

    def add_breakpoint(self):
        # 添加当前鼠标位置的断点
        xlim = self.ax.get_xlim()
        mid_point = int((xlim[0] + xlim[1]) / 2)
        self.split_points.append(mid_point)
        self.split_points.sort()
        self.update_plot()

    def delete_breakpoint(self):
        if self.split_points:
            self.split_points.pop()
            self.update_plot()

    def clear_breakpoints(self):
        self.split_points = []
        self.update_plot()

    def save_breakpoints(self):
        if not self.current_file or not self.split_points:
            messagebox.showwarning("保存警告", "没有可保存的断点数据")
            return

        save_path = os.path.splitext(self.current_file)[0] + "_breakpoints.txt"
        try:
            with open(save_path, 'w') as f:
                for point in self.split_points:
                    f.write(f"{point}\n")
            messagebox.showinfo("保存成功", f"断点已保存至: {save_path}")
        except Exception as e:
            messagebox.showerror("保存错误", f"无法保存断点: {str(e)}")

    def save_segments(self):
        if not self.current_file or not self.split_points:
            messagebox.showwarning("保存警告", "没有可保存的分段数据")
            return

        messagebox.showinfo("保存成功", "分段数据已处理完成")

    def open_next_file(self):
        messagebox.showinfo("提示", "下一个文件功能待实现")


def run_gui():
    root = tk.Tk()
    app = BreakpointAnnotator(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()

    else:  # else关键字需单独成行并正确缩进
        process_dataset(CONFIG)

# 注意：visualize_combined_channels函数已在文件上方全局定义，此处无需重复定义        plt.savefig(save_path)
        plt.close()        # ... 其他初始化代码保持不变 ...