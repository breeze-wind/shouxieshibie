import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from scipy.signal import savgol_filter
from tqdm import tqdm
import chardet
from scipy.interpolate import interp1d
import json
CONFIG = {
    # ======================== 核心配置 ========================
    "version": "2.0",  # 配置版本标识（便于多版本兼容）
    "description": "书写过程断点标注与样本生成配置",  # 配置用途说明

    # ======================== 目录配置 ========================
    "base_input_dir": "dataset",  # 基础输入目录
    "base_output_dir": "processed_data",  # 基础输出目录
    "base_visualization_dir": "visualizations",  # 基础可视化目录
    "base_delete_dir": "dataset/deleted",  # 基础删除目录
    "last_subdir": "data/ying/A",  # 动态子目录名
    "modify_last_subdir": True,  # 是否动态调整路径
    "input_dir": "dataset/b_rl",  # 完整输入目录（自动生成）
    "output_dir": "processed_sample/data/ying/A",  # 完整输出目录（自动生成）
    "visualization_dir": "visualizations/b_rl",  # 完整可视化目录
    "delete_dir": "dataset/deleted/Four_channel/mao",  # 不完整文件存放目录

    # ======================== 数据参数 ========================
    "num_channels": 1,  # 通道数量
    "channel_columns": [25061405],
    # 单通道：[25061405]，四通道：["[25071720]", "[25071721]", "[25071722]", "[25071723]"] 通道名称（需与CSV表头对应）
    "channel_colors": ['b', 'g', 'r', 'c'],  # 通道可视化颜色（与通道顺序对应）

    "seq_length": 50,  # 标准化序列长度
    "min_seq_length": 50,  # 最小序列长度校验阈值
    "max_seq_length": 1000,  # 最大序列长度校验阈值

    # ======================== 预处理配置 ========================
    "filter_window": 20,  # 滤波窗口大小（Savitzky-Golay）
    "filter_polyorder": 3,  # 滤波多项式阶数
    "normalize_data": True,  # 是否归一化数据至[0,1]
    "interpolation_method": "linear",  # 序列插值方法（linear/quadratic/cubic）

    # ======================== 分割配置 ========================
    "min_writing_length": 20,  # 最小有效书写长度
    "min_segment_length": 20,  # 最小分割段长度
    "fixed_peak_threshold": 0.125,  # 极大值固定阈值（归一化后）
    "writing_descent_threshold": 0.60,  # 书写段下降比例阈值（相对于峰值）
    "keep_press_segment": False,  # 是否保留按纸过程
    "press_threshold": 0.8,  # 按纸过程检测阈值（相对于初始电压）
    "press_min_length": 30,  # 最小按纸长度

    # ======================== 可视化配置 ========================
    "visualize_samples": True,  # 是否生成可视化结果
    "visualization_figsize": {
        "combined": (12, 8),  # 四通道联合图尺寸
        "single_segment": (8, 4)  # 单段可视化图尺寸
    },

    # ======================== 文件处理配置 ========================
    "move_incomplete_files": True,  # 是否移动数据组不足的文件
    "min_samples_to_keep": 6,  # 保留文件的最小样本数阈值

    # ======================== 调试配置 ========================
    "debug_segment": True,  # 是否输出段分割详细信息

    # ======================== 手动分割配置 ========================
    "manual_split": {
        "enable": True,  # 是否启用手动分割模式
        "split_file_dir": "manual_splits",  # 断点配置文件保存目录
        "file_ext": ".csv"  # 断点配置文件扩展名
    },

    # ======================== GUI配置 ========================
    "use_gui": True,  # 是否默认使用GUI模式（未提供--gui参数时）
    "gui_geometry": "1200x800"  # GUI窗口初始尺寸（宽x高）
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
    is_removed = config["move_incomplete_files"] and 0 < len(segment_status) and sum(segment_status) < config[
        "min_samples_to_keep"]
    title = f'Splits for {file_name}'
    if is_removed:
        title += ' (REMOVED)'
    plt.title(title)

    # 如果文件被移除，添加醒目的红色标记
    if is_removed:
        plt.text(len(plot_data) * 0.5, max(plot_data) * 0.5,
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
        self.root.geometry("1200x800")

        # 新增：样本审核状态跟踪（键为段索引，值为是否合格）
        self.segment_status = {}  # {segment_index: True/False}
        self.current_hover_segment = -1  # 当前鼠标悬停的段索引

        # 数据状态
        self.current_file = None
        self.current_channel = 0
        self.resistance_data = None  # 原始电阻数据
        self.split_points = []  # 当前标注的断点
        self.channels = CONFIG["channel_columns"]
        # 使用CONFIG通道颜色（原硬编码['b','g','r','c']迁移至此）
        self.channel_colors = CONFIG["channel_colors"]

        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建菜单栏（修改为按钮布局）
        self.toolbar_frame = ttk.Frame(self.main_frame)
        self.toolbar_frame.pack(fill=tk.X, pady=5)

        # 创建展开的按钮组
        ttk.Button(self.toolbar_frame, text="打开CSV", command=self.load_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar_frame, text="保存断点", command=self.save_breakpoints).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar_frame, text="保存分段", command=self.save_segments).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar_frame, text="上一个样本", command=self.open_previous_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar_frame, text="下一个样本", command=self.open_next_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar_frame, text="退出", command=root.quit).pack(side=tk.RIGHT, padx=2)
        ttk.Button(self.toolbar_frame, text="自动分割", command=self.auto_split).pack(side=tk.LEFT, padx=2)  # 新增按钮
        ttk.Button(self.toolbar_frame, text="放大视图", command=self.zoom_to_200_steps).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar_frame, text="还原全图", command=self.reset_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar_frame, text="导出配置到文件", command=self.export_config_to_file).pack(side=tk.RIGHT,
                                                                                                       padx=2)
        ttk.Button(self.toolbar_frame, text="保存配置到内存", command=self.save_config_to_memory).pack(side=tk.RIGHT,
                                                                                                       padx=2)
        ttk.Button(self.toolbar_frame, text="配置管理", command=self.edit_config).pack(side=tk.RIGHT, padx=2)
        # 通道选择框架
        self.channel_frame = ttk.Frame(self.main_frame)
        self.channel_frame.pack(fill=tk.X, pady=5)
        self.view_start = 0
        self.view_window_size = 200  # 固定窗口大小为200步
        self.is_zoomed = False  # 是否处于缩放模式

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
        # 绑定滚轮事件（新增）

        ttk.Button(
            self.breakpoint_frame, text="添加断点", command=self.add_breakpoint
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            self.breakpoint_frame, text="删除选中断点", command=self.delete_breakpoint
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            self.breakpoint_frame, text="清空断点", command=self.clear_breakpoints
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.toolbar_frame, text="设置保存路径", command=self.set_save_directory).pack(side=tk.RIGHT, padx=2)
        ttk.Button(
            self.breakpoint_frame, text="一键通过", command=self.approve_all_segments
        ).pack(side=tk.LEFT, padx=5)
        # Matplotlib图表区域
        # 在文件顶部导入matplotlib后添加字体配置
        import matplotlib.pyplot as plt
        # 配置中文字体支持
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

        self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=100)
        # 设置图表标题字体（确保中文显示）
        self.ax.set_title("四通道联合标注（点击添加全局断点）", fontproperties="SimHei")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.main_frame)
        self.toolbar.update()
        #↓绑定中键
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.toolbar.update()
        # 绑定鼠标点击事件
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)  # 新增：鼠标移动事件

        self.custom_output_dir = None
        self.custom_split_dir = None
        # 初始化状态文本
        self.status_text = tk.StringVar()
        self.status_text.set("就绪: 请打开CSV文件开始标注 | 鼠标中键点击段区域审核样本")  # 更新提示
        ttk.Label(self.main_frame, textvariable=self.status_text).pack(fill=tk.X, pady=5)

    def approve_all_segments(self):
        """将所有分段标记为合格"""
        if not self.split_points:
            messagebox.showwarning("警告", "请先添加断点创建分段")
            return

        # 遍历所有可能的段索引
        total_segments = len(self.split_points) - 1
        for seg_idx in range(total_segments):
            self.segment_status[seg_idx] = True

        self.update_plot()
        self.status_text.set(f"已一键通过 {total_segments} 个分段")
    def open_previous_file(self):
        """打开当前文件所在目录的上一个CSV文件"""
        if not self.current_file:
            messagebox.showwarning("警告", "请先打开一个文件")
            return

        current_dir = os.path.dirname(self.current_file)
        csv_files = sorted([f for f in os.listdir(current_dir) if f.lower().endswith('.csv')])

        if not csv_files:
            messagebox.showinfo("信息", "当前目录没有CSV文件")
            return

        try:
            current_index = csv_files.index(os.path.basename(self.current_file))
            # 修改索引计算方式实现循环切换
            previous_index = (current_index - 1) % len(csv_files)
            previous_file = os.path.join(current_dir, csv_files[previous_index])

            # 加载并更新显示（与下一个样本保持相同逻辑）
            self.current_file = previous_file
            self.resistance_data = load_data(previous_file)
            self.split_points = []
            self.segment_status = {}  # 重置审核状态
            self.update_plot()
            base_name = os.path.basename(previous_file)
            self.root.title(f"书写过程断点标注工具 - {base_name}")
            self.status_text.set(f"已加载上一个样本: {base_name}")
        except ValueError:
            messagebox.showerror("错误", "当前文件不在目录列表中")
        plt.close()
    def set_save_directory(self):
        """设置自定义保存路径"""
        dir_path = filedialog.askdirectory(title="选择保存目录")
        if dir_path:
            self.custom_output_dir = dir_path
            self.custom_split_dir = os.path.join(dir_path, "manual_splits")
            self.status_text.set(f"保存路径已设置为: {dir_path}")

    def on_scroll(self, event):
        """处理鼠标滚轮滚动事件"""
        if not self.is_zoomed or self.resistance_data is None:
            return

        # 向上滚动：向前200步，向下滚动：向后200步
        if event.button == 'up':
            self.scroll_view(forward=False)
        elif event.button == 'down':
            self.scroll_view(forward=True)

    def scroll_view(self, forward=True):
        """控制视图滚动（新增方法）"""
        data_length = len(self.resistance_data)
        step = 50

        if forward:
            new_start = self.view_start + step
            # 检查是否超出数据范围
            if new_start + step > data_length:
                new_start = 0
        else:
            new_start = self.view_start - step
            # 处理负数情况
            if new_start < 0:
                new_start = data_length - (data_length % step or step)

        self.view_start = new_start
        self.update_plot()
        self.status_text.set(f"当前视图范围: {self.view_start}-{self.view_start + self.view_window_size}")

    def zoom_to_200_steps(self):
        """切换到200步窗口视图"""
        if self.resistance_data is None:
            return

        data_length = len(self.resistance_data)
        if data_length <= self.view_window_size:
            messagebox.showinfo("提示", "数据长度不足200步，已显示全部数据")
            self.is_zoomed = False
            self.view_start = 0
            return

        self.is_zoomed = True
        self.scroll_view(forward=True)  # 改为调用统一滚动方法


    def reset_view(self):
        """恢复完整视图"""
        self.is_zoomed = False
        self.update_plot()
    def auto_split(self):
        """调用原有算法进行自动分割"""
        if self.resistance_data is None:
            messagebox.showwarning("警告", "请先加载CSV文件")
            return

        try:
            # 获取当前通道数据
            ch_idx = self.current_channel
            ch_data = self.resistance_data[:, ch_idx]

            # 预处理数据
            filtered_data = preprocess(ch_data, CONFIG)

            # 调用原有分割算法
            split_points = split_by_continuity(filtered_data, CONFIG)

            # 合并并更新分割点（保留原有手动添加的点）
            self.split_points = sorted(list(set(split_points + self.split_points)))
            self.segment_status = {}  # 清空审核状态
            self.update_plot()
            self.status_text.set(f"自动分割完成，生成 {len(split_points)} 个断点")

        except Exception as e:
            messagebox.showerror("错误", f"自动分割失败: {str(e)}")

    def open_next_file(self):
        """打开当前文件所在目录的下一个CSV文件"""
        if not self.current_file:
            messagebox.showwarning("警告", "请先打开一个文件")
            return

        current_dir = os.path.dirname(self.current_file)
        csv_files = sorted([f for f in os.listdir(current_dir) if f.lower().endswith('.csv')])

        if not csv_files:
            messagebox.showinfo("信息", "当前目录没有CSV文件")
            return

        try:
            current_index = csv_files.index(os.path.basename(self.current_file))
            next_index = (current_index + 1) % len(csv_files)
            next_file = os.path.join(current_dir, csv_files[next_index])

            # 加载并更新显示
            self.current_file = next_file
            self.resistance_data = load_data(next_file)
            self.split_points = []
            self.segment_status = {}  # 重置审核状态
            self.update_plot()
            base_name = os.path.basename(next_file)
            self.root.title(f"书写过程断点标注工具 - {base_name}")
            self.status_text.set(f"已加载下一个样本: {base_name}")

        except ValueError:
            messagebox.showerror("错误", "当前文件不在目录列表中")
        plt.close()

    def load_file(self):
        """加载CSV文件并显示当前通道数据"""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        self.current_file = file_path
        self.resistance_data = load_data(file_path)
        if self.resistance_data is None:
            messagebox.showerror("错误", "无法加载文件或数据无效")
            return

        # 在窗口标题添加文件名显示
        base_name = os.path.basename(self.current_file)
        self.root.title(f"书写过程断点标注工具 - {base_name}")

        self.current_channel = 0
        self.channel_combobox.current(0)
        self.split_points = []
        self.segment_status = {}
        self.update_plot()
        self.status_text.set(f"已加载: {base_name}")
    def on_channel_change(self, event):
        """切换通道时更新图表"""
        if self.resistance_data is None:
            return
        self.current_channel = self.channel_combobox.current()
        self.update_plot()

    def update_plot(self):
        """更新图表，增加样本审核状态显示"""
        self.ax.clear()
        if self.resistance_data is None:
            return
            # 计算显示范围
        data_length = len(self.resistance_data)
        if self.is_zoomed and data_length > self.view_window_size:
                start = self.view_start
                end = min(start + self.view_window_size, data_length)
                xlim = (start, end)
        else:
                xlim = (0, data_length)

        # 绘制所有通道数据
        for ch_idx in range(CONFIG["num_channels"]):
            ch_resistance = self.resistance_data[:, ch_idx]
            filtered_resistance = preprocess(ch_resistance, CONFIG)
            self.ax.plot(
                filtered_resistance,
                color=self.channel_colors[ch_idx],
                alpha=0.7,
                label=f'通道 {CONFIG["channel_columns"][ch_idx]}'
            )
        self.ax.set_xlim(xlim)
        # 绘制分段区域（带审核状态颜色）
        if self.split_points:
            sorted_points = sorted(self.split_points)
            for i in range(len(sorted_points) - 1):
                start = sorted_points[i]
                end = sorted_points[i + 1]
                seg_idx = i  # 段索引从0开始

                # 根据审核状态设置背景色
                if self.segment_status.get(seg_idx, False):
                    # 合格：绿色背景
                    self.ax.axvspan(start, end, color='green', alpha=0.2)
                    status_text = "合格"
                else:
                    # 不合格/未审核：红色背景（未审核时半透明）
                    alpha = 0.1 if seg_idx not in self.segment_status else 0.2
                    self.ax.axvspan(start, end, color='red', alpha=alpha)
                    status_text = "未审核" if seg_idx not in self.segment_status else "不合格"

                # 显示段索引和状态
                mid_x = (start + end) / 2
                mid_y = self.ax.get_ylim()[1] * 0.9  # 顶部90%位置
                self.ax.text(
                    mid_x, mid_y,
                    f"段 {seg_idx}: {status_text}",
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7)
                )

        # 绘制全局断点
        for point in sorted(self.split_points):
            self.ax.axvline(x=point, color='m', linestyle='--', alpha=0.7)
        if hasattr(self, 'auto_split_points'):
            for point in self.auto_split_points:
                self.ax.axvline(x=point, color='orange', linestyle=':', alpha=0.5)
        # 添加图例和提示
        self.ax.legend()
        self.ax.set_title("四通道联合标注（左键加断点 | 中键审核样本 | 右键删断点）")
        self.ax.set_xlabel("时间步")
        self.ax.set_ylabel("电阻值（归一化）" if CONFIG["normalize_data"] else "电阻值")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        if self.is_zoomed:
            self.ax.set_title(f"四通道联合标注 - 200步窗口视图 [{xlim[0]}-{xlim[1]}]")
        else:
            self.ax.set_title("四通道联合标注（点击添加全局断点）")
        self.canvas.draw()

    def on_click(self, event):
        """处理鼠标点击事件，增加中键审核功能"""
        if event.inaxes != self.ax or self.resistance_data is None:
            return

        # 左键添加断点（原有功能）
        if event.button == 1:
            self.split_points.append(round(event.xdata))
            self.split_points = sorted(list(set(self.split_points)))
            self.status_text.set(f"全局断点: {self.split_points} (共 {len(self.split_points)} 个)")

        # 右键删除断点（原有功能）
        elif event.button == 3:
            if not self.split_points:
                return
            nearest = min(self.split_points, key=lambda x: abs(x - event.xdata))
            if abs(nearest - event.xdata) < 10:
                self.split_points.remove(nearest)
                # 清除与该断点相关的段状态
                sorted_points = sorted(self.split_points)
                self.segment_status = {k: v for k, v in self.segment_status.items() if k < len(sorted_points) - 1}
                self.status_text.set(f"已删除断点 {nearest}，剩余 {len(self.split_points)} 个")

        # 新增：中键审核样本（切换合格/不合格状态）
        elif event.button == 2:
            if len(self.split_points) < 2:
                self.status_text.set("请先添加至少两个断点创建分段")
                return

            # 找到点击位置所在的段
            click_pos = round(event.xdata)
            sorted_points = sorted(self.split_points)
            for i in range(len(sorted_points) - 1):
                start = sorted_points[i]
                end = sorted_points[i + 1]
                if start <= click_pos <= end:
                    seg_idx = i
                    # 切换状态（未审核→合格→不合格→合格...）
                    current_status = self.segment_status.get(seg_idx, False)
                    new_status = not current_status if seg_idx in self.segment_status else True
                    self.segment_status[seg_idx] = new_status
                    status_text = "合格" if new_status else "不合格"
                    self.status_text.set(f"段 {seg_idx} 已标记为: {status_text}")
                    break

        self.split_points = sorted(self.split_points)
        self.update_plot()

    def on_mouse_move(self, event):
        """鼠标移动时显示当前所在段信息"""
        if event.inaxes != self.ax or self.resistance_data is None or not self.split_points:
            return

        pos = round(event.xdata)
        sorted_points = sorted(self.split_points)
        for i in range(len(sorted_points) - 1):
            if sorted_points[i] <= pos <= sorted_points[i + 1]:
                if self.current_hover_segment != i:
                    self.current_hover_segment = i
                    status = "合格" if self.segment_status.get(i,
                                                               False) else "不合格" if i in self.segment_status else "未审核"
                    self.status_text.set(f"当前段: {i} (状态: {status}) | 中键点击切换状态")
                return
        self.current_hover_segment = -1

    def add_breakpoint(self):
        """手动输入全局断点时间步"""
        if self.resistance_data is None:
            messagebox.showwarning("警告", "请先加载文件")
            return
        try:
            value = simpledialog.askinteger(
                "添加全局断点", "输入时间步值:",
                minvalue=0, maxvalue=len(self.resistance_data) - 1
            )
            if value is not None:
                self.split_points.append(value)
                self.split_points = sorted(list(set(self.split_points)))
                self.update_plot()
                self.status_text.set(f"全局断点: {self.split_points} (共 {len(self.split_points)} 个)")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的整数")

    def delete_breakpoint(self):
        """删除选中的断点"""
        if not self.split_points:
            return
        try:
            value = simpledialog.askinteger(
                "删除断点", f"当前断点: {self.split_points}\n输入要删除的值:"
            )
            if value in self.split_points:
                self.split_points.remove(value)
                # 清除与该断点相关的段状态
                sorted_points = sorted(self.split_points)
                self.segment_status = {k: v for k, v in self.segment_status.items() if k < len(sorted_points) - 1}
                self.update_plot()
                self.status_text.set(
                    f"断点: {self.split_points} (共 {len(self.split_points)} 个)"
                )
        except ValueError:
            pass

    def edit_config(self):
        """配置编辑窗口"""
        self.config_window = tk.Toplevel(self.root)
        self.config_window.title("算法参数配置")
        self.config_window.geometry("500x600")
        # 滚动区域设置
        container = ttk.Frame(self.config_window)
        container.pack(fill='both', expand=True)
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(self.config_window, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        # 配置滚动区域
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 布局滚动组件
        container.pack(fill="both", expand=True)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # 配置项
        config_fields = [
            ('min_writing_length', '最小有效书写长度（整数）', int),
            ('min_segment_length', '最小分割段长度（整数）', int),
            ('fixed_peak_threshold', '极大值固定阈值（0-1小数）', float),
            ('writing_descent_threshold', '书写下降阈值（0-1小数）', float),
            ('press_threshold', '按纸检测阈值（0-1小数）', float),
            ('press_min_length', '最小按纸长度（整数）', int)
        ]

        self.config_entries = {}
        for row, (key, label, dtype) in enumerate(config_fields):
            ttk.Label(self.scrollable_frame, text=label).grid(row=row, column=0, padx=5, pady=5, sticky="w")
            entry = ttk.Entry(self.scrollable_frame)
            entry.insert(0, str(CONFIG[key]))
            entry.grid(row=row, column=1, padx=5, pady=5)
            self.config_entries[key] = (entry, dtype)

        # 布尔型配置
        self.keep_press_var = tk.BooleanVar(value=CONFIG["keep_press_segment"])
        ttk.Checkbutton(self.scrollable_frame, text="保留按纸过程", variable=self.keep_press_var).grid(
            row=len(config_fields), columnspan=2)

        # 布局配置
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def save_config_to_memory(self):
        """内存保存逻辑"""
        try:
            for key, (entry, dtype) in self.config_entries.items():
                CONFIG[key] = dtype(entry.get())
            CONFIG["keep_press_segment"] = self.keep_press_var.get()
            messagebox.showinfo("成功", "内存配置已更新")
        except Exception as e:
            messagebox.showerror("错误", f"类型错误: {str(e)}")

    def export_config_to_file(self):
        """文件导出逻辑"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(CONFIG, f, indent=4)
                messagebox.showinfo("成功", f"配置已导出到\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败: {str(e)}")

    def clear_breakpoints(self):
        """清空所有断点"""
        if messagebox.askyesno("确认", "确定要清空所有断点吗?"):
            self.split_points = []
            self.segment_status = {}  # 同时清空审核状态
            self.update_plot()
            self.status_text.set("已清空所有断点和审核状态")

    def save_segments(self):
        """修改保存逻辑，仅保存合格样本"""

        if not self.split_points or self.resistance_data is None:
            messagebox.showwarning("警告", "无断点或数据，请先标注")
            return

        # 统计合格样本数量
        qualified_count = sum(self.segment_status.values()) if self.segment_status else 0
        if qualified_count == 0:
            messagebox.showwarning("警告", "没有合格样本可保存，请先审核样本")
            return

        # 创建输出目录

        base_output = self.custom_output_dir or CONFIG["output_dir"]
        base_output = os.path.join(base_output, os.path.splitext(os.path.basename(self.current_file))[0])

        os.makedirs(base_output, exist_ok=True)

        # 处理每个通道
        for ch_idx in range(CONFIG["num_channels"]):
            ch_name = str(CONFIG["channel_columns"][ch_idx]).replace("[", "").replace("]", "")
            ch_resistance = self.resistance_data[:, ch_idx]
            filtered_resistance = preprocess(ch_resistance, CONFIG)
            sorted_points = sorted(self.split_points)

            # 按全局断点分割段（仅处理合格样本）
            qualified_segments = []
            for i in range(len(sorted_points) - 1):
                # 跳过不合格样本
                if not self.segment_status.get(i, False):
                    continue

                start, end = sorted_points[i], sorted_points[i + 1]
                segment = filtered_resistance[start:end]
                if len(segment) < CONFIG["min_segment_length"]:
                    continue  # 跳过过短段
                qualified_segments.append((i, start, end, segment))

            # 保存合格分段
            ch_dir = os.path.join(base_output, f"ch{ch_idx}_{ch_name}")
            os.makedirs(ch_dir, exist_ok=True)

            for seg_idx, (orig_idx, start, end, seg_data) in enumerate(qualified_segments):
                # 标准化段长度
                if len(seg_data) != CONFIG["seq_length"]:
                    x_old = np.arange(len(seg_data))
                    x_new = np.linspace(0, len(seg_data) - 1, CONFIG["seq_length"])
                    seg_data = interp1d(x_old, seg_data, kind='linear')(x_new)

                # 保存npy文件（文件名包含原始段索引）
                npy_path = os.path.join(ch_dir, f"segment_{orig_idx}_{start}-{end}_qualified.npy")
                np.save(npy_path, seg_data)

                # 生成段可视化图
                self.visualize_single_segment(seg_data, start, end, orig_idx, ch_idx, ch_dir, is_qualified=True)

        # 保存断点和审核结果
        save_dir = CONFIG["manual_split"]["split_file_dir"]
        os.makedirs(save_dir, exist_ok=True)

        # 保存断点配置
        bp_path = os.path.join(
            save_dir,
            f"{os.path.splitext(os.path.basename(self.current_file))[0]}_global.csv"
        )
        pd.DataFrame(sorted(self.split_points)).to_csv(bp_path, index=False, header=False)

        # 保存审核结果
        status_path = os.path.join(
            save_dir,
            f"{os.path.splitext(os.path.basename(self.current_file))[0]}_status.csv"
        )
        pd.DataFrame([(k, v) for k, v in self.segment_status.items()],
                     columns=["segment_index", "is_qualified"]).to_csv(status_path, index=False)

        messagebox.showinfo("成功", f"已保存 {qualified_count} 个合格样本至:\n{base_output}")
        self.status_text.set(f"已保存 {qualified_count} 个合格样本: {base_output}")

    def visualize_single_segment(self, seg_data, start, end, seg_idx, ch_idx, save_dir, is_qualified):
        """修改可视化函数，添加合格标识"""
        plt.figure(figsize=(8, 4))
        plt.plot(seg_data, color=self.channel_colors[ch_idx], alpha=0.8)

        # 添加合格标识
        title = f"通道 {CONFIG['channel_columns'][ch_idx]} 段 {seg_idx} ({start}-{end})"
        if is_qualified:
            title += " [合格]"
            plt.axhspan(min(seg_data), max(seg_data), color='green', alpha=0.1)  # 合格样本绿色背景
        else:
            title += " [不合格]"


        plt.title(title)
        plt.xlabel("标准化时间步")
        plt.ylabel("电阻值（归一化）")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 合格样本文件名特殊标记
        qual_flag = "qualified" if is_qualified else "rejected"
        plt.savefig(os.path.join(save_dir, f"segment_{seg_idx}_{start}-{end}_{qual_flag}.png"))
        plt.close()

    def save_breakpoints(self):
        """保存断点配置并触发分段保存"""
        # 保存断点配置文件（供后续批处理使用）
        save_dir = self.custom_split_dir or CONFIG["manual_split"]["split_file_dir"]
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(
            save_dir,
            # 修复：使用self.current_file获取文件名，而非未定义的file_name
            f"{os.path.splitext(os.path.basename(self.current_file))[0]}_ch{self.current_channel}.csv"
        )


        pd.DataFrame(sorted(self.split_points)).to_csv(
            save_path, index=False, header=False
        )
        messagebox.showinfo("成功", f"断点已保存到:\n{save_path}")
        self.status_text.set(f"已保存断点: {save_path}")


# 将主程序逻辑移到类定义之外
def run_gui():
    """启动GUI应用"""
    # 修复：GUI启动时动态更新路径（与批处理模式保持一致）
    update_directory_paths(CONFIG)  # <-- 添加此行，确保路径动态生成
    root = tk.Tk()
    app = BreakpointAnnotator(root)
    root.mainloop()


# 将可视化函数移至全局作用域（与其他工具函数并列）
def visualize_combined_channels(resistance_data, split_points, file_name, save_dir, config):
    """生成四通道联合可视化图（两种模式共用）"""
    plt.figure(figsize=(12, 8))
    channel_colors = ['b', 'g', 'r', 'c']  # 与GUI颜色方案一致

    for ch_idx in range(config["num_channels"]):
        ch_resistance = resistance_data[:, ch_idx]
        filtered_resistance = preprocess(ch_resistance, config)
        plt.plot(filtered_resistance, color=channel_colors[ch_idx], alpha=0.7,
                 label=f'通道 {config["channel_columns"][ch_idx]}')

    # 绘制全局断点
    for point in split_points:
        plt.axvline(x=point, color='m', linestyle='--', alpha=0.7)

    plt.legend()
    plt.title(f"四通道联合标注 - {file_name}")
    plt.xlabel("时间步")
    plt.ylabel("电阻值（归一化）" if config["normalize_data"] else "电阻值")
    plt.grid(True, alpha=0.3)

    # 保存路径与GUI统一（文件目录下的combined_plot.png）
    save_path = os.path.join(save_dir, "combined_channels_plot.png")
    plt.savefig(save_path)
    plt.close()


# 修改主函数，支持GUI启动
if __name__ == "__main__":
    # 添加命令行参数判断（优先级高于配置文件）
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        run_gui()
    # 若无命令行参数，使用配置文件中的use_gui选项
    elif CONFIG.get("use_gui", False):
        run_gui()

    else:  # else关键字需单独成行并正确缩进
        process_dataset(CONFIG)
