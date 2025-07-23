import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
import chardet
import json
from tqdm import tqdm  # 用于显示处理进度

# 设置环境变量关闭 oneDNN 警告
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# === 处理配置（用于process_dataset）===
PROCESS_CONFIG = {
    "input_dir": "dataset/test",          # 原始数据目录（测试集CSV）
    "output_dir": "processed_data/test",  # 处理后NPY保存目录
    "visualization_dir": "visualizations/test",  # 可视化保存目录
    "min_writing_length": 28,  # 最小有效书写长度
    "seq_length": 50,  # 标准化序列长度
    "visualize_samples": False,  # 处理时不生成可视化
    "filter_window": 7,  # 滤波窗口大小
    "filter_polyorder": 3,  # 滤波多项式阶数
    "resistance_column": "DeviceInfo_[23082300]",  # 电阻值所在列名
    "normalize_data": True,  # 启用归一化
    "fixed_peak_threshold": 0.5,  # 极大值固定阈值
    "min_segment_length": 20,  # 最小分割段长度
    "keep_press_segment": False,  # 不保留按纸过程
    "press_threshold": 0.8,  # 按纸过程检测阈值
    "press_min_length": 30,  # 最小按纸长度
    "writing_descent_threshold": 0.4,  # 书写段下降阈值
    "debug_segment": True,  # 不输出段详细信息
}

# === 预测配置 ===
PREDICT_CONFIG = {
    "model_path": "models/test_bk/best_model.h5",  # 模型路径
    "processed_dir": "predict_samples/bk/xunlian/a",  # 处理后NPY目录
    "output_dir": "prediction_results/2.0.4/xunlian_a",  # 预测结果保存目录
    "seq_length": 50,  # 序列长度
    "global_normalization": True,  # 使用全局归一化
    # 原有配置保持不变
    # 添加校验参数
    "min_seq_length": 50,
    "max_seq_length": 1000
}

# === 绘图配置 ===
plt.rcParams["font.family"] = ["Arial", "sans-serif"]  # 使用英文字体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


### 数据处理函数 ###
def load_data(file_path, skip_rows=0, resistance_column="DeviceInfo_[23082300]"):
    """加载CSV数据，处理复合表头"""
    try:
        # 检测文件编码
        with open(file_path, 'rb') as f:
            raw_data = f.read(1024)
            result = chardet.detect(raw_data)
            encodings = [result['encoding'], 'utf-8', 'gbk', 'gb2312', 'latin-1']

        # 尝试多种编码加载
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, header=None)
                df.columns = ['DeviceNo', 'Unnamed: 1', resistance_column, 'DeviceRange_[--------]', 'Unnamed: 4']
                df = df[skip_rows:]  # 跳过指定行数
                df = df.reset_index(drop=True)

                if resistance_column not in df.columns:
                    print(f"错误：文件中不存在列 {resistance_column}")
                    return None

                resistance = pd.to_numeric(df[resistance_column], errors='coerce')
                resistance = resistance.dropna().values
                if len(resistance) == 0:
                    print("错误：处理后电阻值列为空")
                    return None

                return resistance

            except UnicodeDecodeError:
                continue

        print(f"无法使用任何编码加载文件 {file_path}")
        return None

    except Exception as e:
        print(f"加载文件 {file_path} 失败: {e}")
        return None


def normalize(resistance, global_min=None, global_max=None):
    """将电阻值归一化到[0,1]区间"""
    if len(resistance) == 0:
        return resistance
    if global_min is not None and global_max is not None:
        min_val, max_val = global_min, global_max
    else:
        min_val = np.min(resistance)
        max_val = np.max(resistance)
    if max_val == min_val:
        return np.zeros_like(resistance)
    return (resistance - min_val) / (max_val - min_val)


def preprocess(resistance, filter_window=7, filter_polyorder=3):
    """应用Savitzky-Golay滤波平滑数据"""
    from scipy.signal import savgol_filter
    if len(resistance) < filter_window:
        return resistance
    return savgol_filter(resistance, filter_window, filter_polyorder)


def split_by_continuity(resistance, config):
    """基于拐点检测分割按纸和书写过程"""
    if len(resistance) < 10:
        return [0, len(resistance)]

    processed_resistance = normalize(resistance)
    initial_voltage = np.mean(processed_resistance[:10])

    press_threshold = config.get("press_threshold", 0.8)
    press_min_length = config.get("press_min_length", 30)

    # 检测按纸开始和结束
    press_start = np.where(processed_resistance < press_threshold * initial_voltage)[0][0] if len(
        np.where(processed_resistance < press_threshold * initial_voltage)[0]) > 0 else 10
    press_min_idx = press_start + np.argmin(
        processed_resistance[press_start:press_start + press_min_length * 2])
    press_end = press_min_idx
    for i in range(press_min_idx + 1,
                   min(press_min_idx + press_min_length * 3, len(processed_resistance) - 1)):
        if processed_resistance[i] > processed_resistance[i - 1] and processed_resistance[i] > press_threshold * processed_resistance[press_min_idx]:
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

    fixed_threshold = config.get("fixed_peak_threshold", 0.5)
    filtered_maxima = [i for i in maxima_points if writing_segment[i] >= fixed_threshold]

    writing_split_points = []
    for point in filtered_maxima:
        if not writing_split_points or point - writing_split_points[-1] >= config.get("min_segment_length", 20):
            writing_split_points.append(point)

    split_points = [0, press_start, press_end] + [press_end + p for p in writing_split_points] + [
        len(processed_resistance) - 1]
    return split_points


def generate_samples(resistance, config):
    """从电阻数据中生成多个书写样本"""
    samples = []
    split_points = split_by_continuity(resistance, config)
    start_idx = 2  # 从书写段开始（跳过按纸过程）

    global_min = np.min(resistance)
    global_max = np.max(resistance)

    for i in range(start_idx, len(split_points) - 1):
        start, end = split_points[i], split_points[i + 1]
        raw_segment = resistance[start:end]

        if len(raw_segment) == 0:
            continue

        filtered_segment = preprocess(raw_segment, config.get("filter_window", 7), config.get("filter_polyorder", 3))

        if config.get("global_normalization", True):
            normalized_segment = normalize(filtered_segment, global_min, global_max)
        else:
            normalized_segment = normalize(filtered_segment)

        length = end - start
        if length < config.get("min_writing_length", 28):
            continue

        local_maxima, local_minima = [], []
        for j in range(1, len(normalized_segment) - 1):
            if (normalized_segment[j] > normalized_segment[j-1] and normalized_segment[j] > normalized_segment[j+1]):
                local_maxima.append(normalized_segment[j])
            if (normalized_segment[j] < normalized_segment[j-1] and normalized_segment[j] < normalized_segment[j+1]):
                local_minima.append(normalized_segment[j])

        if local_maxima and local_minima:
            peak_value = max(local_maxima)
            valley_value = min(local_minima)
            descent = peak_value - valley_value
        else:
            peak_value = np.max(normalized_segment)
            valley_value = np.min(normalized_segment)
            descent = peak_value - valley_value

        if descent >= config.get("writing_descent_threshold", 0.4):
            sample = normalized_segment
            # 修改硬编码引用
            if len(sample) < config["seq_length"]:
                sample = np.pad(sample, (0, config["seq_length"] - len(sample)), 'constant')
            else:
                sample = sample[:config["seq_length"]]
            samples.append(sample)

    return samples


def process_dataset(config):
    """处理数据集并生成NPY文件"""
    if not os.path.exists(config["input_dir"]):
        print(f"错误：输入目录 {config['input_dir']} 不存在")
        return 0, 0

    csv_files = [f for f in os.listdir(config["input_dir"]) if f.lower().endswith('.csv')]
    print(f"发现 {len(csv_files)} 个CSV文件，开始处理...")

    total_samples = 0
    processed_files = 0

    for file in tqdm(csv_files, desc="处理文件"):
        file_path = os.path.join(config["input_dir"], file)
        resistance = load_data(file_path, 3, config["resistance_column"])
        if resistance is None:
            continue

        samples = generate_samples(resistance, config)
        if not samples:
            continue

        output_dir = config["output_dir"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, f'{os.path.splitext(file)[0]}_processed.npy')
        np.save(output_path, np.array(samples))
        total_samples += len(samples)
        processed_files += 1

    print(f"处理完成! 成功处理 {processed_files} 个文件，生成 {total_samples} 个样本")
    print(f"处理后的数据保存在: {config['output_dir']}")
    return processed_files, total_samples


### 预测相关函数 ###
def predict_sample(model, sample, idx_to_label):
    """对单个样本进行预测"""
    model_input = sample.reshape(1, -1, 1).astype(np.float32)
    predictions = model.predict(model_input, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    class_name = idx_to_label.get(predicted_class, f"Unknown Class ({predicted_class})")
    return predicted_class, class_name, confidence, predictions[0]


def visualize_prediction(sample, predictions, class_name, confidence, file_name, sample_idx, output_dir, idx_to_label):
    """可视化预测结果"""
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(sample)
    plt.title(f"Resistance Sequence: {file_name} (Sample {sample_idx+1})", fontsize=14)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Resistance Value", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.bar(range(len(predictions)), predictions, color='skyblue', alpha=0.8)

    class_names = [idx_to_label.get(i, f"Class {i}") for i in range(len(predictions))]
    plt.xticks(range(len(predictions)), class_names, fontsize=10, rotation=45)

    plt.title(f"Prediction Result: {class_name} (Confidence: {confidence:.2f}%)", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Prediction Probability", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"prediction_{os.path.splitext(file_name)[0]}_sample{sample_idx+1}.png"), dpi=300)
    plt.close()


def main():
    """主预测流程：先处理CSV生成NPY，再加载NPY进行预测"""
    print(f"=== 开始预处理CSV文件 ===")
    processed_files, total_samples = process_dataset(PROCESS_CONFIG)
    if processed_files == 0:
        print("错误：未生成有效NPY文件，预测终止")
        return

    print(f"\n=== 开始模型预测 ===")

    # 加载label_map
    model_dir = os.path.dirname(PREDICT_CONFIG["model_path"])
    label_map_path = os.path.join(model_dir, "label_map.json")
    try:
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        idx_to_label = {int(idx): name for name, idx in label_map.items()}
        print(f"加载训练集标签映射: {idx_to_label}")
    except Exception as e:
        print(f"加载标签映射失败: {e}")
        return

    # 加载模型
    try:
        model = load_model(PREDICT_CONFIG["model_path"])
        print(f"模型加载成功: {PREDICT_CONFIG['model_path']}")
        print(f"模型输入形状: {model.input_shape}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 加载处理后的NPY文件
    processed_dir = PREDICT_CONFIG["processed_dir"]
    if not os.path.exists(processed_dir):
        print(f"错误：处理后NPY目录 {processed_dir} 不存在")
        return

    npy_files = [f for f in os.listdir(processed_dir) if f.lower().endswith('.npy')]
    if not npy_files:
        print(f"错误：在 {processed_dir} 中未找到NPY样本文件")
        return

    print(f"找到 {len(npy_files)} 个NPY文件，开始预测...")
    total_predicted = 0

    for file in npy_files:
        file_path = os.path.join(processed_dir, file)
        print(f"\n处理NPY文件: {file}")

        try:
            samples = np.load(file_path)
            print(f"加载 {len(samples)} 个样本")

            for idx, sample in enumerate(samples):
                # 执行预测
                predicted_class, class_name, confidence, predictions = predict_sample(
                    model, sample, idx_to_label
                )

                # 输出预测结果
                print(f"  样本 {idx+1}/{len(samples)} - 预测类别: {class_name} ({predicted_class})")
                print(f"  样本 {idx+1}/{len(samples)} - 置信度: {confidence:.2f}%")
                print(f"  样本 {idx+1}/{len(samples)} - 各类别概率: {np.round(predictions, 3)}")

                # 生成可视化图表
                visualize_prediction(sample, predictions, class_name, confidence, file, idx,
                                    PREDICT_CONFIG["output_dir"], idx_to_label)

            total_predicted += len(samples)

        except Exception as e:
            print(f"处理NPY文件 {file} 时出错: {e}")
            continue

    print(f"\n=== 预测完成 ===")
    print(f"共预测 {total_predicted} 个样本")
    print(f"预测结果已保存至: {PREDICT_CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
