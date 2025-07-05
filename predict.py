import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from process_samples import preprocess, normalize, CONFIG, load_data
import pandas as pd
import chardet
import json

# 设置环境变量关闭 oneDNN 警告
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# === 配置参数 ===
PREDICT_CONFIG = {
    "model_path": "models/best_model.h5",  # 模型路径
    "data_path": "dataset/test",  # 测试数据目录
    "seq_length": 800,  # 序列长度
    "output_dir": "prediction_results",  # 结果保存目录
    "skip_rows": 3,  # 跳过前3行
}

# === 绘图配置 ===
plt.rcParams["font.family"] = ["Arial", "sans-serif"]  # 使用英文字体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


### 函数定义 ###
def detect_encoding(file_path):
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))  # 读取前10KB用于编码检测
    return result['encoding']


def load_and_preprocess_sample(file_path, skip_rows=0):
    """加载并预处理样本数据"""
    encoding = detect_encoding(file_path)
    print(f"检测到文件编码: {encoding}")

    resistance = load_data(file_path)
    if resistance is None:
        raise ValueError(f"加载数据失败: {file_path}")

    # 跳过指定行数
    if skip_rows > 0 and len(resistance) > skip_rows:
        resistance = resistance[skip_rows:]

    # 数据预处理（滤波和归一化）
    filtered_resistance = preprocess(resistance, CONFIG)
    normalized_data = normalize(filtered_resistance)

    # 调整序列长度至固定值
    if len(normalized_data) < PREDICT_CONFIG["seq_length"]:
        normalized_data = np.pad(normalized_data,
                                 (0, PREDICT_CONFIG["seq_length"] - len(normalized_data)),
                                 'constant')
    else:
        normalized_data = normalized_data[:PREDICT_CONFIG["seq_length"]]

    # 重塑为模型输入格式
    sample = normalized_data.reshape(1, PREDICT_CONFIG["seq_length"], 1)
    return sample, resistance


def predict_sample(model, sample, idx_to_label):
    """对单个样本进行预测"""
    predictions = model.predict(sample, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    class_name = idx_to_label.get(predicted_class, f"Unknown Class ({predicted_class})")
    return predicted_class, class_name, confidence, predictions[0]


def visualize_prediction(resistance, predictions, class_name, confidence, file_name, output_dir, idx_to_label):
    """可视化预测结果（使用训练集的类别名称）"""
    plt.figure(figsize=(12, 8))

    # 绘制电阻值序列图
    plt.subplot(2, 1, 1)
    plt.plot(resistance)
    plt.title(f"Resistance Sequence: {file_name}", fontsize=14)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Resistance Value", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 绘制预测概率分布图
    plt.subplot(2, 1, 2)
    plt.bar(range(len(predictions)), predictions, color='skyblue', alpha=0.8)

    # 使用训练集的类别名称
    class_names = [idx_to_label.get(i, f"Class {i}") for i in range(len(predictions))]
    plt.xticks(range(len(predictions)), class_names, fontsize=10, rotation=45)

    plt.title(f"Prediction Result: {class_name} (Confidence: {confidence:.2f}%)", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Prediction Probability", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存可视化结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"prediction_{os.path.splitext(file_name)[0]}.png"), dpi=300)
    plt.close()


def main():
    """主预测流程"""
    print(f"开始模型预测...")

    # 从 models 文件夹加载 label_map
    model_dir = os.path.dirname(PREDICT_CONFIG["model_path"])
    label_map_path = os.path.join(model_dir, "label_map.json")

    try:
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        # 将标签映射转换为 {索引: 类别名称} 格式
        idx_to_label = {int(idx): name for name, idx in label_map.items()}
        print(f"加载训练集标签映射: {idx_to_label}")

    except FileNotFoundError:
        print(f"错误：未找到标签映射文件 {label_map_path}，请先在训练脚本中保存标签映射！")
        return
    except json.JSONDecodeError:
        print(f"错误：{label_map_path} 不是有效的JSON文件，请检查格式")
        return
    except Exception as e:
        print(f"加载标签映射时出错: {e}")
        return

    # 检查依赖库
    try:
        import chardet
    except ImportError:
        print("错误：需要安装chardet库: pip install chardet")
        return

    # 加载模型
    try:
        model = load_model(PREDICT_CONFIG["model_path"])
        print(f"模型加载成功: {PREDICT_CONFIG['model_path']}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 检查测试数据目录
    data_path = PREDICT_CONFIG["data_path"]
    if not os.path.exists(data_path):
        print(f"错误：测试数据目录 {data_path} 不存在")
        return

    # 获取测试样本
    try:
        all_files = os.listdir(data_path)
    except Exception as e:
        print(f"读取目录失败: {e}")
        return

    # 筛选支持的文件格式
    test_files = [
        f for f in all_files
        if os.path.splitext(f)[1].lower() in ('.txt', '.npy', '.csv')
    ]
    if not test_files:
        print(f"错误：在 {data_path} 中未找到支持的测试样本（支持格式: .txt, .npy, .csv）")
        return

    print(f"找到 {len(test_files)} 个测试样本: {test_files}")

    # 逐个处理样本
    for file in test_files:
        file_path = os.path.join(data_path, file)
        print(f"\n正在预测: {file}")

        try:
            # 加载并预处理样本
            sample, original_data = load_and_preprocess_sample(file_path, PREDICT_CONFIG["skip_rows"])

            # 执行预测
            predicted_class, class_name, confidence, predictions = predict_sample(
                model, sample, idx_to_label
            )

            # 输出预测结果
            print(f"预测类别: {class_name} ({predicted_class})")
            print(f"置信度: {confidence:.2f}%")
            print(f"各类别概率: {np.round(predictions, 3)}")

            # 生成可视化图表
            visualize_prediction(original_data, predictions, class_name, confidence, file,
                                 PREDICT_CONFIG["output_dir"], idx_to_label)

        except Exception as e:
            print(f"处理样本 {file} 时出错: {e}")
            continue

    print(f"\n预测完成！结果已保存至: {PREDICT_CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
