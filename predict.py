import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import json
from tqdm import tqdm  # 用于显示处理进度
import joblib
# 设置环境变量关闭 oneDNN 警告
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# === 预测配置 ===
PREDICT_CONFIG = {
    "model_path": "models/test_bk/best_model.h5",  # 模型路径
    "processed_dir": "predict_samples/bk/yuce"
                     ,  # 处理后NPY目录
    "output_dir": "prediction_results/2.0.6/bk",  # 预测结果保存目录
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
    """主预测流程：加载NPY进行预测"""
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

    # 加载标准化器
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    try:
        scaler = joblib.load(scaler_path)
        print(f"标准化器加载成功: {scaler_path}")
    except Exception as e:
        print(f"标准化器加载失败: {e}")
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
                # 应用标准化
                sample_2d = sample.reshape(1, -1)  # 调整为 (1, 50) 形状
                sample_scaled = scaler.transform(sample_2d)
                sample = sample_scaled.reshape(-1, 1)  # 恢复为 (50, 1) 形状

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
