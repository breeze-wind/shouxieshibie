import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from process_samples import preprocess, normalize, CONFIG  # 复用样本处理函数

# === 配置参数 ===
PREDICT_CONFIG = {
    "model_path": "models/best_model.h5",  # 模型路径
    "data_path": "dataset/a_bk/Sample20250624131411.csv",  # 测试样本目录
    "seq_length": 800,  # 序列长度
    "output_dir": "prediction_results",  # 结果保存目录
    "label_map": {0: "类别0", 1: "类别1", 2: "类别2", 3: "类别3", 4: "类别4"},  # 标签映射
}


# === 预测函数 ===
def load_and_preprocess_sample(file_path):
    """加载并预处理单个样本"""
    # 加载数据
    resistance = np.loadtxt(file_path) if file_path.endswith('.csv') else np.load(file_path)

    # 数据预处理
    filtered_resistance = preprocess(resistance, CONFIG)

    # 归一化
    normalized_data = normalize(filtered_resistance)

    # 调整到固定长度
    if len(normalized_data) < PREDICT_CONFIG["seq_length"]:
        normalized_data = np.pad(normalized_data,
                                 (0, PREDICT_CONFIG["seq_length"] - len(normalized_data)),
                                 'constant')
    else:
        normalized_data = normalized_data[:PREDICT_CONFIG["seq_length"]]

    # 重塑为模型输入格式: (1, seq_length, 1)
    sample = normalized_data.reshape(1, PREDICT_CONFIG["seq_length"], 1)

    return sample, resistance


def predict_sample(model, sample, label_map):
    """对单个样本进行预测"""
    # 模型预测
    predictions = model.predict(sample, verbose=0)

    # 获取预测类别和置信度
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    # 获取类别名称
    class_name = label_map.get(predicted_class, f"未知类别({predicted_class})")

    return predicted_class, class_name, confidence, predictions[0]


def visualize_prediction(resistance, predictions, class_name, confidence, file_name, output_dir):
    """可视化预测结果"""
    plt.figure(figsize=(12, 8))

    # 绘制电阻值序列
    plt.subplot(2, 1, 1)
    plt.plot(resistance)
    plt.title(f"电阻值序列: {file_name}")
    plt.xlabel("时间点")
    plt.ylabel("电阻值")

    # 绘制预测概率分布
    plt.subplot(2, 1, 2)
    plt.bar(range(len(predictions)), predictions, color='skyblue')
    plt.xticks(range(len(predictions)),
               [PREDICT_CONFIG["label_map"].get(i, f"类别{i}") for i in range(len(predictions))])
    plt.title(f"预测结果: {class_name} (置信度: {confidence:.2f}%)")
    plt.xlabel("类别")
    plt.ylabel("预测概率")
    plt.ylim(0, 1)

    plt.tight_layout()

    # 保存可视化结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"prediction_{os.path.splitext(file_name)[0]}.png"))
    plt.close()


def main():
    """主预测流程"""
    print(f"开始模型预测... (配置: {PREDICT_CONFIG})")

    # 加载模型
    try:
        model = load_model(PREDICT_CONFIG["model_path"])
        print(f"模型加载成功: {PREDICT_CONFIG['model_path']}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 获取测试样本
    test_files = [f for f in os.listdir(PREDICT_CONFIG["data_path"])
                  if f.endswith(('.txt', '.npy'))]

    if not test_files:
        print(f"错误：在 {PREDICT_CONFIG['data_path']} 中未找到测试样本")
        return

    print(f"找到 {len(test_files)} 个测试样本")

    # 对每个样本进行预测
    for file in test_files:
        file_path = os.path.join(PREDICT_CONFIG["data_path"], file)
        print(f"\n正在预测: {file}")

        # 加载并预处理样本
        sample, original_data = load_and_preprocess_sample(file_path)

        # 预测
        predicted_class, class_name, confidence, predictions = predict_sample(
            model, sample, PREDICT_CONFIG["label_map"]
        )

        # 输出预测结果
        print(f"预测类别: {class_name} ({predicted_class})")
        print(f"预测置信度: {confidence:.2f}%")
        print(f"所有类别概率: {np.round(predictions, 3)}")

        # 可视化预测结果
        visualize_prediction(original_data, predictions, class_name, confidence, file, PREDICT_CONFIG["output_dir"])

    print(f"\n预测完成！结果已保存至: {PREDICT_CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
