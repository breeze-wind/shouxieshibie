import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D,AveragePooling1D, LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D  # 添加GlobalAveragePooling1D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import time
import json  # 用于保存label_map
import joblib

# === 配置参数 ===
CONFIG = {
    "data_dir": "processed_data/data/ying",  # 处理后样本目录
    "model_dir": "models/data/ying",  # 模型保存目录
    "seq_length": 50,  # 统一序列长度
    "batch_size": 64,  # 训练批次大小
    "epochs": 500,  # 最大训练轮数
    "n_classes": 26,  # 分类类别数（需根据实际数据修改）
    # 添加校验参数
    "min_seq_length": 50,
    "max_seq_length": 1000,
    "use_lstm": True,  # 新增LSTM开关配置
}


# === 数据加载与预处理 ===
def load_processed_data(data_dir, seq_length=None):
    """加载处理后的样本数据及标签"""
    all_samples = []
    all_labels = []
    label_map = {}

    # 获取第一层目录作为标签
    label_dirs = [d for d in os.listdir(data_dir)
                 if os.path.isdir(os.path.join(data_dir, d))]

    if not label_dirs:
        # 若没有子目录，使用当前目录名作为标签
        current_dir = os.path.basename(data_dir.rstrip('/'))
        label_map[current_dir] = 0
        label_dirs = [data_dir]
    else:
        # 建立标签映射
        label_map = {label: idx for idx, label in enumerate(sorted(label_dirs))}

    # 递归读取所有子目录中的npy文件
    for label in label_map:
        label_path = os.path.join(data_dir, label)

        # 添加样本计数器
        label_sample_count = 0

        # 递归遍历所有子目录
        for root, dirs, files in os.walk(label_path):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    try:
                        # 加载numpy数组
                        samples = np.load(file_path)

                        # 添加维度校验
                        if samples.ndim == 0:
                            raise ValueError(f"文件 {file_path} 包含标量数据")

                        # 修正样本处理逻辑
                        if samples.ndim == 1:
                            # 单个样本 (seq_length,)
                            if len(samples) != CONFIG["seq_length"]:
                                raise ValueError(f"文件 {file_path} 长度不匹配: {len(samples)} != {CONFIG['seq_length']}")
                            all_samples.append(samples)
                            all_labels.append(label_map[label])
                            label_sample_count += 1
                        elif samples.ndim == 2:
                            # 批量样本 (n_samples, seq_length)
                            for sample in samples:
                                all_samples.append(sample)
                                all_labels.append(label_map[label])
                                label_sample_count += 1
                        else:
                            raise ValueError(f"文件 {file_path} 维度异常: {samples.ndim}")

                    except Exception as e:
                        print(f"加载样本文件 {file_path} 失败: {e}")

        # 添加类别样本数统计
        print(f"类别 {label} 加载样本数: {label_sample_count}")

    # 保持训练脚本的现有处理逻辑（约73-81行）
    # 合并所有样本
    X = np.concatenate(all_samples, axis=0)
    y = np.array(all_labels)

    # 添加长度校验（保持原校验逻辑不变）
    if seq_length < CONFIG["min_seq_length"] or seq_length > CONFIG["max_seq_length"]:
        raise ValueError(f"序列长度{seq_length}超出允许范围({CONFIG['min_seq_length']}-{CONFIG['max_seq_length']})")

    # 统一重塑逻辑（与预测脚本保持一致）
    X = X.reshape(-1, CONFIG["seq_length"], 1)

    print(f"数据加载完成: 样本数={X.shape[0]}, 类别数={len(label_map)}")
    print(f"类别映射: {label_map}")
    return X, y, label_map


def preprocess_data(X, y, n_classes):
    """数据预处理：标准化、划分数据集"""
    # 标准化数据
    X_2d = X.reshape(-1, X.shape[1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_2d)
    X = X_scaled.reshape(X.shape)
    print(f"标准化后数据: mean={X.mean():.4f}, std={X.std():.4f}")

    # 保存标准化器（新增目录创建逻辑）
    os.makedirs(CONFIG["model_dir"], exist_ok=True)  # <-- 新增目录创建
    scaler_path = os.path.join(CONFIG["model_dir"], "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"标准化器已保存至: {scaler_path}")

    # 划分训练集、验证集和测试集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"数据集划分: 训练集={X_train.shape}, 验证集={X_val.shape}, 测试集={X_test.shape}")

    # 转换标签为one-hot编码
    y_train_onehot = to_categorical(y_train, n_classes)
    y_val_onehot = to_categorical(y_val, n_classes)
    y_test_onehot = to_categorical(y_test, n_classes)

    return X_train, X_val, X_test, y_train_onehot, y_val_onehot, y_test_onehot


# === 模型构建与训练 ===
def build_cnn_lstm_model(seq_length=800, n_classes=10):
    """构建CNN-LSTM模型"""
    model = Sequential([
        # CNN层 - 提取局部特征
        Conv1D(filters=256, kernel_size=8, activation='relu', padding='causal',
               input_shape=(seq_length, 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=128, kernel_size=10, activation='relu', padding='same',
               input_shape=(seq_length, 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=128, kernel_size=11, activation='relu', padding='causal'),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),

        Conv1D(filters=256, kernel_size=12, activation='relu', padding='causal'),
        AveragePooling1D(pool_size=2),
        Dropout(0.4),

        Conv1D(filters=512, kernel_size=15, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # 新增LSTM开关逻辑
        *([  # 当use_lstm=True时添加的层
            Bidirectional(LSTM(256, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(256)),
            Dropout(0.3),
        ] if CONFIG["use_lstm"] else [
            # 当关闭LSTM时可选的替代结构（可选）
            GlobalAveragePooling1D()
        ]),

        # 分类输出层保持不变
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


def train_model(X_train, X_val, y_train, y_val, n_classes, seq_length, epochs, batch_size):
    """训练模型并返回训练历史"""
    # 确保模型目录存在
    if not os.path.exists(CONFIG["model_dir"]):
        os.makedirs(CONFIG["model_dir"])

    # 创建模型
    model = build_cnn_lstm_model(seq_length, n_classes)

    # 设置回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(CONFIG["model_dir"], 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]

    # 开始训练
    start_time = time.time()
    print(f"\n开始训练模型... (批次大小={batch_size}, 轮数={epochs})")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - start_time
    print(f"训练完成！耗时: {elapsed:.2f}秒")

    return model, history


# === 模型评估 ===
def evaluate_model(model, X_test, y_test, y_test_classes, label_map):
    """评估模型性能"""
    # 在测试集上评估
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n测试集评估: 损失={test_loss:.4f}, 准确率={test_acc:.4f}")

    # 预测并生成混淆矩阵
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    print(cm)

    # 生成分类报告
    print("\n分类报告:")
    # 获取实际存在的类别索引
    present_labels = np.unique(y_test_classes)
    # 根据存在的类别构建目标名称
    target_names = [k for k, v in label_map.items() if v in present_labels]

    report = classification_report(
        y_test_classes,
        y_pred_classes,
        labels=present_labels,
        target_names=target_names
    )
    print(report)

    return cm, report


# === 可视化训练过程（英文绘图）===
def plot_training_history(history):
    """可视化训练历史"""
    plt.figure(figsize=(12, 4))
    plt.rcParams["font.family"] = ["Arial", "sans-serif"]  # 确保英文字体

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("训练历史图表已保存为 training_history.png")


# === 主函数 ===
def main():
    """模型训练主流程"""
    start_time = time.time()
    print(f"开始模型训练... (配置: {CONFIG})")

    try:
        # 1. 加载处理后的样本数据
        X, y, label_map = load_processed_data(CONFIG["data_dir"], CONFIG["seq_length"])

        # 2. 数据预处理
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
            X, y, CONFIG["n_classes"]
        )

        # 3. 训练模型
        model, history = train_model(
            X_train, X_val, y_train, y_val,
            CONFIG["n_classes"], CONFIG["seq_length"],
            CONFIG["epochs"], CONFIG["batch_size"]
        )

        # 4. 评估模型
        y_test_classes = np.argmax(y_test, axis=1)
        evaluate_model(model, X_test, y_test, y_test_classes, label_map)

        # 5. 可视化训练历史
        plot_training_history(history)

        # 6. 保存label_map到模型目录
        label_map_path = os.path.join(CONFIG["model_dir"], "label_map.json")
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
        print(f"标签映射已保存至: {label_map_path}")

        print(f"\n模型训练完成！最佳模型已保存至: {os.path.join(CONFIG['model_dir'], 'best_model.h5')}")

    except Exception as e:
        print(f"模型训练出错: {e}")
    finally:
        elapsed = time.time() - start_time
        print(f"总耗时: {elapsed:.2f}秒")


if __name__ == "__main__":
    main()