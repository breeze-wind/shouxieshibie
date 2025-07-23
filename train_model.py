import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D  # 添加GlobalAveragePooling1D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import time
import json  # 用于保存label_map

# === 配置参数 ===
CONFIG = {
    "data_dir": "processed_data/test/bk",  # 处理后样本目录
    "model_dir": "models/test_bk",  # 模型保存目录
    "seq_length": 50,  # 统一序列长度
    "batch_size": 32,  # 训练批次大小
    "epochs": 500,  # 最大训练轮数
    "n_classes": 10,  # 分类类别数（需根据实际数据修改）
    # 添加校验参数
    "min_seq_length": 50,
    "max_seq_length": 1000,
    "use_lstm": False,  # 新增LSTM开关配置
}


# === 数据加载与预处理 ===
def load_processed_data(data_dir, seq_length=None):
    """加载处理后的样本数据及标签"""
    all_samples = []
    all_labels = []
    label_counter = 0
    label_map = {}

    # 假设目录结构为: data_dir/label_XXX/*.npy
    label_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if not label_dirs:
        # 若没有子目录，假设所有.npy文件属于同一类别
        label_dirs = [data_dir]
        label_map[data_dir] = 0
        label_counter = 1

    for label_dir in label_dirs:
        full_path = os.path.join(data_dir, label_dir)
        if not os.path.isdir(full_path):
            continue

        # 为当前标签分配索引
        if label_dir not in label_map:
            label_map[label_dir] = label_counter
            label_counter += 1

        # 加载所有样本文件
        for file in os.listdir(full_path):
            if file.endswith("_processed.npy"):
                file_path = os.path.join(full_path, file)
                try:
                    samples = np.load(file_path)
                    all_samples.append(samples)
                    all_labels.extend([label_map[label_dir]] * len(samples))
                except Exception as e:
                    print(f"加载样本文件 {file_path} 失败: {e}")

    if not all_samples:
        raise ValueError(f"在 {data_dir} 中未找到有效样本文件")

    # 合并所有样本
    X = np.concatenate(all_samples, axis=0)
    y = np.array(all_labels)

    # 添加长度校验
    if seq_length < CONFIG["min_seq_length"] or seq_length > CONFIG["max_seq_length"]:
        raise ValueError(f"序列长度{seq_length}超出允许范围({CONFIG['min_seq_length']}-{CONFIG['max_seq_length']})")

    # 重塑数据以适应CNN-LSTM输入: (样本数, 序列长度, 特征数)
    X = X.reshape(-1, seq_length, 1)

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
        Conv1D(filters=64, kernel_size=7, activation='relu', padding='same',
               input_shape=(seq_length, 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.1),

        Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.1),

        Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.1),

        Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.1),

        # 新增LSTM开关逻辑
        *([  # 当use_lstm=True时添加的层
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(64)),
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
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
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
    # 这里根据label_map的键（类别名称）和值（类别索引），构建分类报告的目标名称
    target_names = [k for k, v in label_map.items()]
    report = classification_report(y_test_classes, y_pred_classes, target_names=target_names)
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