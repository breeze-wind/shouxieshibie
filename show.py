import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 使用相对路径读取文件
file_path = 'dataset/a_rl/Sample20250713163827.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# 获取csv文件所在目录和文件名
csv_dir = os.path.dirname(file_path)  # 新增：获取csv所在目录
csv_filename = os.path.splitext(os.path.basename(file_path))[0]

# 创建png存储目录（与csv同级的png文件夹）
png_dir = os.path.join(csv_dir, 'png')  # 新增：构建png目录路径
os.makedirs(png_dir, exist_ok=True)     # 新增：自动创建目录

# 假设电阻值在第三列（索引为 2），将该列转换为数值类型
df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')

# 生成从 0 开始的单位时间序列
time_sequence = list(range(len(df)))

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(time_sequence, df.iloc[:, 2])

# 添加标题和轴标签
plt.title(csv_filename)  # 修复：添加引号
plt.xlabel('unit time')
plt.xticks(rotation=45)
plt.ylabel('resistance')

# 修改保存路径逻辑
plt.savefig(os.path.join(png_dir, f'{csv_filename}.png'), bbox_inches='tight')  # 修改路径
print(f"图表已保存为 {os.path.join(png_dir, csv_filename)}.png")  # 修改输出信息

# 显示图表
plt.show()