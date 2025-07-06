import pandas as pd
import matplotlib.pyplot as plt

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 使用相对路径读取文件，假设文件名为 Sample20250624213927.csv 且与代码在同一目录
file_path = './dataset/deleted/n_bk/jq.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# 假设电阻值在第三列（索引为 2），将该列转换为数值类型
df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')

# 生成从 0 开始的单位时间序列
time_sequence = list(range(len(df)))

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(time_sequence, df.iloc[:, 2])

# 添加标题和轴标签
plt.title('电阻值折线图')
plt.xlabel('单位时间')
plt.xticks(rotation=45)
plt.ylabel('电阻值')

# 保存图表为 PNG 文件（默认为当前目录）
plt.savefig('resistance_chart.png', bbox_inches='tight')
print("图表已保存为 resistance_chart.png")

# 显示图表
plt.show()
