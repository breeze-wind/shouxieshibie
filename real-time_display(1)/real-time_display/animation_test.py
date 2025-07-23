import math
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# 创建一个新的Tkinter窗口
root = tk.Tk()
root.title("IMU")

# 创建一个新的matplotlib图形
fig = plt.figure(num="IMU数据UI")

# 创建三个空的列表来存储数据
pitch_data = []
roll_data = []
yaw_data = []

# 创建三个空的列表来存储数据
frame_count = 0  # 添加帧计数器


# 创建三个空的列表来存储数据
def animate(i):
    global frame_count

    # 生成正弦波数据 (不同频率和相位)
    pitch = math.sin(frame_count * 0.05)  # 基础正弦波
    roll = math.sin(frame_count * 0.05 + math.pi / 3)  # 相位偏移
    yaw = math.sin(frame_count * 0.1)  # 双倍频率

    # 添加数据到列表
    pitch_data.append(pitch)
    roll_data.append(roll)
    yaw_data.append(yaw)

    # 限制数据长度
    max_points = 200
    if len(pitch_data) > max_points:
        del pitch_data[0]
    if len(roll_data) > max_points:
        del roll_data[0]
    if len(yaw_data) > max_points:
        del yaw_data[0]

    # 清除当前的图形
    plt.cla()

    # 绘制新的图形
    plt.plot(pitch_data, label='Pitch', color='blue')
    plt.plot(roll_data, label='Roll', color='green')
    plt.plot(yaw_data, label='Yaw', color='red')

    # 添加图例和标题
    plt.legend(loc='upper right')
    plt.title(f'IMU 数据 (帧: {frame_count})')
    plt.xlabel('时间')
    plt.ylabel('数值')
    plt.ylim(-1.5, 1.5)  # 固定Y轴范围
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    frame_count += 1  # 更新帧计数器


# 创建动画
ani = animation.FuncAnimation(
    fig,
    animate,
    interval=20,  # 20ms更新一次 (50FPS)
    cache_frame_data=False  # 提高性能
)

# 显示图形
plt.show()

# 主循环
root.mainloop()