import sys
import os
import time
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QFileDialog, QLabel, QLineEdit)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime


class RealTimeResistanceMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时电阻监测系统")
        self.setGeometry(300, 300, 1200, 800)

        # 主组件初始化
        self.init_ui()
        self.init_data_processor()
        self.init_monitor_timer()
        self.static_plot_count = 0
        self.last_value = None
        self.no_change_counter = 0

        # 配置默认保存路径
        self.default_save_dir = os.path.expanduser("./static_images")
        if not os.path.exists(self.default_save_dir):
            os.makedirs(self.default_save_dir)
        self.save_path_input.setText(self.default_save_dir)

    def init_ui(self):
        """初始化用户界面"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # 控制面板
        ctrl_panel = QWidget()
        ctrl_layout = QHBoxLayout()
        ctrl_panel.setLayout(ctrl_layout)

        # 文件选择组件
        self.browse_button = QPushButton("选择CSV文件")
        self.browse_button.clicked.connect(self.select_csv_file)
        self.file_label = QLabel("未选择文件")

        # 路径配置组件
        path_panel = QWidget()
        path_layout = QHBoxLayout()
        path_panel.setLayout(path_layout)
        self.save_path_input = QLineEdit()
        self.browse_save_button = QPushButton("浏览")
        self.browse_save_button.clicked.connect(self.select_save_directory)
        path_layout.addWidget(QLabel("保存路径:"))
        path_layout.addWidget(self.save_path_input)
        path_layout.addWidget(self.browse_save_button)

        # 控制按钮
        self.start_button = QPushButton("开始监测")
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button = QPushButton("停止监测")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.save_button = QPushButton("保存当前图表")
        self.save_button.clicked.connect(self.save_static_plot)

        # 状态显示
        self.status_label = QLabel("状态: 等待开始")
        self.value_label = QLabel("当前电阻值: - Ω")

        # 添加组件到控制面板
        ctrl_layout.addWidget(self.browse_button)
        ctrl_layout.addWidget(self.file_label)
        ctrl_layout.addWidget(path_panel)
        ctrl_layout.addWidget(self.start_button)
        ctrl_layout.addWidget(self.stop_button)
        ctrl_layout.addWidget(self.save_button)

        # 绘图区域
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.line, = self.ax.plot([], [], 'b-', linewidth=1.5)
        self.ax.set_title("实时电阻变化曲线")
        self.ax.set_xlabel("时间 (s)")
        self.ax.set_ylabel("电阻值 (Ω)")
        self.ax.grid(True)

        # 将所有组件添加到主布局
        layout.addWidget(ctrl_panel)
        layout.addWidget(self.status_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.canvas)

        # 初始禁用按钮
        self.toggle_buttons(False)

    def toggle_buttons(self, monitoring):
        """切换按钮状态"""
        self.start_button.setEnabled(not monitoring)
        self.stop_button.setEnabled(monitoring)
        self.save_button.setEnabled(monitoring)

    def select_csv_file(self):
        """选择CSV文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择实时数据CSV文件",
            "", "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if file_path:
            self.file_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.toggle_buttons(True)

    def select_save_directory(self):
        """选择保存目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择图表保存目录",
            self.default_save_dir
        )
        if dir_path:
            self.save_path_input.setText(dir_path)

    def init_data_processor(self):
        """初始化数据处理线程"""
        self.data_thread = DataProcessorThread()
        self.data_thread.new_data.connect(self.update_plot)
        self.data_thread.file_path = None
        self.data_buffer = []
        self.time_buffer = []
        self.start_time = None

    def init_monitor_timer(self):
        """初始化定时监测器"""
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.check_data_change)

    def start_monitoring(self):
        """开始监测"""
        if not hasattr(self, 'file_path'):
            self.status_label.setText("状态: 错误 - 未选择文件")
            return

        self.data_thread.file_path = self.file_path
        self.data_thread.start()
        self.monitor_timer.start(1000)  # 每秒检查一次数据变化
        self.start_time = time.time()
        self.toggle_buttons(True)
        self.status_label.setText("状态: 监测中...")

        # 重置缓冲区
        self.data_buffer = []
        self.time_buffer = []
        self.last_value = None
        self.no_change_counter = 0

    def stop_monitoring(self):
        """停止监测"""
        self.data_thread.running = False
        self.data_thread.quit()
        self.monitor_timer.stop()
        self.status_label.setText("状态: 已停止")
        self.save_static_plot(autosave=True)
        self.toggle_buttons(False)

    def check_data_change(self):
        """检查数据是否停止变化"""
        current_value = self.data_buffer[-1] if self.data_buffer else None

        if current_value is None:
            return

        if self.last_value is not None and abs(current_value - self.last_value) < 0.001:
            self.no_change_counter += 1
            if self.no_change_counter >= 5:  # 连续5次无变化
                self.status_label.setText("状态: 自动停止 - 数据无变化")
                self.stop_monitoring()
        else:
            self.no_change_counter = 0

        self.last_value = current_value

    def update_plot(self, new_data):
        """更新实时图表"""
        if not new_data:
            return

        current_time = time.time() - self.start_time

        # 追加新数据
        self.data_buffer.extend(new_data)
        self.time_buffer.extend([current_time + i * 0.1 for i in range(len(new_data))])  # 模拟时间轴

        # 限制数据量以提高性能
        max_points = 1000
        if len(self.data_buffer) > max_points:
            self.data_buffer = self.data_buffer[-max_points:]
            self.time_buffer = self.time_buffer[-max_points:]

        # 更新实时图表
        self.line.set_data(self.time_buffer, self.data_buffer)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

        # 更新当前值显示
        self.value_label.setText(f"当前电阻值: {self.data_buffer[-1]:.2f} Ω")

    def save_static_plot(self, autosave=False):
        """保存静态图表"""
        if not self.data_buffer:
            return

        save_dir = self.save_path_input.text()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if autosave:
            filename = f"autosave_{timestamp}.png"
        else:
            self.static_plot_count += 1
            filename = f"plot_{self.static_plot_count}_{timestamp}.png"

        save_path = os.path.join(save_dir, filename)

        # 创建独立的静态图表
        static_fig = Figure(figsize=(10, 6), dpi=100)
        static_ax = static_fig.add_subplot(111)
        static_ax.plot(self.time_buffer, self.data_buffer, 'r-', linewidth=1.5)
        static_ax.set_title(f"电阻变化曲线 ({timestamp})")
        static_ax.set_xlabel("时间 (s)")
        static_ax.set_ylabel("电阻值 (Ω)")
        static_ax.grid(True)

        # 保存图表
        static_canvas = FigureCanvas(static_fig)
        static_fig.savefig(save_path, bbox_inches='tight')

        if not autosave:
            self.status_label.setText(f"状态: 图表已保存到 {filename}")


class DataProcessorThread(QThread):
    """数据采集线程"""
    new_data = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.file_path = None
        self.running = False
        self.last_position = 0

    def run(self):
        self.running = True
        while self.running:
            try:
                # 实时读取CSV新增数据
                with open(self.file_path, 'r') as f:
                    f.seek(0, 2)  # 移动到文件末尾
                    if f.tell() < self.last_position:
                        self.last_position = 0  # 文件可能被重置
                    f.seek(self.last_position)

                    # 读取新行
                    new_lines = f.readlines()
                    if new_lines:
                        self.last_position = f.tell()
                        data = []
                        for line in new_lines[1:]:  # 跳过标题行
                            try:
                                value = float(line.strip().split(',')[-1])  # 取最后一列
                                data.append(value)
                            except (ValueError, IndexError):
                                continue

                        if data:
                            self.new_data.emit(data)
            except Exception as e:
                print(f"数据读取错误: {e}")

            time.sleep(0.1)  # 控制读取频率


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealTimeResistanceMonitor()
    window.show()
    sys.exit(app.exec_())
