import csv
import time
import random
from datetime import datetime, timedelta
import os


class DataGenerator:
    """电阻数据模拟生成器（续接现有CSV文件）"""

    def __init__(self, csv_file, interval=0.25):
        """
        csv_file: 输出CSV文件路径
        interval: 生成数据间隔(秒)，默认0.25秒(每秒4个点)
        """
        self.csv_file = csv_file
        self.interval = interval
        self.last_time = self._get_last_timestamp()

        # 检查文件是否存在，不存在则初始化
        if not os.path.exists(csv_file) or self.last_time is None:
            print(f"初始化新CSV文件: {csv_file}")
            self.init_csv()
            self.last_time = datetime.now()  # 使用当前时间作为起始时间

    def _get_last_timestamp(self):
        """读取CSV文件的最后一行时间戳"""
        if not os.path.exists(self.csv_file):
            print("文件不存在")
            return None

        try:
            # 检查文件大小
            file_size = os.path.getsize(self.csv_file)
            if file_size == 0:
                print("文件为空")
                return None

            with open(self.csv_file, 'r', encoding='gbk') as f:
                lines = f.readlines()

                # 检查是否只有表头（至少需要4行：3行表头+1行数据）
                if len(lines) <= 3:
                    print(f"文件只有表头，行数: {len(lines)}")
                    return None

                # 获取最后一行数据（跳过空行）
                last_line = None
                for line in reversed(lines):
                    if line.strip():
                        last_line = line
                        break

                if not last_line:
                    print("未找到有效数据行")
                    return None

                # 解析时间戳
                try:
                    # 格式："2025-07-07\t15:30:45:123"
                    time_str = last_line.split('\t')[0]
                    return datetime.strptime(time_str, '%Y-%m-%d\t%H:%M:%S:%f')
                except ValueError as ve:
                    print(f"时间戳解析失败: {ve}")
                    print(f"错误的时间戳: {time_str}")
                    return None

        except Exception as e:
            print(f"读取最后时间戳失败: {e}")
            return None

    def init_csv(self):
        """初始化CSV文件格式"""
        with open(self.csv_file, 'w', newline='', encoding='gbk') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["DeviceNo_DeviceInfo", "", "[23082300]_[]", "[--------]_[]", ""])
            writer.writerow(["DeviceInfo", "", "[]", "[]", ""])
            writer.writerow(["DeviceRange", "", "[20M次]", "[--]", ""])

    def generate_time(self):
        """生成连续时间戳（从上一个时间点继续）"""
        if self.last_time is None:
            self.last_time = datetime.now()
        else:
            self.last_time += timedelta(milliseconds=int(self.interval * 1000))
        return self.last_time.strftime('%Y-%m-%d\t%H:%M:%S:%f')[:-3]

    def generate_resistance(self):
        """生成0-7000欧姆的随机电阻值"""
        resistance = int(random.gauss(3500, 1000))
        resistance = max(0, min(7000, resistance))
        return f"{resistance:05d}"

    def run(self):
        """开始生成数据"""
        print(f"开始续接电阻数据（0-7000欧姆），写入: {self.csv_file}")
        print("按Ctrl+C停止")

        try:
            while True:
                # 使用追加模式打开文件
                with open(self.csv_file, 'a', newline='', encoding='gbk') as f:
                    writer = csv.writer(f, delimiter='\t')

                    # 每秒生成4个数据点
                    for _ in range(4):
                        time_str = self.generate_time()
                        resistance = self.generate_resistance()
                        writer.writerow([time_str, '', resistance, '----', 'nan'])

                        # 打印当前生成的数据
                        print(f"生成: {time_str} - 电阻: {resistance}Ω")

                        # 等待间隔时间
                        time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n数据生成已停止")


if __name__ == "__main__":
    # 指定要续接的CSV文件
    csv_file = "../dataset/test/Sample20250713150931.csv"  # 替换为您的CSV文件名

    generator = DataGenerator(csv_file)
    generator.run()