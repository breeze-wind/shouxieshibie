import re
import csv
from datetime import datetime


def extract_csv_data(file_path):
    device_info = {"DeviceNo": "", "DeviceRange": ""}
    records = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                # 提取设备信息
                if line.startswith('DeviceNo'):
                    match = re.search(r'\[([^\]]+)\]', line)
                    if match:
                        device_info["DeviceNo"] = match.group(1)

                elif line.startswith('DeviceRange'):
                    match = re.search(r'\[([^\]]+)\]', line)
                    if match:
                        device_info["DeviceRange"] = match.group(1)

                # 提取数据记录
                elif re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}:\d{3}', line[:23]):
                    parts = line.split(maxsplit=3)
                    if len(parts) >= 3:
                        timestamp_str = parts[0] + ' ' + parts[1]
                        value_str = parts[2].strip()

                        # 验证数据格式
                        if not re.fullmatch(r'\d{5}', value_str):
                            raise ValueError(f"数据格式错误: {line}")

                        records.append({
                            "timestamp": timestamp_str,
                            "value": value_str
                        })

    except Exception as e:
        # 错误处理不修改原始数据
        raise RuntimeError(f"数据处理异常: {str(e)}") from e

    return {
        "device_info": device_info,
        "records": records
    }


# 使用示例
if __name__ == "__main__":
    data = extract_csv_data("../../dataset/u_bk/Sample20250706102612.csv")

    # 保存设备信息
    with open("device_info.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["DeviceNo", "DeviceRange"])
        writer.writeheader()
        writer.writerow(data["device_info"])

    # 保存原始数据
    with open("raw_data.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "value"])
        writer.writeheader()
        for record in data["records"]:
            # 保持数据原始格式
            writer.writerow({
                "timestamp": record["timestamp"],
                "value": record["value"]  # 保留前导零
            })