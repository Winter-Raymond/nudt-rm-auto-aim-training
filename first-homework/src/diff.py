import csv
import os

# 路径配置（根据你的目录结构写死）
# 你的 CSV 文件路径
cpp_csv_path = "./cpp/output_data/frame_speed.csv"
python_csv_path = "./python/data/speed.csv"
output_csv_path = "./diff_result.csv"


def read_csv(csv_path):
    """读取 CSV，返回帧号 -> (speed, angle) 的字典"""
    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame_idx"])
            speed = float(row["speed_pixel"])
            angle = float(row["direction_deg"])
            data[frame_idx] = (speed, angle)
    return data


def main():
    # 读取两个 CSV
    cpp_data = read_csv(cpp_csv_path)
    py_data = read_csv(python_csv_path)

    # 取交集的帧号
    common_frames = sorted(set(cpp_data.keys()) & set(py_data.keys()))

    if not common_frames:
        print("两个 CSV 没有共同的帧号，无法对比！")
        return

    # 写结果 CSV
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx",
            "time_s",
            "speed_pixel_py",
            "speed_pixel_cpp",
            "speed_diff",
            "direction_deg_py",
            "direction_deg_cpp",
            "direction_diff"
        ])

        # 读取 Python CSV 以获取 time_s（因为两个文件的 time_s 是一样的）
        with open(python_csv_path, "r", encoding="utf-8") as pyf:
            reader = csv.DictReader(pyf)
            time_map = {int(row["frame_idx"]): float(row["time_s"]) for row in reader}

        for frame_idx in common_frames:
            py_speed, py_angle = py_data[frame_idx]
            cpp_speed, cpp_angle = cpp_data[frame_idx]
            time_s = time_map[frame_idx]

            speed_diff = cpp_speed - py_speed
            angle_diff = cpp_angle - py_angle

            writer.writerow([
                frame_idx,
                round(time_s, 3),
                round(py_speed, 2),
                round(cpp_speed, 2),
                round(speed_diff, 2),
                round(py_angle, 2),
                round(cpp_angle, 2),
                round(angle_diff, 2)
            ])

    print(f"对比完成！结果已写入：{os.path.abspath(output_csv_path)}")
    print(f"对比了 {len(common_frames)} 个共同帧号")


if __name__ == "__main__":
    main()
