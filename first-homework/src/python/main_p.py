import cv2
import numpy as np
import os
import json
import csv
import math

# ====================== 配置读取 ======================
def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)

    cfg = {}
    # 路径
    cfg["input_video"] = cfg_data["path"]["input_video"]
    cfg["output_video"] = cfg_data["path"]["output_video"]
    cfg["output_data"] = cfg_data["path"]["output_data"]
    # 相机参数
    cfg["camera_matrix"] = np.array(cfg_data["camera"]["camera_matrix"], dtype=np.float32)
    cfg["dist_coeffs"] = np.array(cfg_data["camera"]["dist_coeffs"], dtype=np.float32)
    cfg["ball_radius"] = cfg_data["camera"]["ball_radius"]
    # 颜色检测
    cfg["lower_green"] = np.array(cfg_data["color_detect"]["lower_green"], dtype=np.uint8)
    cfg["upper_green"] = np.array(cfg_data["color_detect"]["upper_green"], dtype=np.uint8)
    cfg["morph_kernel_size"] = cfg_data["color_detect"]["morph_kernel_size"]
    cfg["min_contour_area"] = cfg_data["color_detect"]["min_contour_area"]
    # 形状筛选
    cfg["min_center_std"] = cfg_data["shape_filter"]["min_center_std"]
    cfg["axis_ratio_weight"] = cfg_data["shape_filter"]["axis_ratio_weight"]
    cfg["center_bias_weight"] = cfg_data["shape_filter"]["center_bias_weight"]
    # 卡尔曼
    cfg["kf_process_noise"] = cfg_data["kalman"]["process_noise"]
    cfg["kf_measure_noise"] = cfg_data["kalman"]["measurement_noise"]
    cfg["max_lost_frame"] = cfg_data["kalman"]["max_lost_frame"]
    # 可视化
    cfg["speed_scale"] = cfg_data["visual"]["speed_scale"]
    cfg["show_track_line"] = cfg_data["visual"]["show_track_line"]
    cfg["track_line_length"] = cfg_data["visual"]["track_line_length"]
    # 有效区域
    cfg["margin_w_ratio"] = cfg_data["roi_margin"]["margin_w_ratio"]
    cfg["margin_h_ratio"] = cfg_data["roi_margin"]["margin_h_ratio"]

    return cfg

# ====================== 卡尔曼滤波 ======================
class KalmanFilter2D:
    def __init__(self, dt, process_noise, measure_noise):
        self.dt = dt
        self.state = np.zeros((4, 1), dtype=np.float32)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.R = np.eye(2, dtype=np.float32) * measure_noise
        self.P = np.eye(4, dtype=np.float32) * 1000

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].flatten()

    def update(self, pt):
        z = np.array(pt, dtype=np.float32).reshape(2, 1)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[:2].flatten()

# ====================== 工具函数 ======================
def calc_contour_center_stdev(cnt, center):
    cx, cy = center
    distances = []
    for p in cnt:
        x, y = p[0]
        d = np.sqrt((x - cx)**2 + (y - cy)**2)
        distances.append(d)
    return np.std(distances)

def calc_speed_and_dir(prev_pt, curr_pt):
    """
    计算像素速度大小 + 运动方向角(°)
    :return: speed, angle_deg
    """
    if prev_pt is None or curr_pt is None:
        return 0.0, 0.0
    dx = curr_pt[0] - prev_pt[0]
    dy = curr_pt[1] - prev_pt[1]
    speed = math.hypot(dx, dy)
    # 计算弧度并转为角度，标准数学坐标系
    rad = math.atan2(dy, dx)
    angle = math.degrees(rad)
    # 转为 0~360°
    if angle < 0:
        angle += 360.0
    return round(speed, 2), round(angle, 2)

# ====================== 目标检测 ======================
def auto_filter_green_objects(frame, cfg):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, cfg["lower_green"], cfg["upper_green"])

    kernel = np.ones((cfg["morph_kernel_size"], cfg["morph_kernel_size"]), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_ball = None
    best_score = -1.0

    margin_w = int(w * cfg["margin_w_ratio"])
    margin_h = int(h * cfg["margin_h_ratio"])
    center_x, center_y = w // 2, h // 2

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg["min_contour_area"]:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if (cx < margin_w or cx > w - margin_w or
            cy < margin_h or cy > h - margin_h):
            continue

        try:
            ellipse = cv2.fitEllipse(cnt)
            (_, _), (major, minor), _ = ellipse
            if minor < 1e-3:
                continue
            axis_ratio = major / minor
        except:
            continue

        stdev = calc_contour_center_stdev(cnt, (cx, cy))
        if stdev > cfg["min_center_std"]:
            continue

        dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        pos_weight = 1.0 / (1.0 + dist_to_center / 100)
        shape_weight = 1.0 / (1.0 + abs(axis_ratio - 1.0))

        score = (shape_weight * cfg["axis_ratio_weight"] + pos_weight * cfg["center_bias_weight"]) * np.sqrt(area)

        if score > best_score:
            best_score = score
            best_ball = (cnt, (cx, cy), ellipse)

    ball_center = None
    best_cnt = None
    if best_ball is not None:
        best_cnt, ball_center, ell = best_ball
        cv2.ellipse(frame, ell, (0, 255, 0), 2)
        cv2.circle(frame, ball_center, 8, (0, 0, 255), -1)

    return frame, ball_center, best_cnt

# ====================== PnP 三维解算 ======================
def pnp_3d_estimate(center_2d, cnt, cfg):
    if center_2d is None or cnt is None:
        return None
    obj_pts = np.array([
            [-cfg["ball_radius"], 0, 0],
            [cfg["ball_radius"], 0, 0],
            [0, -cfg["ball_radius"], 0],
            [0, cfg["ball_radius"], 0],
            [0, 0, -cfg["ball_radius"]],
            [0, 0, cfg["ball_radius"]]
    ], dtype=np.float32)
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    img_pts = np.array([
        [cx - r, cy], [cx + r, cy],
        [cx, cy - r], [cx, cy + r],
        [cx, cy], [cx, cy]
    ], dtype=np.float32)
    ret, _, tvec = cv2.solvePnP(obj_pts, img_pts, cfg["camera_matrix"], cfg["dist_coeffs"])
    if ret:
        return tvec.flatten()
    return None

# ====================== 绘制运动信息 ======================
def draw_motion(frame, curr_pt, prev_pt, kf_pt, speed_scale):
    if curr_pt is None:
        return frame
    cv2.circle(frame, (int(kf_pt[0]), int(kf_pt[1])), 6, (255, 0, 0), 2)
    if prev_pt is not None:
        dx = curr_pt[0] - prev_pt[0]
        dy = curr_pt[1] - prev_pt[1]
        speed = math.hypot(dx, dy)
        end_x = int(curr_pt[0] + dx * speed_scale * 3)
        end_y = int(curr_pt[1] + dy * speed_scale * 3)
        cv2.arrowedLine(frame, curr_pt, (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)
        cv2.putText(frame, f"Speed:{speed:.1f}",
                    (curr_pt[0]+10, curr_pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    return frame

# ====================== 主函数 ======================
def main():
    cfg = load_config()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 创建视频输出目录
    video_dir = os.path.dirname(cfg["output_video"])
    os.makedirs(video_dir, exist_ok=True)

    # 2. 创建数据输出目录 & CSV 文件
    data_dir = os.path.dirname(cfg["output_data"])
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "speed.csv")

    cap = cv2.VideoCapture(cfg["input_video"])
    if not cap.isOpened():
        print("视频打开失败，请检查路径配置")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cfg["output_video"], fourcc, fps, (w, h))

    kf = KalmanFilter2D(1.0/fps, cfg["kf_process_noise"], cfg["kf_measure_noise"])
    prev_center = None
    track_points = []
    lost_frames = 0
    frame_idx = 0

    # 初始化 CSV 并写入表头
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "time_s", "speed_pixel", "direction_deg"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 当前帧绝对时间
        current_time = frame_idx / fps

        # 目标检测
        frame, curr_center, cnt = auto_filter_green_objects(frame, cfg)

        # PnP 解算
        tvec = pnp_3d_estimate(curr_center, cnt, cfg)
        if tvec is not None:
            cv2.putText(frame, f"3D: {tvec[0]:.2f},{tvec[1]:.2f},{tvec[2]:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # 计算速度、方向
        speed, angle = calc_speed_and_dir(prev_center, curr_center)

        # 卡尔曼更新
        pred_pt = kf.predict()
        if curr_center is not None:
            kf_pt = kf.update(curr_center)
            track_points.append((int(kf_pt[0]), int(kf_pt[1])))
            if len(track_points) > cfg["track_line_length"]:
                track_points.pop(0)
            lost_frames = 0
        else:
            lost_frames += 1
            if lost_frames > cfg["max_lost_frame"]:
                kf = KalmanFilter2D(1.0/fps, cfg["kf_process_noise"], cfg["kf_measure_noise"])
                track_points.clear()
                lost_frames = 0
            kf_pt = pred_pt

        # 绘制轨迹
        if cfg["show_track_line"] and len(track_points) > 1:
            for i in range(1, len(track_points)):
                cv2.line(frame, track_points[i-1], track_points[i], (255,0,255), 2)

        frame = draw_motion(frame, curr_center, prev_center, kf_pt, cfg["speed_scale"])

        # 追加写入当前帧数据到 CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([frame_idx, round(current_time, 3), speed, angle])

        # 更新状态
        prev_center = curr_center
        out.write(frame)
        cv2.imshow("Ball Track", frame)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成！")
    print(f"结果视频: {cfg['output_video']}")
    print(f"帧速度数据: {csv_path}")

if __name__ == "__main__":
    main()
