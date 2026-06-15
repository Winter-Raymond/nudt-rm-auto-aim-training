import cv2
import numpy as np
import os
from typing import Optional, List, Tuple

# 绿球候选目标结构体
class BallCandidate:
    def __init__(self):
        self.center = (0.0, 0.0)
        self.radius = 0.0
        self.area = 0.0
        self.circularity = 0.0
        self.aspect_ratio = 0.0
        self.bounding_box = None

# 绿球跟踪状态结构体
class BallTrack:
    def __init__(self):
        self.center = (0.0, 0.0)
        self.velocity = (0.0, 0.0)
        self.radius = 0.0
        self.speed = 0.0
        self.bounding_box = None
        self.lost_frames = 0
        self.initialized = False

# 常量定义
K_ARROW_COLOR = (255, 64, 64)  # BGR格式
K_MAX_LOST_FRAMES = 8

def is_near_frame_border(bounding_box: Tuple[int, int, int, int], frame_size: Tuple[int, int], margin: int = 8) -> bool:
    """判断边界框是否靠近帧边缘"""
    x, y, w, h = bounding_box
    frame_w, frame_h = frame_size
    return (x <= margin or y <= margin or
            x + w >= frame_w - margin or
            y + h >= frame_h - margin)

def clamp_point_to_frame(point: Tuple[float, float], frame_size: Tuple[int, int]) -> Tuple[float, float]:
    """将点限制在帧范围内"""
    x = np.clip(point[0], 0.0, float(frame_size[0] - 1))
    y = np.clip(point[1], 0.0, float(frame_size[1] - 1))
    return (x, y)

def detect_ball_candidates(frame: np.ndarray) -> List[BallCandidate]:
    """在HSV空间检测绿球候选目标"""
    # 转换为HSV颜色空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 绿色阈值范围（适配原C++参数）
    lower_green = np.array([35, 45, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    
    # 形态学操作去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    frame_size = (frame.shape[1], frame.shape[0])
    
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        if area < 7000.0 or area > 32000.0:
            continue
        
        # 计算周长
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0.0:
            continue
        
        # 计算边界框
        bounding_box = cv2.boundingRect(contour)
        x, y, w, h = bounding_box
        if h <= 0:
            continue
        
        # 计算长宽比和圆度
        aspect_ratio = float(w) / float(h)
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        near_border = is_near_frame_border(bounding_box, frame_size)
        
        # 筛选几何特征
        if not near_border:
            if circularity < 0.68 or aspect_ratio < 0.85 or aspect_ratio > 1.20:
                continue
        else:
            if circularity < 0.45 or aspect_ratio < 0.55 or aspect_ratio > 1.45:
                continue
        
        # 最小包围圆
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        
        # 创建候选目标
        candidate = BallCandidate()
        candidate.center = (center_x, center_y)
        candidate.radius = radius
        candidate.area = area
        candidate.circularity = circularity
        candidate.aspect_ratio = aspect_ratio
        candidate.bounding_box = bounding_box
        candidates.append(candidate)
    
    return candidates

def compute_shape_score(candidate: BallCandidate) -> float:
    """计算候选目标的形状分数"""
    # 圆度分数
    circularity_score = np.clip((candidate.circularity - 0.68) / 0.12, 0.0, 1.0)
    # 长宽比分数
    aspect_score = np.clip(1.0 - abs(candidate.aspect_ratio - 1.0) / 0.20, 0.0, 1.0)
    # 面积分数
    area_score = np.clip(1.0 - abs(candidate.area - 12000.0) / 10000.0, 0.0, 1.0)
    # 加权总分
    return 0.45 * circularity_score + 0.35 * aspect_score + 0.20 * area_score

def choose_best_candidate(candidates: List[BallCandidate], previous_track: BallTrack) -> Optional[BallCandidate]:
    """选择最优候选目标"""
    if not candidates:
        return None
    
    best_score = -np.inf
    best_candidate = None
    
    for candidate in candidates:
        score = compute_shape_score(candidate)
        
        if previous_track.initialized:
            # 预测位置
            pred_x = previous_track.center[0] + previous_track.velocity[0]
            pred_y = previous_track.center[1] + previous_track.velocity[1]
            # 计算距离
            distance = np.linalg.norm([
                candidate.center[0] - pred_x,
                candidate.center[1] - pred_y
            ])
            # 最大允许距离
            max_distance = max(previous_track.radius * 1.8, previous_track.speed * 3.5 + 35.0)
            
            if distance > max_distance:
                continue
            
            # 距离惩罚
            score -= distance * 0.01
            
            # 半径变化惩罚
            radius_gap = abs(candidate.radius - previous_track.radius)
            if radius_gap > previous_track.radius * 0.45:
                continue
            score -= radius_gap * 0.02
        
        # 更新最优候选
        if score > best_score:
            best_score = score
            best_candidate = candidate
    
    return best_candidate

def build_status_text(speed_px_per_sec: float, speed_delta: float) -> str:
    """生成速度状态文本"""
    text = f"speed: {speed_px_per_sec:.1f} px/s"
    if abs(speed_delta) < 1e-3:
        text += "  stable"
    elif speed_delta > 0.0:
        text += "  accelerating"
    else:
        text += "  decelerating"
    return text

def draw_track_overlay(frame: np.ndarray, current_track: BallTrack, 
                       previous_track: BallTrack, fps: float, 
                       frame_index: int, is_predicted: bool):
    """绘制跟踪标注"""
    # 绘制检测圆
    center = (int(round(current_track.center[0])), int(round(current_track.center[1])))
    radius = int(round(current_track.radius))
    cv2.circle(frame, center, radius, (0, 255, 255), 2)
    cv2.circle(frame, center, 4, (0, 0, 255), -1)
    
    # 计算速度相关参数
    speed_px_per_sec = current_track.speed * fps
    prev_speed_px_per_sec = previous_track.speed * fps if previous_track.initialized else 0.0
    speed_delta = speed_px_per_sec - prev_speed_px_per_sec
    
    # 计算箭头向量
    arrow_vector = np.array(current_track.velocity)
    if np.linalg.norm(arrow_vector) < 1.0:
        arrow_vector = np.array([30.0, 0.0])
    
    # 调整箭头长度
    arrow_scale = np.clip(current_track.speed * 4.5, 25.0, 140.0)
    velocity_norm = np.linalg.norm(arrow_vector)
    arrow_direction = arrow_vector * (arrow_scale / velocity_norm)
    arrow_end = (
        int(round(current_track.center[0] + arrow_direction[0])),
        int(round(current_track.center[1] + arrow_direction[1]))
    )
    
    # 绘制箭头
    cv2.arrowedLine(
        frame, center, arrow_end,
        K_ARROW_COLOR, 3, cv2.LINE_AA, 0, 0.28
    )
    
    # 计算文本位置
    if current_track.bounding_box:
        x, y, _, _ = current_track.bounding_box
        info_x = max(20, x - 20)
        info_y = max(40, y - 16)
    else:
        info_x, info_y = 20, 40
    
    # 绘制状态文本
    status_text = "green ball predicted" if is_predicted else "green ball"
    cv2.putText(
        frame, status_text, (info_x, info_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
    )
    
    # 绘制速度文本
    speed_text = build_status_text(speed_px_per_sec, speed_delta)
    cv2.putText(
        frame, speed_text, (info_x, info_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    
    # 绘制帧号
    cv2.putText(
        frame, f"frame: {frame_index}", (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )

def main():
    # 默认路径配置
    default_input = "first-homework/first-homework.mp4"
    default_output = "/home/taishang/桌面/作业1/nudt-rm-auto-aim-training/first-homework/week1-greenball/liujinhao/first-homework.mp4"
    
    # 获取输入输出路径
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开输入视频: {input_path}")
        return 1
    
    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0.0:
        fps = 30.0
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        output_path, fourcc, fps,
        (frame_width, frame_height)
    )
    if not writer.isOpened():
        print(f"无法创建输出视频: {output_path}")
        return 1
    
    # 初始化跟踪状态
    previous_track = BallTrack()
    frame_index = 0
    
    # 逐帧处理
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测候选目标
        candidates = detect_ball_candidates(frame)
        # 选择最优候选
        selected_candidate = choose_best_candidate(candidates, previous_track)
        
        if selected_candidate is not None:
            # 创建当前跟踪状态
            current_track = BallTrack()
            current_track.center = selected_candidate.center
            current_track.radius = selected_candidate.radius
            current_track.initialized = True
            current_track.bounding_box = selected_candidate.bounding_box
            current_track.lost_frames = 0
            
            # 计算速度
            if previous_track.initialized:
                dx = current_track.center[0] - previous_track.center[0]
                dy = current_track.center[1] - previous_track.center[1]
                current_track.velocity = (dx, dy)
                current_track.speed = np.linalg.norm([dx, dy])
            else:
                current_track.velocity = (0.0, 0.0)
                current_track.speed = 0.0
            
            # 绘制标注
            draw_track_overlay(frame, current_track, previous_track, fps, frame_index, False)
            # 更新跟踪状态
            previous_track = current_track
        
        elif previous_track.initialized and previous_track.lost_frames < K_MAX_LOST_FRAMES:
            # 预测跟踪（短时间丢失）
            predicted_track = BallTrack()
            predicted_track.center = clamp_point_to_frame(
                (previous_track.center[0] + previous_track.velocity[0],
                 previous_track.center[1] + previous_track.velocity[1]),
                (frame_width, frame_height)
            )
            predicted_track.radius = previous_track.radius
            predicted_track.velocity = previous_track.velocity
            predicted_track.speed = previous_track.speed
            predicted_track.initialized = True
            predicted_track.lost_frames = previous_track.lost_frames + 1
            # 计算预测边界框
            predicted_track.bounding_box = (
                int(round(predicted_track.center[0] - predicted_track.radius)),
                int(round(predicted_track.center[1] - predicted_track.radius)),
                int(round(predicted_track.radius * 2.0)),
                int(round(predicted_track.radius * 2.0))
            )
            
            # 绘制预测标注
            draw_track_overlay(frame, predicted_track, previous_track, fps, frame_index, True)
            # 更新跟踪状态
            previous_track = predicted_track
        
        else:
            # 目标丢失
            cv2.putText(
                frame, "green ball lost", (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
            )
            previous_track = BallTrack()
        
        # 写入帧
        writer.write(frame)
        frame_index += 1
    
    # 释放资源
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    print(f"标注视频已保存至: {output_path}")
    return 0

if __name__ == "__main__":
    main()
