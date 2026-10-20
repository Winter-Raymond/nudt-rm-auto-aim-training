# -*- coding: utf-8 -*-
"""
绿球识别作业 —— 识别视频里的绿球，用箭头标注它的【运动方向】和【速度】。

规则：
  - 箭头方向 = 球的运动方向
  - 箭头越长 = 速度越快（越短越慢）

总体思路（建议按顺序读，每一步都有注释）：
  1. 读视频，逐帧取出画面
  2. 用 HSV 颜色分割，把"绿色"的部分挑出来
  3. 在绿色里区分【绿球】和【绿色圆柱】——这是本题的关键（见 find_ball 的注释）
  4. 记录每一帧球心的位置 (x, y)
  5. 由位置变化算出速度和方向（先平滑去抖，再求导）
  6. 把速度映射成箭头长度，画箭头、画球、写速度数值
  7. 写回视频 output.mp4

只依赖 opencv 和 numpy，无 GPU、无深度学习。
"""

import cv2
import numpy as np

# ================= 参数区（想调就改这里） =================
INPUT  = "first-homework.mp4"   # 输入视频文件名
OUTPUT = "output.mp4"           # 输出视频文件名

GREEN_LOW  = (35, 60, 60)       # HSV 绿色的下界 (H, S, V)
GREEN_HIGH = (85, 255, 255)     # HSV 绿色的上界

MIN_AREA = 500                  # 面积小于这个值的色块当成噪点忽略（单位：像素²）
FILL_MIN = 0.73                 # "实心圆占比"下限：大于它才认为是球（圆柱低于它）

SMOOTH_WIN = 5                  # 平滑窗口：取前后几帧平均，消除抖动
ARROW_MIN = 25                  # 箭头最短像素
ARROW_MAX = 160                 # 箭头最长像素
SPEED_PERCENT = 95              # 用速度的 95% 分位作为"箭头最长"的标准（避免异常值）

# ========================================================


def find_ball(frame):
    """
    在一帧画面里找到绿球，返回 (x, y, r)（圆心和半径）；找不到返回 None。

    【本题的关键】画面里有两个绿色物体：
      - 绿球：目标，接近一个实心圆
      - 绿色圆柱：干扰物，绿色、更大、但不是圆（是个柱体/矩形）
    如果"取最大的绿色轮廓"，抓到的一定是圆柱（它更大）。
    正确做法是用【形状】区分：球是实心圆，圆柱不是。

    判断"像不像实心圆"用这个比例：
        实心圆占比 = 轮廓面积 / 最小外接圆面积
      - 球：0.75 ~ 0.96（接近 1）
      - 圆柱：0.18 ~ 0.73（明显小）
    所以取"实心圆占比最高"的轮廓，就一定是球。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1) 颜色分割：只保留绿色像素（白色=绿，黑色=不是绿）
    mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)

    # 2) 形态学去噪：开运算去掉孤立噪点，闭运算填补球内的反光空洞
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # 3) 提取轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4) 在绿色轮廓里，挑"实心圆占比最高"的那个（= 球）
    best = None
    best_fill = -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)      # 最小外接圆
        fill = area / (np.pi * r * r)              # 实心圆占比
        if fill > best_fill:
            best_fill = fill
            best = (x, y, r)

    if best is not None and best_fill > FILL_MIN:
        return best
    return None


def detect_all(cap, total):
    """逐帧检测球心，返回列表，每帧是 (x, y, r) 或 None（没找到）。"""
    centers = []
    for _ in range(total):
        ok, frame = cap.read()
        if not ok:
            centers.append(None)
            continue
        centers.append(find_ball(frame))
    return centers


def fill_missing(centers):
    """球在某几帧没找到(丢失)时，用前后已知位置线性插值补上，避免轨迹断掉。"""
    idx = [i for i, c in enumerate(centers) if c is not None]
    if len(idx) == 0:
        return centers
    xs = [centers[i][0] for i in idx]
    ys = [centers[i][1] for i in idx]
    rs = [centers[i][2] for i in idx]
    all_i = np.arange(len(centers))
    fx = np.interp(all_i, idx, xs)
    fy = np.interp(all_i, idx, ys)
    fr = np.interp(all_i, idx, rs)
    return list(zip(fx, fy, fr))


def moving_average(vals, win):
    """
    滑动平均：对最近 win 帧取平均，去掉检测噪声（否则箭头会疯狂抖动）。
    注意：两端用"重复边界值"补齐，而不是补 0。
    （np.convolve 默认补 0 会把首尾几帧的值拉低，造成速度在末尾爆掉。）
    """
    pad = win // 2
    vals = np.asarray(vals, dtype=float)
    xp = np.concatenate([np.full(pad, vals[0]), vals, np.full(pad, vals[-1])])
    k = np.ones(win) / win
    return np.convolve(xp, k, mode="valid")


def compute_velocity(centers, fps):
    """
    由球心位置序列算出每一帧的速度 (vx, vy, speed)，单位像素/秒。
    先平滑（取前后几帧平均，去掉检测噪声），再用"中心差分"求速度。
    """
    xs = np.array([c[0] for c in centers])
    ys = np.array([c[1] for c in centers])

    # 平滑：最近 SMOOTH_WIN 帧取平均
    sx = moving_average(xs, SMOOTH_WIN)
    sy = moving_average(ys, SMOOTH_WIN)

    dt = 1.0 / fps
    vx = np.zeros_like(sx)
    vy = np.zeros_like(sy)
    # 中间帧：中心差分 (x[i+1]-x[i-1]) / (2*dt)，无滞后
    vx[1:-1] = (sx[2:] - sx[:-2]) / (2.0 * dt)
    vy[1:-1] = (sy[2:] - sy[:-2]) / (2.0 * dt)
    # 首尾帧：单边差分
    vx[0] = (sx[1] - sx[0]) / dt
    vy[0] = (sy[1] - sy[0]) / dt
    vx[-1] = (sx[-1] - sx[-2]) / dt
    vy[-1] = (sy[-1] - sy[-2]) / dt

    speed = np.hypot(vx, vy)
    return vx, vy, speed


def speed_to_length(speed, speed_max):
    """速度 -> 箭头长度（像素）。越快越长，但限制在 [ARROW_MIN, ARROW_MAX]。"""
    ratio = float(np.clip(speed / speed_max, 0.0, 1.0))   # 0~1
    return ARROW_MIN + ratio * (ARROW_MAX - ARROW_MIN)


def draw_overlay(frame, x, y, r, vx, vy, speed, speed_max):
    """在一帧上画出：球（青色圆环）、方向箭头、速度数值。"""
    theta = float(np.arctan2(vy, vx))          # 方向（图像 y 轴向下，这里已正确处理）
    L = speed_to_length(speed, speed_max)

    # 球：青色圆环 + 中心点，表示"检测到了球"
    cv2.circle(frame, (int(x), int(y)), int(r), (255, 255, 0), 2)
    cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), -1)

    # 箭头：从球边缘起画（否则低速短箭头会埋在球里看不见），越长越快
    x0 = x + r * np.cos(theta)
    y0 = y + r * np.sin(theta)
    x1 = x0 + L * np.cos(theta)
    y1 = y0 + L * np.sin(theta)
    cv2.arrowedLine(frame, (int(x0), int(y0)), (int(x1), int(y1)),
                    (0, 0, 255), 3, tipLength=0.3)

    # 速度数值
    cv2.putText(frame, f"v={speed:.0f} px/s", (int(x) + int(r) + 8, int(y) - int(r) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def main():
    cap = cv2.VideoCapture(INPUT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频: {w}x{h}, {fps:.1f} fps, {total} 帧")

    # ---- 第一遍：逐帧检测球心 ----
    print("第一遍：检测球心...")
    centers = detect_all(cap, total)
    cap.release()

    found = sum(1 for c in centers if c is not None)
    print(f"  检测到 {found}/{total} 帧（{found/total*100:.1f}%）")

    # ---- 补丢失帧 + 平滑 + 求速度 ----
    centers = fill_missing(centers)
    vx, vy, speed = compute_velocity(centers, fps)

    # 用速度的 95% 分位作为"箭头最长"的标准（分位比 max 稳，不被异常值带偏）
    speed_max = float(np.percentile(speed, SPEED_PERCENT))
    print(f"  速度范围: 最小 {speed.min():.0f} / 中位 {np.median(speed):.0f} / 最大 {speed.max():.0f} px/s")
    print(f"  箭头映射: {speed_max:.0f} px/s -> 最长箭头 {ARROW_MAX} px")

    # ---- 第二遍：重读视频，画箭头并写输出 ----
    print("第二遍：画箭头、写视频...")
    cap = cv2.VideoCapture(INPUT)
    writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        x, y, r = centers[i]
        draw_overlay(frame, x, y, r, vx[i], vy[i], speed[i], speed_max)
        # 左上角图例
        cv2.putText(frame, "arrow direction = motion  |  arrow length = speed",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"完成，输出 {OUTPUT}（{total} 帧）")


if __name__ == "__main__":
    main()
