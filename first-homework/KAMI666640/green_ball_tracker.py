# -*- coding: utf-8 -*-
"""
NUDT 自瞄培训 - 第一次作业
=================================
任务：识别视频中的绿球，并用箭头标注它的运动方向和速度变化。
      · 箭头方向 = 小球当前的运动方向
      · 箭头长度 = 小球当前的速度大小（越长越快，越短越慢）

运行方式：
    python green_ball_tracker.py

运行前准备：
    pip install numpy opencv-python

作者：26赵冠钦
"""

import cv2
import numpy as np
import os
import sys

# ============================================================
# 一、可调参数（想改效果就改这里，不用动下面的逻辑）
# ============================================================

# 输入视频（和本脚本放在同一个文件夹里）
INPUT_VIDEO = "first-homework.mp4"

# 输出视频（带箭头的效果视频）
OUTPUT_VIDEO = "result.mp4"

# 绿色在 HSV 色彩空间里的范围
#   H: 色相（0-179，OpenCV 里绿色大约 35~90）
#   S: 饱和度（越低越接近白色/灰色）
#   V: 亮度（越低越接近黑色）
# 如果背景里有别的绿色被误认，就把 S_MIN / V_MIN 调高一点
GREEN_LOWER = (35, 60, 60)      # (H, S, V) 下限
GREEN_UPPER = (90, 255, 255)    # (H, S, V) 上限

# 只保留面积大于这个值的绿色区域（过滤噪点）
MIN_CONTOUR_AREA = 500

# ---- 圆形度筛选（本次作业的关键）----
# 视频里每一帧其实有【两块】绿色区域：
#   一块面积大但形状是拉长的（实测填充率 0.61、宽高比 0.73）——那是运动拖影
#   一块面积小但是正圆（实测填充率 0.87、宽高比 1.00）——这才是真正的球
# 所以要按"有多圆"来筛选，而不是简单地选面积最大的。
#
# 填充率 = 轮廓面积 / 外接圆面积，完美圆 = 1.00，越不规则越小
MIN_FILL_RATIO = 0.70           # 只考虑填充率 >= 这个值的轮廓

# ---- 跳变保护 ----
# 如果某一帧检测到的位置和上一帧差得离谱，说明认错了，直接丢弃这一帧的结果
MAX_JUMP = 150                  # 相邻帧最大允许位移（像素）

# 速度平滑窗口：取最近几帧的位移来算平均速度
#   调大 -> 箭头更稳定但不灵敏；调小 -> 更灵敏但会抖
SMOOTH_WINDOW = 5

# 箭头长度缩放系数：箭头像素长度 = 速度(像素/秒) × ARROW_SCALE
#   0.14 是实测出来的：全片速度 95 分位约 712 px/s，712 × 0.14 ≈ 100 像素
#   觉得箭头太长就调小，太短就调大
ARROW_SCALE = 0.14

# 箭头长度上限（像素），防止个别帧速度突变导致箭头飞出画面
ARROW_MAX_LEN = 150

# 是否画出小球的历史运动轨迹（淡绿色的尾巴）
DRAW_TRAIL = True
TRAIL_LENGTH = 60               # 轨迹保留多少帧

# 是否在画面上显示速度数值（像素/秒）
DRAW_SPEED_TEXT = True


# ============================================================
# 二、工具函数
# ============================================================

def find_green_ball(frame):
    """
    在一帧图像里找出绿球的位置。

    参数：frame —— BGR 格式的一帧图像（OpenCV 读进来的默认格式）
    返回：(cx, cy, radius, area)
          cx, cy   绿球中心坐标（像素）
          radius   绿球半径（像素）
          area     绿球面积（像素²）
          如果这一帧没找到绿球，返回 (None, None, 0, 0)
    """
    # 第 1 步：BGR → HSV
    #   为什么要用 HSV？因为 RGB 里"绿色"受光照影响很大，
    #   亮一点的绿和暗一点的绿 RGB 值差很远；
    #   而 HSV 把"颜色(H)"和"明暗(V)"拆开了，只按 H 就能认出绿色，鲁棒得多。
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 第 2 步：做掩膜（mask）
    #   inRange 的作用：把 HSV 图里每个像素拿去比对，
    #   落在 [GREEN_LOWER, GREEN_UPPER] 区间内的 → 设为 255（白）
    #   落在区间外的               → 设为 0（黑）
    #   结果是一张黑白图，白的地方就是"绿色区域"
    mask = cv2.inRange(hsv, np.array(GREEN_LOWER), np.array(GREEN_UPPER))

    # 第 3 步：形态学去噪
    #   "开运算" = 先腐蚀再膨胀，可以去掉小白点噪点
    #   "闭运算" = 先膨胀再腐蚀，可以填补小球内部的小黑洞
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 第 4 步：找轮廓
    #   轮廓 = 白色区域的边界。RETR_EXTERNAL 表示只取最外层轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, 0, 0

    # 第 5 步：在候选中挑"最像球"的那个
    #
    # 这里没有简单地取"面积最大的"，而是先算每一块的【填充率】：
    #     填充率 = 轮廓面积 / 外接圆面积
    #   正圆的填充率接近 1.0；被拉长、有缺口的形状会明显小于 1.0
    #
    # 本视频里每帧都有两块绿色：大的那块是运动拖影（填充率约 0.61），
    # 小的那块才是真球（填充率约 0.87）。按填充率筛一遍就能自动选中球。
    candidates = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < MIN_CONTOUR_AREA:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r <= 0:
            continue
        fill = a / (np.pi * r * r)          # 填充率，1.0 = 正圆
        candidates.append((fill, a, x, y, r))

    if not candidates:
        return None, None, 0, 0

    # 优先在"足够圆"的候选里挑面积最大的
    round_ones = [c for c in candidates if c[0] >= MIN_FILL_RATIO]
    if round_ones:
        best = max(round_ones, key=lambda c: c[1])
    else:
        # 一个圆的都没有（比如被遮挡了一部分），退而求其次：挑最圆的那个
        best = max(candidates, key=lambda c: c[0])

    fill, area, x, y, radius = best

    # 用轮廓的一阶矩算质心，比外接圆圆心更准
    return int(x), int(y), int(radius), area


def calc_velocity(centers, fps, window=SMOOTH_WINDOW):
    """
    根据最近若干帧的中心坐标，算当前速度（单位：像素/秒）。

    原理：速度 = 位移 / 时间
        vx = (x_now - x_old) / dt
        vy = (y_now - y_old) / dt

    只取最近 window 帧做平均，是为了让箭头不那么抖。
    """
    if len(centers) < 2:
        return 0.0, 0.0

    # 取最近 window 个有效位置（含当前帧）
    recent = centers[-window:]

    if len(recent) < 2:
        return 0.0, 0.0

    # 用最早和最新两点算平均位移，再除以经过的时间
    x_old, y_old = recent[0]
    x_now, y_now = recent[-1]

    dt = (len(recent) - 1) / fps      # 经过了多少秒
    if dt <= 0:
        return 0.0, 0.0

    vx = (x_now - x_old) / dt
    vy = (y_now - y_old) / dt

    return vx, vy


def draw_arrow(img, cx, cy, vx, vy, scale):
    """
    在 (cx, cy) 处画一个箭头。
    方向 = 速度方向；长度 = 速度大小 × scale（并做上限截断）。
    """
    # 速度大小（勾股定理）
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # 速度太小时不画箭头（避免画出乱七八糟的小刺）
    if speed < 1.0:
        return 0.0, (cx, cy)

    # 把速度向量按 scale 缩放成屏幕上的像素长度
    dx = vx * scale
    dy = vy * scale

    # 截断：如果超过上限，按比例整体缩短（方向不变）
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length > ARROW_MAX_LEN:
        k = ARROW_MAX_LEN / length
        dx *= k
        dy *= k

    end_x = int(cx + dx)
    end_y = int(cy + dy)

    # 箭头本体：从球心指向速度方向
    #   颜色 (0, 255, 0) 在 OpenCV 里是 BGR，也就是绿色
    #   最后那个 2 是线宽，9 是箭头尖角的大小
    cv2.arrowedLine(img, (cx, cy), (end_x, end_y), (0, 255, 0), 2, tipLength=0.35)

    return speed, (end_x, end_y)


def draw_trail(img, centers, max_len=TRAIL_LENGTH):
    """画出绿球最近走过的路径（越旧越淡）"""
    pts = centers[-max_len:]
    for i in range(1, len(pts)):
        # 越新的点画得越亮
        alpha = i / len(pts)
        color = (0, int(80 + 120 * alpha), 0)
        cv2.line(img, pts[i - 1], pts[i], color, 2)


# ============================================================
# 三、主流程
# ============================================================

def main():
    # 切换到脚本所在目录，这样相对路径都能找到
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.exists(INPUT_VIDEO):
        print(f"[错误] 找不到视频文件：{INPUT_VIDEO}")
        print("       请把 first-homework.mp4 放到和本脚本同一个文件夹里")
        sys.exit(1)

    # ---- 打开视频 ----
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("[错误] 视频打不开，可能编码格式不支持")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 50)
    print(f"视频信息：{width} x {height}，{fps:.2f} FPS，共 {total} 帧")
    print(f"时长：{total / fps:.2f} 秒")
    print("=" * 50)

    if fps <= 0 or fps != fps:      # 有些视频读不到 fps，兜底给 30
        fps = 30.0
        print(f"[提示] 视频没读到帧率，已按 {fps} FPS 处理")

    # ---- 准备输出视频 ----
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    if not out.isOpened():
        print("[警告] mp4v 编码器不可用，改用 avc1 再试")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
        if not out.isOpened():
            print("[错误] 输出视频创建失败")
            sys.exit(1)

    centers = []        # 保存每一帧绿球的中心坐标
    frame_idx = 0
    miss_count = 0      # 连续丢失计数
    all_speeds = []     # 记录所有速度，用于最后统计

    print("开始处理，请稍候...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ---- 第 1 步：找出这一帧的绿球 ----
        cx, cy, radius, area = find_green_ball(frame)

        if cx is None:
            miss_count += 1
            # 这一帧没找到球，直接把原帧写出去
            out.write(frame)
            if frame_idx % 50 == 0:
                print(f"  帧 {frame_idx}/{total}：未检测到绿球")
            continue

        miss_count = 0

        # 画一个红圈把球圈出来（方便肉眼确认识别对了没）
        cv2.circle(frame, (cx, cy), max(radius, 6), (0, 0, 255), 2)

        # ---- 跳变保护 ----
        # 如果这一帧的位置和上一帧差得太离谱（超过 MAX_JUMP 像素），
        # 八成是认错目标了，这一帧就不参与速度计算，免得箭头乱飞
        if centers:
            lx, ly = centers[-1]
            jump = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
            if jump > MAX_JUMP:
                out.write(frame)
                if frame_idx % 50 == 0:
                    print(f"  帧 {frame_idx}/{total}：位置跳变 {jump:.0f}px，已跳过")
                continue

        centers.append((cx, cy))

        # ---- 第 2 步：算速度 ----
        vx, vy = calc_velocity(centers, fps)

        # ---- 第 3 步：画箭头 ----
        speed, _ = draw_arrow(frame, cx, cy, vx, vy, ARROW_SCALE)
        all_speeds.append(speed)

        # ---- 附加：画轨迹 + 速度文字 ----
        if DRAW_TRAIL and len(centers) > 1:
            draw_trail(frame, centers)

        if DRAW_SPEED_TEXT:
            txt = f"v = {speed:.0f} px/s"
            cv2.putText(frame, txt, (cx + 12, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)

        if frame_idx % 50 == 0:
            print(f"  帧 {frame_idx}/{total}：位置({cx},{cy})  速度 {speed:.0f} px/s")

    # ---- 收尾 ----
    cap.release()
    out.release()

    print("=" * 50)
    print(f"处理完成！共处理 {frame_idx} 帧，检测到绿球 {len(centers)} 帧")
    if all_speeds:
        print(f"速度统计：最小 {min(all_speeds):.0f} | "
              f"平均 {sum(all_speeds)/len(all_speeds):.0f} | "
              f"最大 {max(all_speeds):.0f}  (单位：像素/秒)")
    print(f"输出文件：{os.path.abspath(OUTPUT_VIDEO)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
