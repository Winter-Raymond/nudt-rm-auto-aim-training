#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import math

# 下限
NOISE_AREA = 20           
CIRC_THRESH = 0.70        
SOLIDITY_THRESH = 0.85    
ELLIP_RATIO_THRESH = 0.80 

# 箭头
SMOOTH_ALPHA = 0.3
SPEED_SCALE = 5.0        
MIN_ARROW_LEN = 15       
MAX_ARROW_LEN = 100
TRAJECTORY_LEN = 20

def main():

    cap = cv2.VideoCapture(sys.argv[1])

    prev_center = None
    pause = False
    frame_count = 0
    wait_time = 30
    auto_mode = True
    trajectory = []
    smooth_dx, smooth_dy = 0.0, 0.0

    cv2.namedWindow('Ball Tracking', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)

    # HSV 滑块
    cv2.createTrackbar('H Min', 'Mask', 40, 180, lambda x: None)
    cv2.createTrackbar('H Max', 'Mask', 85, 180, lambda x: None)
    cv2.createTrackbar('S Min', 'Mask', 50, 255, lambda x: None)
    cv2.createTrackbar('S Max', 'Mask', 255, 255, lambda x: None)
    cv2.createTrackbar('V Min', 'Mask', 50, 255, lambda x: None)
    cv2.createTrackbar('V Max', 'Mask', 255, 255, lambda x: None)

    print(" 空格键 - 暂停/继续")
    print(" n键    - 逐帧播放")
    print(" q键    - 退出")

    while True:
        if not pause:
            if auto_mode:
                ret, frame = cap.read()
                if not ret:
                    print("视频播放完毕")
                    break
                frame_count += 1

        if 'frame' not in locals():
            continue

        #  HSV 
        h_min = cv2.getTrackbarPos('H Min', 'Mask')
        h_max = cv2.getTrackbarPos('H Max', 'Mask')
        s_min = cv2.getTrackbarPos('S Min', 'Mask')
        s_max = cv2.getTrackbarPos('S Max', 'Mask')
        v_min = cv2.getTrackbarPos('V Min', 'Mask')
        v_max = cv2.getTrackbarPos('V Max', 'Mask')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)

        #  形态学去噪
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_ball = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < NOISE_AREA:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circ = 4 * math.pi * area / (perimeter * perimeter)

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area

            ellipse_ratio = 1.0
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (major, minor) = ellipse[1]
                if min(major, minor) > 0:
                    ellipse_ratio = min(major, minor) / max(major, minor)

            if (circ > CIRC_THRESH and solidity > SOLIDITY_THRESH and
                 ellipse_ratio > ELLIP_RATIO_THRESH):
                score = area * circ * solidity * ellipse_ratio
                if score > best_score:
                    best_score = score
                    best_ball = cnt

        if best_ball is not None:
            M = cv2.moments(best_ball)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                curr_center = (cx, cy)

                cv2.circle(frame, curr_center, 10, (0, 255, 255), 2)
                cv2.circle(frame, curr_center, 6, (0, 0, 255), -1)

                if prev_center is not None:
                    raw_dx = cx - prev_center[0]
                    raw_dy = cy - prev_center[1]
                    smooth_dx = SMOOTH_ALPHA * raw_dx + (1 - SMOOTH_ALPHA) * smooth_dx
                    smooth_dy = SMOOTH_ALPHA * raw_dy + (1 - SMOOTH_ALPHA) * smooth_dy

                    speed = math.hypot(smooth_dx, smooth_dy)
                    if speed > 0.5:
                        arrow_len = speed * SPEED_SCALE
                        if arrow_len < MIN_ARROW_LEN:
                            arrow_len = MIN_ARROW_LEN
                        if arrow_len > MAX_ARROW_LEN:
                            arrow_len = MAX_ARROW_LEN

                        angle = math.atan2(smooth_dy, smooth_dx)
                        end_x = int(cx + arrow_len * math.cos(angle))
                        end_y = int(cy + arrow_len * math.sin(angle))

                        cv2.arrowedLine(frame, curr_center, (end_x, end_y),
                                        (255, 0, 255), 3, tipLength=0.3)
                        cv2.putText(frame, f"v={speed:.1f}", (cx+15, cy-15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    smooth_dx, smooth_dy = 0.0, 0.0

                # 轨迹更新（忽略微小位移，防止静止时堆积）
                if len(trajectory) == 0 or math.hypot(cx - trajectory[-1][0], cy - trajectory[-1][1]) > 2:
                    trajectory.append(curr_center)
                if len(trajectory) > TRAJECTORY_LEN:
                    trajectory.pop(0)

                prev_center = curr_center

        else:
            prev_center = None
            trajectory.clear()
            smooth_dx, smooth_dy = 0.0, 0.0

        # 渐变色
        for i in range(1, len(trajectory)):
            alpha = i / len(trajectory)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            cv2.line(frame, trajectory[i-1], trajectory[i], color, 2)


        cv2.imshow('Ball Tracking', frame)
        cv2.imshow('Mask', mask)

        # 键盘交互 
        if auto_mode:
            key = cv2.waitKey(wait_time) & 0xFF
        else:
            key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            pause = not pause
        elif key == ord('n'):
            if pause or not auto_mode:
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                pause = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
