import cv2
import json
import os
import numpy as np
import time

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def detect_ball(frame, cfg):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(cfg["color_detect"]["lower_green"]),
                       np.array(cfg["color_detect"]["upper_green"]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg["color_detect"]["morph_kernel_size"],)*2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    margin_w = w * cfg["roi_margin"]["margin_w_ratio"]
    margin_h = h * cfg["roi_margin"]["margin_h_ratio"]
    img_center = (w/2, h/2)
    best_score = -1
    best_pt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg["color_detect"]["min_contour_area"]:
            continue
        # 1. 圆度筛选：过滤掉圆柱等非球形物体
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        # 圆度低于 0.7 的直接过滤（圆柱的圆度会远低于 1，球接近 1）
        if circularity < 0.7:
            continue
        # 2. 椭圆轴比筛选：和之前一样，进一步过滤非球形
        try:
            ellipse = cv2.fitEllipse(cnt)
        except:
            continue
        axis_ratio = max(ellipse[1]) / min(ellipse[1])
        if axis_ratio > 1.3:  # 轴比过大的也过滤掉
            continue
        # 3. 位置打分：优先选离画面中心近的轮廓
        cx, cy = cv2.moments(cnt)["m10"]/cv2.moments(cnt)["m00"], cv2.moments(cnt)["m01"]/cv2.moments(cnt)["m00"]
        if cx < margin_w or cx > w - margin_w or cy < margin_h or cy > h - margin_h:
            continue
        dist = np.hypot(cx - img_center[0], cy - img_center[1])
        # 综合打分：圆度权重 + 轴比权重 + 位置权重
        score = (circularity * 0.5) + (1/(1+abs(axis_ratio-1)) * 0.3) + (1/(1+dist/100) * 0.2)
        if score > best_score:
            best_score = score
            best_pt = (cx, cy)
    return best_pt

def main():
    cfg = load_config("config.json")
    cap = cv2.VideoCapture(cfg["path"]["input_video"])
    if not cap.isOpened():
        print("Open video failed")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    tmp_file = "detector.tmp"
    ready_file = "detector.ready"
    # 清理旧文件
    for f in [tmp_file, ready_file]:
        if os.path.exists(f):
            os.remove(f)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pt = detect_ball(frame, cfg)
        # 写入检测结果
        with open(tmp_file, "w") as f:
            if pt is not None:
                f.write(f"{pt[0]},{pt[1]}\n")
            else:
                f.write("NaN,NaN\n")
        # 写入 ready 标记，表示这一帧处理完了
        open(ready_file, "w").close()
        # 等待 C++ 读取完成（删除 ready 文件）
        while os.path.exists(ready_file):
            time.sleep(0.001)
        cv2.imshow("Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    for f in [tmp_file, ready_file]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    main()
