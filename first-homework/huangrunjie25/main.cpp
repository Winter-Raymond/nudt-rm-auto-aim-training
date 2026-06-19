#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/tracking.hpp>

#include <iostream>
#include <deque>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

// ============================================================
// 参数配置
// ============================================================

// --- 绿色检测 (HSV) ---
const cv::Scalar LOWER_GREEN(38, 55, 55);
const cv::Scalar UPPER_GREEN(88, 255, 255);

// --- 形态学 ---
const int   MORPH_KERNEL_SIZE = 5;
const int   MORPH_OPEN_ITERS  = 1;
const int   MORPH_CLOSE_ITERS = 2;

// --- 轮廓筛选 ---
const double MIN_CONTOUR_AREA   = 300.0;
const double MIN_CIRCULARITY    = 0.65;    // 圆形度 = 4π·面积 / 周长²
const double MIN_RADIUS         = 8.0;
const double MAX_ASPECT_RATIO   = 1.5;     // 包围盒宽高比上限 (球体投影 ≈ 1.0)

// --- 卡尔曼滤波器 ---
const float  KF_PROCESS_NOISE    = 1e-2f;  // 过程噪声 (模型不确定性)
const float  KF_MEASURE_NOISE    = 1e-1f;  // 测量噪声 (检测不确定性)
const int    KF_MAX_LOST         = 15;      // 最大丢失帧数 (超出则重置)

// --- ROI 搜索 ---
const int    ROI_MARGIN = 60;              // 在上一帧位置周围扩展的像素

// --- 轨迹 ---
const int    TRAIL_LENGTH = 40;

// --- 可视化 ---
const double ARROW_SCALE       = 2.8;      // 速度箭头缩放
const double MAX_ARROW_LENGTH  = 140.0;    // 箭头最大长度 (像素)
const bool   SHOW_MASK_WINDOW  = false;    // 是否显示掩膜调试窗口
const bool   SHOW_WINDOW       = false;    // 是否显示预览窗口 (snap环境建议关闭)

// --- 输入/输出 ---
const std::string INPUT_VIDEO  = "../../first-homework.mp4";
const std::string OUTPUT_VIDEO = "tracked_output.mp4";

// ============================================================
// 工具函数：计算两点间的欧氏距离
// ============================================================
inline double distance(const cv::Point2f& a, const cv::Point2f& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

// ============================================================
// 工具函数：计算轮廓的圆形度 (circularity)
//   circularity = 4π × area / perimeter²
//   正圆→1.0, 正方形→≈0.785, 细长形→趋近 0
// ============================================================
inline double circularity(const std::vector<cv::Point>& contour, double area) {
    double perimeter = cv::arcLength(contour, true);
    if (perimeter < 1e-6) return 0.0;
    return (4.0 * CV_PI * area) / (perimeter * perimeter);
}

// ============================================================
// 核心函数：在图像中检测绿色小球
//
// 流程:
//   1. BGR → HSV 颜色空间转换
//   2. 绿色阈值二值化
//   3. 形态学开/闭运算去噪 + 填充空洞
//   4. 轮廓检测
//   5. 三级筛选：面积 → 圆形度 → 宽高比
//   6. 返回最优候选 (圆形度最高)
//
// 参数:
//   frame       - 输入 BGR 图像
//   maskOut     - (可选输出) 处理后的二值掩膜，用于调试
//   searchROI   - (可选) 限定搜索区域，空矩形表示全图搜索
//
// 返回:
//   检测到的球心坐标和半径；若未检测到则 radius = -1
// ============================================================
struct BallDetection {
    cv::Point2f center;
    float radius;
    double confidence;  // 圆形度，作为置信度
};

BallDetection detectGreenBall(const cv::Mat& frame,
                              cv::Mat* maskOut = nullptr,
                              cv::Rect searchROI = cv::Rect()) {
    BallDetection result = {cv::Point2f(-1, -1), -1.0f, 0.0};

    // 确定工作区域
    cv::Mat workRegion;
    int offsetX = 0, offsetY = 0;
    if (!searchROI.empty()) {
        // 确保 ROI 在图像范围内
        cv::Rect roi = searchROI & cv::Rect(0, 0, frame.cols, frame.rows);
        if (roi.width <= 0 || roi.height <= 0) return result;
        workRegion = frame(roi);
        offsetX = roi.x;
        offsetY = roi.y;
    } else {
        workRegion = frame;
    }

    // ---- 1. HSV 转换 + 绿色阈值 ----
    cv::Mat hsv, mask;
    cv::cvtColor(workRegion, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, LOWER_GREEN, UPPER_GREEN, mask);

    // ---- 2. 形态学处理 ----
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                               cv::Size(MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE));
    cv::Mat maskClean;
    cv::morphologyEx(mask, maskClean, cv::MORPH_OPEN,  kernel, cv::Point(-1, -1), MORPH_OPEN_ITERS);
    cv::morphologyEx(maskClean, maskClean, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), MORPH_CLOSE_ITERS);

    if (maskOut) *maskOut = maskClean.clone();

    // ---- 3. 轮廓检测 ----
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(maskClean, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // ---- 4. 三级筛选 ----
    double bestCirc = 0.0;
    cv::Point2f bestCenter(-1, -1);
    float bestRadius = -1.0f;

    for (const auto& cnt : contours) {
        // -- 第一级：面积过滤 --
        double area = cv::contourArea(cnt);
        if (area < MIN_CONTOUR_AREA) continue;

        // -- 第二级：圆形度过滤 --
        double circ = circularity(cnt, area);
        if (circ < MIN_CIRCULARITY) continue;

        // -- 第三级：宽高比过滤 --
        cv::Rect bbox = cv::boundingRect(cnt);
        double aspectRatio = static_cast<double>(std::max(bbox.width, bbox.height)) /
                             std::max(1.0, static_cast<double>(std::min(bbox.width, bbox.height)));
        if (aspectRatio > MAX_ASPECT_RATIO) continue;

        // -- 最小外接圆 --
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(cnt, center, radius);
        if (radius < MIN_RADIUS) continue;

        // 选择圆形度最高的候选
        if (circ > bestCirc) {
            bestCirc = circ;
            bestCenter = center;
            bestRadius = radius;
        }
    }

    if (bestRadius > 0) {
        bestCenter.x += offsetX;
        bestCenter.y += offsetY;
        result.center     = bestCenter;
        result.radius     = bestRadius;
        result.confidence = bestCirc;
    }
    return result;
}

// ============================================================
// 主函数
// ============================================================
int main() {
    // ---- 1. 打开视频 ----
    cv::VideoCapture cap(INPUT_VIDEO);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] 无法打开输入视频: " << INPUT_VIDEO << std::endl;
        return -1;
    }

    int frameWidth  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps      = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) fps = 30.0;

    std::cout << "[INFO] 输入视频: " << frameWidth << "x" << frameHeight
              << " @ " << fps << " FPS" << std::endl;

    // ---- 2. 初始化视频写入器 ----
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer;
    if (!writer.open(OUTPUT_VIDEO, fourcc, fps, cv::Size(frameWidth, frameHeight), true)) {
        std::cerr << "[ERROR] 无法创建输出视频: " << OUTPUT_VIDEO << std::endl;
        return -1;
    }

    // ---- 3. 初始化卡尔曼滤波器 ----
    // 状态: [x, y, vx, vy]^T  (位置 + 速度)
    // 观测: [x, y]^T
    cv::KalmanFilter kf(4, 2, 0, CV_32F);

    // 状态转移矩阵 (匀速模型):  x_t = x_{t-1} + vx_{t-1} * dt
    //                          y_t = y_{t-1} + vy_{t-1} * dt
    //                          vx_t = vx_{t-1}
    //                          vy_t = vy_{t-1}
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);

    // 观测矩阵: 只观测位置
    kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    // 协方差矩阵
    cv::setIdentity(kf.processNoiseCov,   cv::Scalar::all(KF_PROCESS_NOISE));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(KF_MEASURE_NOISE));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1.0));

    // ---- 4. 状态变量 ----
    bool     tracked     = false;     // 是否已捕获目标
    int      lostCount   = 0;         // 连续丢失帧计数
    cv::Mat  measurement(2, 1, CV_32F);
    std::deque<cv::Point2f> trail;   // 轨迹记录

    cv::Mat frame, output;

    // ---- 5. 逐帧处理 ----
    int frameIdx = 0;
    for (frameIdx = 1; ; ++frameIdx) {
        if (!cap.read(frame) || frame.empty()) break;
        frame.copyTo(output);

        // ---- 5a. 绿色小球检测 ----
        BallDetection detection;
        cv::Rect searchROI;

        if (tracked) {
            // 卡尔曼预测下一帧位置
            cv::Mat prediction = kf.predict();

            float px = prediction.at<float>(0);
            float py = prediction.at<float>(1);

            // 在预测位置周围划定 ROI 搜索区域
            int roiX = std::max(0,     static_cast<int>(px - ROI_MARGIN));
            int roiY = std::max(0,     static_cast<int>(py - ROI_MARGIN));
            int roiW = std::min(frameWidth  - roiX, 2 * ROI_MARGIN);
            int roiH = std::min(frameHeight - roiY, 2 * ROI_MARGIN);
            searchROI = cv::Rect(roiX, roiY, roiW, roiH);

            // 在 ROI 中检测
            detection = detectGreenBall(frame, nullptr, searchROI);

            // ROI 搜索失败时回退到全图搜索
            if (detection.radius < 0) {
                detection = detectGreenBall(frame);
            }
        } else {
            // 未捕获目标：全图搜索
            detection = detectGreenBall(frame);
        }

        // ---- 5b. 卡尔曼更新 / 丢失处理 ----
        if (detection.radius > 0) {
            // 检测到目标 → 卡尔曼校正
            measurement.at<float>(0) = detection.center.x;
            measurement.at<float>(1) = detection.center.y;

            if (!tracked) {
                // 首次捕获：用初始测量值初始化卡尔曼状态
                kf.statePost.at<float>(0) = detection.center.x;
                kf.statePost.at<float>(1) = detection.center.y;
                kf.statePost.at<float>(2) = 0;
                kf.statePost.at<float>(3) = 0;
                tracked = true;
            }
            kf.correct(measurement);

            lostCount = 0;
        } else {
            // 未检测到目标
            lostCount++;
            if (lostCount > KF_MAX_LOST) {
                // 长时间丢失：重置追踪器
                tracked   = false;
                lostCount = 0;
                trail.clear();
            }
        }

        // ---- 5c. 获取滤波后的状态 ----
        cv::Point2f filteredPos(-1, -1);
        cv::Point2f filteredVel(0, 0);

        if (tracked && lostCount < KF_MAX_LOST) {
            cv::Mat state = kf.statePost;
            filteredPos = cv::Point2f(state.at<float>(0), state.at<float>(1));
            filteredVel = cv::Point2f(state.at<float>(2), state.at<float>(3));
        }

        // ---- 5d. 可视化绘制 ----

        // -- 原始检测圆 --
        if (detection.radius > 0) {
            cv::circle(output, detection.center,
                       static_cast<int>(detection.radius),
                       cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::circle(output, detection.center, 3,
                       cv::Scalar(0, 200, 0), cv::FILLED, cv::LINE_AA);
        }

        if (filteredPos.x >= 0 && tracked) {
            // -- 卡尔曼滤波圆心 --
            cv::circle(output, filteredPos, 5,
                       cv::Scalar(255, 80, 0), cv::FILLED, cv::LINE_AA);
            cv::circle(output, filteredPos, 7,
                       cv::Scalar(255, 160, 0), 2, cv::LINE_AA);

            // -- 速度箭头 --
            float speed = static_cast<float>(std::hypot(filteredVel.x, filteredVel.y));
            if (speed > 0.05f) {
                cv::Point2f velDir = filteredVel / speed;
                float arrowLen = std::min(static_cast<float>(ARROW_SCALE) * speed,
                                          static_cast<float>(MAX_ARROW_LENGTH));
                cv::Point2f arrowTip = filteredPos + velDir * arrowLen;

                // 颜色: 慢→蓝, 中→黄, 快→红
                cv::Scalar arrowColor;
                if (speed < 5.0f) {
                    arrowColor = cv::Scalar(255, 100, 0);           // 蓝色系
                } else if (speed < 15.0f) {
                    arrowColor = cv::Scalar(0, 230, 230);           // 黄色
                } else {
                    // 黄→红渐变
                    float r = (speed - 15.0f) / 15.0f;
                    r = std::min(1.0f, r);
                    arrowColor = cv::Scalar(0,
                        static_cast<int>(230 * (1.0f - r)),
                        static_cast<int>(230 * (1.0f - r) + 255 * r));
                }

                cv::arrowedLine(output, filteredPos, arrowTip,
                                arrowColor, 3, cv::LINE_AA, 0, 0.25);
            }

            // -- 速度文本 --
            std::string speedStr = cv::format("v: %.1f px/f", speed);
            cv::Point textPos = filteredPos + cv::Point2f(15, -20);
            int baseline = 0;
            cv::Size textSz = cv::getTextSize(speedStr,
                                              cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);
            cv::Rect bg(textPos.x - 4, textPos.y - textSz.height - 4,
                        textSz.width + 8, textSz.height + 8);
            cv::rectangle(output, bg, cv::Scalar(30, 30, 30), cv::FILLED);
            cv::rectangle(output, bg, cv::Scalar(180, 180, 180), 1);
            cv::putText(output, speedStr, textPos,
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(240, 240, 240), 2, cv::LINE_AA);

            // -- 轨迹记录 --
            trail.push_back(filteredPos);
            if (static_cast<int>(trail.size()) > TRAIL_LENGTH) {
                trail.pop_front();
            }
        }

        // -- 轨迹连线 --
        if (trail.size() >= 2) {
            for (size_t i = 1; i < trail.size(); ++i) {
                double alpha = static_cast<double>(i) / trail.size();
                cv::Scalar color(static_cast<int>(255 * (1.0 - alpha)),
                                 static_cast<int>(80 + 120 * alpha),
                                 static_cast<int>(255 * alpha));
                cv::line(output, trail[i - 1], trail[i], color, 2, cv::LINE_AA);
            }
        }

        // -- 目标丢失提示 --
        if (lostCount > 0 && tracked) {
            std::string lostText = cv::format("LOST: %d/%d", lostCount, KF_MAX_LOST);
            cv::putText(output, lostText, cv::Point(frameWidth - 200, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }

        // -- 帧计数 --
        cv::putText(output, cv::format("Frame: %d", frameIdx),
                    cv::Point(frameWidth - 170, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

        // -- 状态指示 --
        cv::Scalar statusColor = tracked ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        std::string statusText = tracked ? "TRACKING" : "SEARCHING";
        cv::circle(output, cv::Point(20, 20), 6, statusColor, cv::FILLED, cv::LINE_AA);
        cv::putText(output, statusText, cv::Point(32, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, statusColor, 2, cv::LINE_AA);

        // -- 图例 (左下角) --
        {
            int lx = 12, ly = frameHeight - 100;
            cv::Rect legendBg(lx, ly, 155, 90);
            cv::rectangle(output, legendBg, cv::Scalar(35, 35, 35), cv::FILLED);
            cv::rectangle(output, legendBg, cv::Scalar(180, 180, 180), 1);

            cv::putText(output, "Detection", cv::Point(lx + 5,  ly + 17),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            cv::putText(output, "Kalman Est.", cv::Point(lx + 5, ly + 35),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 160, 0), 1, cv::LINE_AA);

            cv::arrowedLine(output, cv::Point(lx + 8,  ly + 55),
                           cv::Point(lx + 35, ly + 55),
                           cv::Scalar(255, 100, 0), 2, cv::LINE_AA, 0, 0.2);
            cv::putText(output, "Slow", cv::Point(lx + 42, ly + 58),
                        cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);

            cv::arrowedLine(output, cv::Point(lx + 8,  ly + 72),
                           cv::Point(lx + 45, ly + 72),
                           cv::Scalar(0, 230, 230), 2, cv::LINE_AA, 0, 0.2);
            cv::putText(output, "Mid", cv::Point(lx + 52, ly + 75),
                        cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);

            cv::arrowedLine(output, cv::Point(lx + 80, ly + 72),
                           cv::Point(lx + 125, ly + 72),
                           cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.2);
            cv::putText(output, "Fast", cv::Point(lx + 132, ly + 75),
                        cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        }

        // ---- 6. 写入输出视频 + 显示 ----
        writer.write(output);

        if (SHOW_WINDOW) {
            cv::imshow("Green Ball Tracker", output);
            int key = cv::waitKey(5);
            if (key == 27 || key == 'q') break;
        }

        // 进度打印 (每30帧)
        if (frameIdx % 30 == 0) {
            std::cout << "\r[INFO] 已处理 " << frameIdx << " 帧..." << std::flush;
        }
    }

    std::cout << "\r[INFO] 已处理 " << (frameIdx - 1) << " 帧。" << std::endl;

    // ---- 7. 清理 ----
    cap.release();
    writer.release();
    if (SHOW_WINDOW) cv::destroyAllWindows();

    std::cout << "[INFO] 处理完成。输出视频: " << OUTPUT_VIDEO << std::endl;
    return 0;
}
