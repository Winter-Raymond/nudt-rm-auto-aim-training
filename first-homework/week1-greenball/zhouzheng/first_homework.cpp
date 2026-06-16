#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>

int main(int argc, char* argv[]) {
    const std::string inputPath = "first-homework/first-homework.mp4";
    const std::string outputPath = "first-homework-output.mp4";
    const bool showWindow = false;

    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open input video: " << inputPath << std::endl;
        return 1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) fps = 30.0;

    cv::VideoWriter writer;
    int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
    if (!writer.open(outputPath, fourcc, fps, cv::Size(width, height), true)) {
        std::cerr << "Error: cannot open output video: " << outputPath << std::endl;
        return 1;
    }

    cv::Mat frame, hsv, mask, maskClean;
    cv::Point2f prevCenter(-1, -1);
    int frameIndex = 0;

    // 轨迹记录：保存最近的位置点
    std::deque<cv::Point2f> trail;
    const int maxTrailLength = 30;

    // 平滑参数：指数移动平均（EMA）
    const float emaAlphaPos = 0.25f;   // 位置平滑系数（越小越平滑）
    const float emaAlphaSpd = 0.3f;    // 速度平滑系数
    const float emaAlphaDir = 0.2f;    // 方向平滑系数

    // 平滑后的状态
    cv::Point2f smoothedCenter(-1, -1);  // 平滑后的球心位置
    float smoothedSpeed = 0.0f;          // 平滑后的速度大小
    cv::Point2f smoothedDir(0.0f, 0.0f); // 平滑后的运动方向

    // 速度箭头放大系数（让速度变化更直观可见）
    const float arrowScale = 3.0f;
    const float maxArrowLength = 150.0f;

    while (true) {
        if (!cap.read(frame)) break;
        ++frameIndex;
        if (frame.empty()) break;

        // ---------- 绿色检测 ----------
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // 绿色 HSV 范围（主要绿色区间）
        cv::Scalar lowerGreen(35, 60, 50);
        cv::Scalar upperGreen(90, 255, 255);
        cv::inRange(hsv, lowerGreen, upperGreen, mask);

        // 形态学处理：去除噪点、填充空洞
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});
        cv::morphologyEx(mask, maskClean, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(maskClean, maskClean, cv::MORPH_CLOSE, kernel);

        // ---------- 轮廓查找与筛选 ----------
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(maskClean, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double bestCirc = 0.0;  // 改为按圆形度选择（绿球始终最圆）
        cv::Point2f bestCenter(-1, -1);
        float bestRadius = 0.0f;

        for (auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < 500.0) continue;  // 过滤太小的噪点

            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(contour, center, radius);
            if (radius < 10.0f) continue;

            // 圆形度 = 轮廓面积 / 最小外接圆面积
            // 绿球 ≈ 0.9，圆柱体 ≈ 0.6，用 0.75 作为阈值区分
            double circleArea = CV_PI * radius * radius;
            double circularity = area / circleArea;
            if (circularity < 0.70) continue;  // 排除非圆形物体（圆柱体≈0.5，绿球≥0.73）

            // 亮度检查：绿球 V≈210，圆柱体 V≈110
            if (center.x >= 0 && center.y >= 0 &&
                center.x < hsv.cols && center.y < hsv.rows) {
                int v = static_cast<int>(hsv.at<cv::Vec3b>(
                    static_cast<int>(center.y), static_cast<int>(center.x))[2]);
                if (v < 120) continue;  // 排除较暗的绿色物体（圆柱体V≈90，绿球V≥136）
            }

            // 选择圆形度最高的绿色区域（绿球始终比圆柱体更圆）
            if (circularity > bestCirc) {
                bestCirc = circularity;
                bestCenter = center;
                bestRadius = radius;
            }
        }

        // ---------- 绘制 ----------
        if (bestCirc > 0 && bestRadius > 0.0f) {
            // 1. 球心位置 EMA 平滑（消除帧间抖动）
            if (smoothedCenter.x < 0) {
                smoothedCenter = bestCenter;  // 首帧直接初始化
            } else {
                smoothedCenter = emaAlphaPos * bestCenter + (1.0f - emaAlphaPos) * smoothedCenter;
            }

            // 2. 绘制绿球检测框（用平滑后的圆心）
            cv::circle(frame, bestCenter, static_cast<int>(bestRadius),
                       cv::Scalar(0, 255, 0), 2);
            cv::circle(frame, smoothedCenter, 4, cv::Scalar(0, 255, 255), cv::FILLED);

            // 3. 轨迹记录（用平滑后的位置）
            trail.push_back(smoothedCenter);
            if (trail.size() > static_cast<size_t>(maxTrailLength)) {
                trail.pop_front();
            }

            // 4. 计算速度与运动方向（基于平滑后的位置）
            if (prevCenter.x >= 0 && prevCenter.y >= 0) {
                cv::Point2f delta = smoothedCenter - prevCenter;
                float rawSpeed = std::hypot(delta.x, delta.y);

                // EMA 平滑速度大小
                smoothedSpeed = emaAlphaSpd * rawSpeed + (1.0f - emaAlphaSpd) * smoothedSpeed;

                // 运动方向（基于平滑位置的 delta）
                cv::Point2f dir(0.0f, 0.0f);
                if (rawSpeed > 0.3f) {
                    dir = delta / rawSpeed;

                    // EMA 平滑运动方向
                    if (smoothedDir.x == 0.0f && smoothedDir.y == 0.0f) {
                        smoothedDir = dir;
                    } else {
                        smoothedDir = emaAlphaDir * dir + (1.0f - emaAlphaDir) * smoothedDir;
                        // 归一化
                        float dirNorm = std::hypot(smoothedDir.x, smoothedDir.y);
                        if (dirNorm > 0.001f) {
                            smoothedDir /= dirNorm;
                        }
                    }
                }

                // ---------- 速度箭头（用平滑后的位置和方向）----------
                float arrowLen = std::min(smoothedSpeed * arrowScale, maxArrowLength);
                cv::Point2f arrowTip = smoothedCenter + smoothedDir * arrowLen;

                // 根据速度选择箭头颜色：慢→绿，中→黄，快→红
                cv::Scalar arrowColor;
                if (smoothedSpeed < 10.0f) {
                    arrowColor = cv::Scalar(0, 255, 0);
                } else if (smoothedSpeed < 30.0f) {
                    float t = (smoothedSpeed - 10.0f) / 20.0f;
                    arrowColor = cv::Scalar(0, 255 - static_cast<int>(t * 255),
                                             static_cast<int>(t * 255));
                } else {
                    arrowColor = cv::Scalar(0, 0, 255);
                }

                // 绘制速度箭头
                cv::arrowedLine(frame, smoothedCenter, arrowTip, arrowColor, 3,
                                cv::LINE_AA, 0, 0.2);

                // 5. 速度文本
                std::string speedText = cv::format("Speed: %.1f px/f", smoothedSpeed);
                int baseline = 0;
                cv::Size textSize = cv::getTextSize(speedText, cv::FONT_HERSHEY_SIMPLEX,
                                                    0.6, 2, &baseline);
                cv::Point textOrg = smoothedCenter + cv::Point2f(15, -20);
                cv::Rect bgRect(textOrg.x - 3, textOrg.y - textSize.height - 3,
                                textSize.width + 6, textSize.height + 6);
                cv::rectangle(frame, bgRect, cv::Scalar(40, 40, 40), cv::FILLED);
                cv::rectangle(frame, bgRect, cv::Scalar(200, 200, 200), 1);
                cv::putText(frame, speedText, textOrg,
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

                // 6. 历史位置连接线（淡灰色，基于平滑位置）
                cv::line(frame, prevCenter, smoothedCenter, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
            } else {
                smoothedSpeed = 0.0f;
                smoothedDir = cv::Point2f(0.0f, 0.0f);
            }

            prevCenter = smoothedCenter;  // 记住平滑后的位置
        } else {
            // 丢失目标：重置所有状态
            prevCenter = cv::Point2f(-1, -1);
            smoothedCenter = cv::Point2f(-1, -1);
            smoothedSpeed = 0.0f;
            smoothedDir = cv::Point2f(0.0f, 0.0f);
            trail.clear();
        }

        // ---------- 轨迹绘制 ----------
        for (size_t i = 1; i < trail.size(); ++i) {
            // 轨迹越旧越淡
            float alpha = static_cast<float>(i) / trail.size();
            cv::Scalar trailColor(0, static_cast<int>(255 * (1.0f - alpha)),
                                  static_cast<int>(255 * alpha));
            cv::line(frame, trail[i - 1], trail[i], trailColor, 2, cv::LINE_AA);
        }

        // ---------- 图例 ----------
        int legendX = 15;
        int legendY = height - 90;
        cv::Rect legendBg(legendX, legendY, 160, 80);
        cv::rectangle(frame, legendBg, cv::Scalar(40, 40, 40), cv::FILLED);
        cv::rectangle(frame, legendBg, cv::Scalar(200, 200, 200), 1);

        cv::putText(frame, "--- Speed ---", cv::Point(legendX + 5, legendY + 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1);
        cv::arrowedLine(frame, cv::Point(legendX + 8, legendY + 35),
                        cv::Point(legendX + 38, legendY + 35),
                        cv::Scalar(0, 255, 0), 3, cv::LINE_AA, 0, 0.2);
        cv::putText(frame, "Slow", cv::Point(legendX + 45, legendY + 38),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        cv::arrowedLine(frame, cv::Point(legendX + 8, legendY + 55),
                        cv::Point(legendX + 53, legendY + 55),
                        cv::Scalar(0, 255, 255), 3, cv::LINE_AA, 0, 0.2);
        cv::putText(frame, "Medium", cv::Point(legendX + 60, legendY + 58),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
        cv::arrowedLine(frame, cv::Point(legendX + 8, legendY + 72),
                        cv::Point(legendX + 63, legendY + 72),
                        cv::Scalar(0, 0, 255), 3, cv::LINE_AA, 0, 0.2);
        cv::putText(frame, "Fast", cv::Point(legendX + 70, legendY + 75),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

        // ---------- 帧计数器 ----------
        cv::putText(frame, cv::format("Frame: %d", frameIndex),
                    cv::Point(width - 160, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 2);

        writer.write(frame);

        if (showWindow) {
            cv::imshow("Green Ball Tracking", frame);
            if (cv::waitKey(5) == 27) break;
        }
    }

    cap.release();
    writer.release();
    if (showWindow) cv::destroyAllWindows();

    std::cout << "Processed " << frameIndex << " frames. Output saved to: " << outputPath << std::endl;
    return 0;
}
