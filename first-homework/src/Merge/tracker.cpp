#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include "json.hpp"
#include <unistd.h>

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace cv;
using namespace std;

struct Config {
    string input_video;
    string output_video;
    string output_csv;
    Mat camera_matrix;
    Mat dist_coeffs;
    double ball_radius;
    double kf_process_noise;
    double kf_measure_noise;
    int max_lost_frame;
    double speed_scale;
    bool show_track_line;
    int track_line_length;
};

class KalmanFilter2D {
public:
    KalmanFilter2D(double dt, double proc_noise, double meas_noise) {
        kf.init(4, 2, 0);
        kf.transitionMatrix = (Mat_<float>(4, 4) <<
            1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1);
        setIdentity(kf.measurementMatrix);
        setIdentity(kf.processNoiseCov, Scalar::all(proc_noise));
        setIdentity(kf.measurementNoiseCov, Scalar::all(meas_noise));
        setIdentity(kf.errorCovPost, Scalar::all(10)); // 减小初始协方差，抑制发散
    }
    Point2f predict() {
        Mat pred = kf.predict();
        return Point2f(pred.at<float>(0), pred.at<float>(1));
    }
    Point2f update(Point2f pt) {
        Mat meas = (Mat_<float>(2, 1) << pt.x, pt.y);
        Mat corr = kf.correct(meas);
        return Point2f(corr.at<float>(0), corr.at<float>(1));
    }
private:
    KalmanFilter kf;
};

bool loadConfig(Config& cfg, const string& cfg_path) {
    if (!fs::exists(cfg_path)) {
        cerr << "Config file not found: " << cfg_path << endl;
        return false;
    }
    ifstream f(cfg_path);
    json j;
    f >> j;

    cfg.input_video = j["path"]["input_video"];
    cfg.output_video = j["path"]["output_video"];
    cfg.output_csv = j["path"]["output_csv"];

    auto cm = j["camera"]["camera_matrix"];
    cfg.camera_matrix = Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
        for (int k = 0; k < 3; k++)
            cfg.camera_matrix.at<double>(i, k) = cm[i][k];

    auto dc = j["camera"]["dist_coeffs"];
    cfg.dist_coeffs = Mat(4, 1, CV_64F);
    for (int i = 0; i < 4; i++)
        cfg.dist_coeffs.at<double>(i, 0) = dc[i];

    cfg.ball_radius = j["camera"]["ball_radius"];
    cfg.kf_process_noise = j["kalman"]["process_noise"];
    cfg.kf_measure_noise = j["kalman"]["measurement_noise"];
    cfg.max_lost_frame = j["kalman"]["max_lost_frame"];
    cfg.speed_scale = j["visual"]["speed_scale"];
    cfg.show_track_line = j["visual"]["show_track_line"];
    cfg.track_line_length = j["visual"]["track_line_length"];

    return true;
}

Point2f readDetectorResult(const string& tmp_file, const string& ready_file) {
    // 等待 ready 文件出现
    while (!fs::exists(ready_file)) {
        usleep(1000);
    }
    // 读取检测结果
    ifstream f(tmp_file);
    if (!f.is_open()) return Point2f(-1, -1);
    string line;
    getline(f, line);
    // 删除 ready 文件，通知 Python 已读取
    fs::remove(ready_file);
    if (line == "NaN,NaN") return Point2f(-1, -1);
    size_t comma = line.find(',');
    float x = stof(line.substr(0, comma));
    float y = stof(line.substr(comma + 1));
    return Point2f(x, y);
}

void calcSpeedDir(Point2f prev, Point2f curr, double& speed, double& angle_deg) {
    if (prev.x < 0 || curr.x < 0) {
        speed = 0.0;
        angle_deg = 0.0;
        return;
    }
    double dx = curr.x - prev.x;
    double dy = curr.y - prev.y;
    speed = hypot(dx, dy);
    double rad = atan2(dy, dx);
    angle_deg = rad * 180.0 / CV_PI;
    if (angle_deg < 0) angle_deg += 360.0;
}

int main() {
    Config cfg;
    if (!loadConfig(cfg, "config.json")) {
        cerr << "Failed to load config!" << endl;
        return -1;
    }

    fs::create_directories(fs::path(cfg.output_video).parent_path());
    fs::create_directories(fs::path(cfg.output_csv).parent_path());

    VideoCapture cap(cfg.input_video);
    if (!cap.isOpened()) {
        cerr << "Open video failed! Path: " << cfg.input_video << endl;
        return -1;
    }
    double fps = cap.get(CAP_PROP_FPS);
    int w = cap.get(CAP_PROP_FRAME_WIDTH);
    int h = cap.get(CAP_PROP_FRAME_HEIGHT);

    VideoWriter out_video(cfg.output_video, VideoWriter::fourcc('m','p','4','v'), fps, Size(w, h));
    ofstream csv_f(cfg.output_csv);
    csv_f << "frame_idx,time_s,speed_pixel,direction_deg\n";

    KalmanFilter2D kf(1.0 / fps, cfg.kf_process_noise, cfg.kf_measure_noise);
    Point2f prev_center(-1, -1);
    vector<Point2f> track_pts;
    int lost_frames = 0;
    int frame_idx = 0;
    string tmp_file = "detector.tmp";
    string ready_file = "detector.ready";

    Mat frame;
    while (cap.read(frame)) {
        double time_s = frame_idx / fps;
        // 同步读取检测结果
        Point2f curr_center = readDetectorResult(tmp_file, ready_file);

        // 绘制检测点
        if (curr_center.x > 0) {
            circle(frame, curr_center, 8, Scalar(0, 0, 255), -1);
        }

        // 计算速度和方向（首帧强制为0）
        double speed = 0.0, angle = 0.0;
        if (frame_idx > 0 && prev_center.x > 0 && curr_center.x > 0) {
            calcSpeedDir(prev_center, curr_center, speed, angle);
        }

        // 卡尔曼滤波更新
        Point2f pred_pt = kf.predict();
        Point2f kf_pt = pred_pt;
        if (curr_center.x > 0) {
            kf_pt = kf.update(curr_center);
            track_pts.push_back(kf_pt);
            if ((int)track_pts.size() > cfg.track_line_length) {
                track_pts.erase(track_pts.begin());
            }
            lost_frames = 0;
        } else {
            lost_frames++;
            if (lost_frames > cfg.max_lost_frame) {
                kf = KalmanFilter2D(1.0 / fps, cfg.kf_process_noise, cfg.kf_measure_noise);
                track_pts.clear();
                lost_frames = 0;
            }
        }

        // 绘制轨迹线
        if (cfg.show_track_line && track_pts.size() > 1) {
            for (size_t i = 1; i < track_pts.size(); i++) {
                line(frame, track_pts[i-1], track_pts[i], Scalar(255, 0, 255), 2);
            }
        }

        // 绘制速度箭头和文字
        if (curr_center.x > 0 && prev_center.x > 0) {
            double dx = curr_center.x - prev_center.x;
            double dy = curr_center.y - prev_center.y;
            Point2f end(curr_center.x + dx * cfg.speed_scale, curr_center.y + dy * cfg.speed_scale);
            arrowedLine(frame, curr_center, end, Scalar(0, 255, 0), 3, 0, 0, 0.3);
            putText(frame, format("Speed:%.1f", speed),
                    Point(curr_center.x + 10, curr_center.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
        }

        // 写入CSV
        csv_f << frame_idx << ","
              << fixed << setprecision(3) << time_s << ","
              << fixed << setprecision(2) << speed << ","
              << fixed << setprecision(2) << angle << "\n";

        prev_center = curr_center;
        frame_idx++;

        out_video << frame;
        imshow("Tracker", frame);
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    out_video.release();
    csv_f.close();
    destroyAllWindows();
    return 0;
}
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include "json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace cv;
using namespace std;

struct Config {
    string input_video;
    string output_video;
    string output_csv;
    Mat camera_matrix;
    Mat dist_coeffs;
    double ball_radius;
    double kf_process_noise;
    double kf_measure_noise;
    int max_lost_frame;
    double speed_scale;
    bool show_track_line;
    int track_line_length;
};

class KalmanFilter2D {
public:
    KalmanFilter2D(double dt, double proc_noise, double meas_noise) {
        kf.init(4, 2, 0);
        kf.transitionMatrix = (Mat_<float>(4, 4) <<
            1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1);
        setIdentity(kf.measurementMatrix);
        setIdentity(kf.processNoiseCov, Scalar::all(proc_noise));
        setIdentity(kf.measurementNoiseCov, Scalar::all(meas_noise));
        setIdentity(kf.errorCovPost, Scalar::all(1000));
    }
    Point2f predict() {
        Mat pred = kf.predict();
        return Point2f(pred.at<float>(0), pred.at<float>(1));
    }
    Point2f update(Point2f pt) {
        Mat meas = (Mat_<float>(2, 1) << pt.x, pt.y);
        Mat corr = kf.correct(meas);
        return Point2f(corr.at<float>(0), corr.at<float>(1));
    }
private:
    KalmanFilter kf;
};

// 修正：参数顺序为 (cfg, cfg_path)
bool loadConfig(Config& cfg, const string& cfg_path) {
    if (!fs::exists(cfg_path)) {
        cerr << "Config file not found: " << cfg_path << endl;
        return false;
    }
    ifstream f(cfg_path);
    json j;
    f >> j;

    cfg.input_video = j["path"]["input_video"];
    cfg.output_video = j["path"]["output_video"];
    cfg.output_csv = j["path"]["output_csv"];

    auto cm = j["camera"]["camera_matrix"];
    cfg.camera_matrix = Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
        for (int k = 0; k < 3; k++)
            cfg.camera_matrix.at<double>(i, k) = cm[i][k];

    auto dc = j["camera"]["dist_coeffs"];
    cfg.dist_coeffs = Mat(4, 1, CV_64F);
    for (int i = 0; i < 4; i++)
        cfg.dist_coeffs.at<double>(i, 0) = dc[i];

    cfg.ball_radius = j["camera"]["ball_radius"];
    cfg.kf_process_noise = j["kalman"]["process_noise"];
    cfg.kf_measure_noise = j["kalman"]["measurement_noise"];
    cfg.max_lost_frame = j["kalman"]["max_lost_frame"];
    cfg.speed_scale = j["visual"]["speed_scale"];
    cfg.show_track_line = j["visual"]["show_track_line"];
    cfg.track_line_length = j["visual"]["track_line_length"];

    return true;
}

Point2f readDetectorResult(const string& path) {
    ifstream f(path);
    if (!f.is_open()) return Point2f(-1, -1);
    string line;
    getline(f, line);
    if (line == "NaN,NaN") return Point2f(-1, -1);
    size_t comma = line.find(',');
    if (comma == string::npos) return Point2f(-1, -1);
    float x = stof(line.substr(0, comma));
    float y = stof(line.substr(comma + 1));
    return Point2f(x, y);
}

void calcSpeedDir(Point2f prev, Point2f curr, double& speed, double& angle_deg) {
    if (prev.x < 0 || curr.x < 0) {
        speed = 0.0;
        angle_deg = 0.0;
        return;
    }
    double dx = curr.x - prev.x;
    double dy = curr.y - prev.y;
    speed = hypot(dx, dy);
    double rad = atan2(dy, dx);
    angle_deg = rad * 180.0 / CV_PI;
    if (angle_deg < 0) angle_deg += 360.0;
}

int main() {
    Config cfg;
    // 修正：参数顺序为 (cfg, "config.json")
    if (!loadConfig(cfg, "config.json")) {
        cerr << "Failed to load config!" << endl;
        return -1;
    }

    // 创建输出目录
    fs::create_directories(fs::path(cfg.output_video).parent_path());
    fs::create_directories(fs::path(cfg.output_csv).parent_path());

    VideoCapture cap(cfg.input_video);
    if (!cap.isOpened()) {
        cerr << "Open video failed! Path: " << cfg.input_video << endl;
        return -1;
    }
    double fps = cap.get(CAP_PROP_FPS);
    int w = cap.get(CAP_PROP_FRAME_WIDTH);
    int h = cap.get(CAP_PROP_FRAME_HEIGHT);

    VideoWriter out_video(cfg.output_video, VideoWriter::fourcc('m','p','4','v'), fps, Size(w, h));
    ofstream csv_f(cfg.output_csv);
    csv_f << "frame_idx,time_s,speed_pixel,direction_deg\n";

    KalmanFilter2D kf(1.0 / fps, cfg.kf_process_noise, cfg.kf_measure_noise);
    Point2f prev_center(-1, -1);
    vector<Point2f> track_pts;
    int lost_frames = 0;
    int frame_idx = 0;
    string tmp_file = "detector.tmp";

    Mat frame;
    while (cap.read(frame)) {
        double time_s = frame_idx / fps;
        Point2f curr_center = readDetectorResult(tmp_file);

        // 绘制检测点
        if (curr_center.x > 0) {
            circle(frame, curr_center, 8, Scalar(0, 0, 255), -1);
        }

        // 计算速度和方向（首帧强制为0）
        double speed = 0.0, angle = 0.0;
        if (frame_idx > 0 && prev_center.x > 0 && curr_center.x > 0) {
            calcSpeedDir(prev_center, curr_center, speed, angle);
        }

        // 卡尔曼滤波更新
        Point2f pred_pt = kf.predict();
        Point2f kf_pt = pred_pt;
        if (curr_center.x > 0) {
            kf_pt = kf.update(curr_center);
            track_pts.push_back(kf_pt);
            if ((int)track_pts.size() > cfg.track_line_length) {
                track_pts.erase(track_pts.begin());
            }
            lost_frames = 0;
        } else {
            lost_frames++;
            if (lost_frames > cfg.max_lost_frame) {
                kf = KalmanFilter2D(1.0 / fps, cfg.kf_process_noise, cfg.kf_measure_noise);
                track_pts.clear();
                lost_frames = 0;
            }
        }

        // 绘制轨迹线
        if (cfg.show_track_line && track_pts.size() > 1) {
            for (size_t i = 1; i < track_pts.size(); i++) {
                line(frame, track_pts[i-1], track_pts[i], Scalar(255, 0, 255), 2);
            }
        }

        // 绘制速度箭头和文字
        if (curr_center.x > 0 && prev_center.x > 0) {
            double dx = curr_center.x - prev_center.x;
            double dy = curr_center.y - prev_center.y;
            Point2f end(curr_center.x + dx * cfg.speed_scale, curr_center.y + dy * cfg.speed_scale);
            arrowedLine(frame, curr_center, end, Scalar(0, 255, 0), 3, 0, 0, 0.3);
            putText(frame, format("Speed:%.1f", speed),
                    Point(curr_center.x + 10, curr_center.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
        }

        // 写入CSV
        csv_f << frame_idx << ","
              << fixed << setprecision(3) << time_s << ","
              << fixed << setprecision(2) << speed << ","
              << fixed << setprecision(2) << angle << "\n";

        prev_center = curr_center;
        frame_idx++;

        out_video << frame;
        imshow("Tracker", frame);
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    out_video.release();
    csv_f.close();
    destroyAllWindows();
    return 0;
}
