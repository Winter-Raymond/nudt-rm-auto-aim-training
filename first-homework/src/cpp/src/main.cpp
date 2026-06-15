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

// ===================== 全局配置结构体 =====================
struct Config
{
    string input_video;
    string output_video;

    Mat camera_matrix;
    Mat dist_coeffs;
    double ball_radius;

    Scalar lower_green;
    Scalar upper_green;
    int morph_kernel_size;
    double min_contour_area;

    double min_center_std;
    double axis_ratio_weight;
    double center_bias_weight;

    double kf_process_noise;
    double kf_measure_noise;
    int max_lost_frame;

    double speed_scale;
    bool show_track_line;
    int track_line_length;

    double margin_w_ratio;
    double margin_h_ratio;
};

// ===================== 卡尔曼滤波类 =====================
class KalmanFilter2D
{
public:
    KalmanFilter2D(double dt, double proc_noise, double meas_noise)
    {
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

    Point2f predict()
    {
        Mat pred = kf.predict();
        return Point2f(pred.at<float>(0), pred.at<float>(1));
    }

    Point2f update(Point2f pt)
    {
        Mat meas = (Mat_<float>(2, 1) << pt.x, pt.y);
        Mat corr = kf.correct(meas);
        return Point2f(corr.at<float>(0), corr.at<float>(1));
    }

private:
    KalmanFilter kf;
};

// ===================== 工具函数 =====================
double calcContourStd(const vector<Point>& cnt, Point2f center)
{
    vector<double> dists;
    for (auto& p : cnt)
    {
        double d = hypot(p.x - center.x, p.y - center.y);
        dists.push_back(d);
    }
    double mean = 0;
    for (double d : dists) mean += d;
    mean /= dists.size();

    double stdv = 0;
    for (double d : dists)
        stdv += pow(d - mean, 2);
    return sqrt(stdv / dists.size());
}

void calcSpeedDir(Point2f prev, Point2f curr, double& speed, double& angle_deg)
{
    if (prev.x < 0 || curr.x < 0)
    {
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

bool loadConfig(Config& cfg, const string& cfg_path)
{
    if (!fs::exists(cfg_path))
    {
        cerr << "Config file not found: " << cfg_path << endl;
        return false;
    }
    ifstream f(cfg_path);
    json j;
    f >> j;

    cfg.input_video  = j["path"]["input_video"];
    cfg.output_video = j["path"]["output_video"];

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

    auto lg = j["color_detect"]["lower_green"];
    auto ug = j["color_detect"]["upper_green"];
    cfg.lower_green  = Scalar(lg[0], lg[1], lg[2]);
    cfg.upper_green  = Scalar(ug[0], ug[1], ug[2]);
    cfg.morph_kernel_size = j["color_detect"]["morph_kernel_size"];
    cfg.min_contour_area  = j["color_detect"]["min_contour_area"];

    cfg.min_center_std      = j["shape_filter"]["min_center_std"];
    cfg.axis_ratio_weight   = j["shape_filter"]["axis_ratio_weight"];
    cfg.center_bias_weight  = j["shape_filter"]["center_bias_weight"];

    cfg.kf_process_noise   = j["kalman"]["process_noise"];
    cfg.kf_measure_noise    = j["kalman"]["measurement_noise"];
    cfg.max_lost_frame      = j["kalman"]["max_lost_frame"];

    cfg.speed_scale    = j["visual"]["speed_scale"];
    cfg.show_track_line= j["visual"]["show_track_line"];
    cfg.track_line_length = j["visual"]["track_line_length"];

    cfg.margin_w_ratio = j["roi_margin"]["margin_w_ratio"];
    cfg.margin_h_ratio = j["roi_margin"]["margin_h_ratio"];

    return true;
}

bool detectGreenBall(Mat& frame, const Config& cfg, Point2f& out_center, vector<Point>& out_cnt)
{
    out_center = Point2f(-1, -1);
    out_cnt.clear();
    Mat hsv, mask;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    inRange(hsv, cfg.lower_green, cfg.upper_green, mask);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(cfg.morph_kernel_size, cfg.morph_kernel_size));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int h = frame.rows;
    int w = frame.cols;
    int margin_w = w * cfg.margin_w_ratio;
    int margin_h = h * cfg.margin_h_ratio;
    Point2f img_center(w/2.0, h/2.0);

    double best_score = -1;
    vector<Point> best_cnt;
    Point2f best_pt;

    for (auto& cnt : contours)
    {
        double area = contourArea(cnt);
        if (area < cfg.min_contour_area) continue;

        Moments m = moments(cnt);
        if (m.m00 < 1e-6) continue;
        Point2f c(m.m10/m.m00, m.m01/m.m00);

        if (c.x < margin_w || c.x > w - margin_w || c.y < margin_h || c.y > h - margin_h)
            continue;

        RotatedRect ell;
        try { ell = fitEllipse(cnt); } catch (...) { continue; }
        double major = ell.size.width;
        double minor = ell.size.height;
        if (minor < 1e-3) continue;
        double axis_ratio = major / minor;

        double stdv = calcContourStd(cnt, c);
        if (stdv > cfg.min_center_std) continue;

        double dist_center = hypot(c.x - img_center.x, c.y - img_center.y);
        double pos_w = 1.0 / (1.0 + dist_center / 100.0);
        double shape_w = 1.0 / (1.0 + fabs(axis_ratio - 1.0));
        double score = (shape_w * cfg.axis_ratio_weight + pos_w * cfg.center_bias_weight) * sqrt(area);

        if (score > best_score)
        {
            best_score = score;
            best_cnt = cnt;
            best_pt = c;
        }
    }

    if (best_score < 0) return false;
    out_center = best_pt;
    out_cnt = best_cnt;

    RotatedRect ell = fitEllipse(best_cnt);
    ellipse(frame, ell, Scalar(0,255,0), 2);
    circle(frame, best_pt, 8, Scalar(0,0,255), -1);
    return true;
}

// 修复 double -> float 窄化警告
Vec3d pnpSolve(Point2f center, const vector<Point>& cnt, const Config& cfg)
{
    if (center.x < 0) return Vec3d(0,0,0);
    float ball_r = static_cast<float>(cfg.ball_radius);
    vector<Point3f> obj_pts = {
        {-ball_r, 0, 0}, {ball_r, 0, 0},
        {0, -ball_r, 0}, {0, ball_r, 0},
        {0, 0, -ball_r}, {0, 0, ball_r}
    };
    vector<Point2f> img_pts;
    Point2f c; float radius;
    minEnclosingCircle(cnt, c, radius);
    img_pts.emplace_back(c.x - radius, c.y);
    img_pts.emplace_back(c.x + radius, c.y);
    img_pts.emplace_back(c.x, c.y - radius);
    img_pts.emplace_back(c.x, c.y + radius);
    img_pts.emplace_back(c.x, c.y);
    img_pts.emplace_back(c.x, c.y);

    Vec3d rvec, tvec;
    solvePnP(obj_pts, img_pts, cfg.camera_matrix, cfg.dist_coeffs, rvec, tvec);
    return tvec;
}

void drawMotion(Mat& frame, Point2f curr, Point2f prev, Point2f kf_pt, double scale)
{
    if (curr.x < 0) return;
    circle(frame, kf_pt, 6, Scalar(255,0,0), 2);
    if (prev.x < 0) return;

    double dx = curr.x - prev.x;
    double dy = curr.y - prev.y;
    Point2f end(curr.x + dx * scale, curr.y + dy * scale);
    arrowedLine(frame, curr, end, Scalar(0,255,0), 3, 0, 0, 0.3);

    double sp = hypot(dx, dy);
    putText(frame, format("Speed:%.1f", sp),
        Point(curr.x+10, curr.y-10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,255), 2);
}



int main()
{
    fs::path exe_path = __FILE__;
    fs::path root = exe_path.parent_path().parent_path();
    string cfg_path = (root / "config.json").string();
    cout << "Config path: " << cfg_path << endl;

    Config cfg;
    if (!loadConfig(cfg, cfg_path)) return -1;
    cout << "Video path in config: " << cfg.input_video << endl;

    // 创建输出目录
    fs::create_directories(fs::path(cfg.output_video).parent_path());
    string csv_path = (root / "output_data" / "frame_speed.csv").string();
    fs::create_directories(fs::path(csv_path).parent_path());

    // ========== 修复：构造函数直接指定 FFMPEG 后端，不再 set 只读属性 ==========
    VideoCapture cap(cfg.input_video, CAP_FFMPEG);
    if (!cap.isOpened())
    {
        cerr << "Open video FAILED! Path: " << cfg.input_video << endl;
        // 降级尝试：不指定后端，用默认读取
        cap.open(cfg.input_video);
        if (!cap.isOpened())
        {
            cerr << "All methods failed to open video!" << endl;
            return -1;
        }
    }
    cout << "Open video SUCCESS" << endl;

    double fps = cap.get(CAP_PROP_FPS);
    int w = cap.get(CAP_PROP_FRAME_WIDTH);
    int h = cap.get(CAP_PROP_FRAME_HEIGHT);
    cout << "FPS: " << fps << ", W: " << w << ", H: " << h << endl;

    VideoWriter out_video(cfg.output_video, VideoWriter::fourcc('m','p','4','v'), fps, Size(w,h));

    KalmanFilter2D kf(1.0/fps, cfg.kf_process_noise, cfg.kf_measure_noise);
    Point2f prev_center(-1,-1);
    vector<Point2f> track_pts;
    int lost_frames = 0;
    int frame_idx = 0;

    ofstream csv_f(csv_path, ios::out);
    csv_f << "frame_idx,time_s,speed_pixel,direction_deg\n";

    Mat frame;
    while (cap.read(frame))
    {
        double time_s = frame_idx / fps;
        Point2f curr_center(-1,-1);
        vector<Point> curr_cnt;

        // 1. 目标检测（纯视觉，无卡尔曼前置筛选）
        detectGreenBall(frame, cfg, curr_center, curr_cnt);

        // 2. PnP 3D 标注
        Vec3d tvec = pnpSolve(curr_center, curr_cnt, cfg);
        putText(frame, format("3D:%.2f,%.2f,%.2f", tvec[0], tvec[1], tvec[2]),
            Point(10,30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,0), 2);

        // ========== 修正：第一帧强制写 0 速度 ==========
        double speed = 0.0, angle = 0.0;
        if (frame_idx > 0 && prev_center.x > 0 && curr_center.x > 0)
        {
            calcSpeedDir(prev_center, curr_center, speed, angle);
        }

        // 3. 卡尔曼更新
        Point2f pred_pt = kf.predict();
        Point2f kf_pt = pred_pt;
        if (curr_center.x > 0)
        {
            kf_pt = kf.update(curr_center);
            track_pts.push_back(kf_pt);
            if ((int)track_pts.size() > cfg.track_line_length)
                track_pts.erase(track_pts.begin());
            lost_frames = 0;
        }
        else
        {
            lost_frames++;
            if (lost_frames > cfg.max_lost_frame)
            {
                kf = KalmanFilter2D(1.0/fps, cfg.kf_process_noise, cfg.kf_measure_noise);
                track_pts.clear();
                lost_frames = 0;
            }
        }

        // 绘制轨迹线
        if (cfg.show_track_line && track_pts.size() > 1)
        {
            for (size_t i = 1; i < track_pts.size(); i++)
                line(frame, track_pts[i-1], track_pts[i], Scalar(255,0,255), 2);
        }

        // 绘制运动箭头
        drawMotion(frame, curr_center, prev_center, kf_pt, cfg.speed_scale);

        // 写入 CSV
        csv_f << frame_idx << ","
              << fixed << setprecision(3) << time_s << ","
              << fixed << setprecision(2) << speed << ","
              << fixed << setprecision(2) << angle << "\n";

        // 更新状态（和 Python 对齐：先写入，再更新 prev）
        prev_center = curr_center;
        frame_idx++;

        out_video << frame;
        imshow("Ball Track C++", frame);

        if (waitKey(1) == 'q') break;
    }

    cap.release();
    out_video.release();
    csv_f.close();
    destroyAllWindows();

    cout << "C++ Track Finished!" << endl;
    cout << "Video: " << cfg.output_video << endl;
    cout << "Data:  " << csv_path << endl;
    return 0;
}
