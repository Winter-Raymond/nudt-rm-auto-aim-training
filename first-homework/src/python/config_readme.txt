========== 配置文件说明 config.json ==========
1. path 路径配置
   input_video : 输入视频路径
   output_video: 输出视频保存路径

2. camera 相机与小球物理参数(PnP使用)
   camera_matrix: 相机内参矩阵
   dist_coeffs  : 畸变系数
   ball_radius  : 小球半径(单位:米)

3. color_detect 颜色与形态学参数
   lower_green/upper_green : 绿色HSV阈值
   morph_kernel_size       : 形态学卷积核大小
   min_contour_area        : 最小轮廓面积(过滤极小噪点)

4. shape_filter 形状筛选(区分球/圆柱、抗出框)
   min_center_std      : 轮廓点到中心距离标准差阈值
   axis_ratio_weight   : 椭圆轴比打分权重
   center_bias_weight  : 画面中心位置权重
   tracking_bias_weight: 跟踪轨迹权重(防跳变)
   max_tracking_dist   : 允许目标偏离卡尔曼预测点最大像素距离

5. kalman 卡尔曼滤波参数
   process_noise    : 过程噪声
   measurement_noise: 观测噪声
   max_lost_frame   : 连续丢失帧数上限，超过自动重置滤波

6. visual 可视化参数
   speed_scale      : 运动箭头长度缩放系数
   show_track_line  : 是否绘制轨迹线
   track_line_length: 轨迹最大点数

7. roi_margin 有效区域比例
   margin_w_ratio/margin_h_ratio : 画面边缘留白比例，放宽可支持物体出框
