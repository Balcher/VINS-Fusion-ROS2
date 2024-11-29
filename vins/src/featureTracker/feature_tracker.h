/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

// #define GPU_MODE 1

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#ifdef GPU_MODE
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_DEBUG RCUTILS_LOG_DEBUG
#define ROS_ERROR RCUTILS_LOG_ERROR

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

/**
 * @brief 用于在视觉slam或相关视觉处理任务中对图像特征点进行跟踪和处理。
 *
 * 主要的目标是从连续帧中检测、跟踪和管理特征点，以实现视觉里程计、特征跟踪和地图构建等功能。
 * 主要功能：
 * 1. 特征点检测和跟踪：管理图像中关键点的提取、跟踪、滤波（如去除运动外点）
 * 2. 特征点去畸变：通过相机内参去畸变特征点的像素坐标
 * 3. 特征点预测：结合历史数据和预测算法，推测下一帧特征点的位置
 * 4. 特征点可视化：显示跟踪效果以便调试和验证
 */
class FeatureTracker
{
public:
    // ==== 构造函数 ====
    FeatureTracker();
    // ==== 核心函数 ====
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    // ==== 特征点操作 ====
    void setMask();                                                                         ///< 设置掩膜，用于限制特征点的分布
    void addPoints();                                                                       ///< 在图像中添加新的特征点
    void readIntrinsicParameter(const vector<string> &calib_file);                          ///< 读取相机内参文件，用于相机模型初始化
    void undistortedPoints();                                                               ///< 对当前帧的特征点进行去畸变处理
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam); ///< 对输入的特征点列表 pts 进行去畸变
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts); ///< 计算特征点的速度（光流）
    void removeOutliers(set<int> &removePtsIds);                                                            ///< 从特征点列表中移除异常点（如跟踪失败、运动异常等）

    // ==== 可视化和调试 ====
    void showUndistortion(const string &name); ///< 显示当前图像的去畸变结果，用于调试
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2,
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2); ///< 显示左右两张图像及其特征点，用于调试双目特征匹配
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                   vector<int> &curLeftIds,
                   vector<cv::Point2f> &curLeftPts,
                   vector<cv::Point2f> &curRightPts,
                   map<int, cv::Point2f> &prevLeftPtsMap); ///< 绘制特征点的跟踪轨迹（主图像和右图像）
    cv::Mat getTrackImage();                               ///< 获取绘制了特征点轨迹的图像，用于可视化和调试

    // ===== 特征点的几何操作 =====
    void rejectWithF();                                        ///< 使用基本矩阵（Fundamental Matrix）剔除错误匹配的特征点
    void setPrediction(map<int, Eigen::Vector3d> &predictPts); ///< 设置特征点的预测位置，用于优化跟踪性能
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);       ///< 计算两点之间的欧几里得距离
    bool inBorder(const cv::Point2f &pt);                      ///< 判断点是否在图像边界内

    int row, col;                                                      ///< 图像的行和列
    cv::Mat imTrack;                                                   ///< 用于存储特征跟踪的图像
    cv::Mat mask;                                                      ///< 动态掩膜，涌现限制特征点的分布（例如避免特征点密集在某一区域）
    cv::Mat fisheye_mask;                                              ///< 鱼眼掩膜，限制鱼眼畸变区域内的点
    cv::Mat prev_img, cur_img;                                         ///< 存储前一帧和当前帧的图像
    vector<cv::Point2f> n_pts;                                         ///< 当前帧中新检测到的特征点（尚未跟踪的点）
    vector<cv::Point2f> predict_pts;                                   ///< 当前帧中特征点的预测位置（基于运动模型）
    vector<cv::Point2f> predict_pts_debug;                             ///< 调试用的预测点数据
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;              ///< 上一帧图像中的特征点、当前帧图像中的特征点、当前帧右图像（用于双目相机）中的特征点
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;     ///< 上一帧图像中的特征点、当前帧图像中的特征点、当前帧右图像（用于双目相机）中的特征点
    vector<cv::Point2f> pts_velocity, right_pts_velocity;              ///< 特征点的速度（基于运动模型）、右图像中的特征点的速度（基于运动模型）
    vector<int> ids, ids_right;                                        ///< 主图像的特征点ID、右图像的特征点ID
    vector<int> track_cnt;                                             ///< 跟踪计数，用于跟踪特征点的连续出现次数
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;             ///< 当前帧中每个特征点（以 ID 为键）的去畸变坐标、上一帧中每个特征点的去畸变坐标
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map; ///< 当前帧右图像中特征点的去畸变坐标、上一帧右图像中特征点的去畸变坐标
    map<int, cv::Point2f> prevLeftPtsMap;                              ///< 上一帧左图像中每个特征点的位置，用于双目特征匹配或验证
    vector<camodocal::CameraPtr> m_camera;                             ///< 相机模型的指针
    double cur_time;                                                   ///< 当前帧的时间戳
    double prev_time;                                                  ///< 上一帧的时间戳
    bool stereo_cam;                                                   ///< 指示是否是双目相机系统
    int n_id;                                                          ///< 当前最大特征点ID，用于分配新特征点的唯一标识
    bool hasPrediction;                                                ///< 指示当前帧是否有预测点位信息（通常基于运动模型）
};
