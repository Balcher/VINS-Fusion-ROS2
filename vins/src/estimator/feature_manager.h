/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

// #include <ros/console.h>
#include <rcpputils/asserts.hpp>

#include "parameters.h"
#include "../utility/tic_toc.h"

// 日志宏定义
#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_DEBUG RCUTILS_LOG_DEBUG
#define ROS_ERROR RCUTILS_LOG_ERROR

// 该类表示每一帧中的特征点
// 包含特征点的三维坐标、图像坐标（uv）、速度和时间戳。
// rightObservation 方法用于处理双目视觉中的右图特征点。
class FeaturePerFrame
{
public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);    // 特征点的x坐标（世界坐标系）
        point.y() = _point(1);    // 特征点的y坐标（世界坐标系）
        point.z() = _point(2);    // 特征点的z坐标（世界坐标系）
        uv.x() = _point(3);       // 特征点在图像中的 x 坐标（像素坐标）。
        uv.y() = _point(4);       // 特征点在图像中的 y 坐标（像素坐标）。
        velocity.x() = _point(5); // 特征点的 x 方向速度。
        velocity.y() = _point(6); // 特征点的 y 方向速度。
        cur_td = td;              // 时间差值，通常表示当前帧与上一帧之间的时间间隔。
        is_stereo = false;        // 表示该特征点最初不被认为是立体观测的（即只有单目视图）。
    }
    // 7x1的Eigen矩阵，包含了右侧图像中对应特征点的多种信息
    void rightObservation(const Eigen::Matrix<double, 7, 1> &_point)
    {
        pointRight.x() = _point(0);    // 右图特征点的 x 坐标（世界坐标系）。
        pointRight.y() = _point(1);    // 右图特征点的 y 坐标（世界坐标系）。
        pointRight.z() = _point(2);    // 右图特征点的 z 坐标（世界坐标系）。
        uvRight.x() = _point(3);       // 右图特征点在图像中的 x 坐标（像素坐标）。
        uvRight.y() = _point(4);       // 右图特征点在图像中的 y 坐标（像素坐标）。
        velocityRight.x() = _point(5); // 右图特征点的 x 方向速度。
        velocityRight.y() = _point(6); // 右图特征点的 y 方向速度。
        is_stereo = true;              // 表示该特征点现在被认为是立体观测的
    }
    double cur_td;                    //  记录当前帧与上一帧之间的时间差（时间间隔）
    Vector3d point, pointRight;       //  表示特征点在三维空间中的位置（世界坐标系）,右图像中对应特征点在三维空间中的位置
    Vector2d uv, uvRight;             // 特征点在左图像中的像素坐标（x, y）,特征点在右图像中的像素坐标（x, y）
    Vector2d velocity, velocityRight; // 特征点在左图像中的速度（x, y）, 特征点在右图像中的速度（x, y）
    bool is_stereo;                   // 当 is_stereo 为 true 时，表示该特征点在左右图像中都有观测数据
};

// 表示与特征点相关的信息，包括特征点的ID、起始帧、在多帧中的特征信息
class FeaturePerId
{
public:
    const int feature_id;                      // 特征点的唯一标识
    int start_frame;                           // 特征点首次出现的帧编号
    vector<FeaturePerFrame> feature_per_frame; // 存储与该特征点相关的多个帧的信息，每个元素是一个FeaturePerFrame对象
    int used_num;                              // 表示当前已使用的观测数量
    double estimated_depth;                    // 估计的深度值，初始值-1,0表示未估计，用于存储从立体视觉或其他深度估计方法计算得到的深度信息
    int solve_flag;                            // 0 haven't solve yet; 1 solve succ; 2 solve fail; 表示求解状态的标志位

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();
};

// 用于管理特征点的跟踪、深度估计和位姿计算等功能。
class FeatureManager
{
public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);
    void clearState();
    int getFeatureCount();
    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);
    // void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth();
    VectorXd getDepthVector();
    void triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                          Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
    void initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial,
                        vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier(set<int> &outlierIndex);

    list<FeaturePerId> feature;   // 存储特征点的列表，每个元素是一个 FeaturePerId 对象
    int last_track_num;           // 上一次跟踪的特征点数量
    double last_average_parallax; // 上一次平均视差值
    int new_feature_num;          // 新增特征点的数量
    int long_track_num;           // 长期跟踪的特征点数量

private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count); // 计算视差的补偿值，用于评估特征点的有效性
    const Matrix3d *Rs;                                                          // 指向旋转矩阵的指针，可能用于存储相机的旋转信息
    Matrix3d ric[2];                                                             // 存储两个相机的旋转信息
};

#endif