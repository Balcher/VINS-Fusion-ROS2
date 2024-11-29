/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 * 这个文件主要用来读取参数
 *******************************************************/

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

using namespace std;

#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_ERROR RCUTILS_LOG_ERROR

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10; // 找到的默认窗口数
const int NUM_OF_F = 1000;
// #define UNIT_SPHERE_ERROR
/*
extern是一个关键字，它告诉编译器存在着一个变量或者一个函数，如果在当前编译语句的前面中没有找到相应的变量或者函数，
也会在当前文件的后面或者其它文件中定义，来看下面的例子。
*/
extern double INIT_DEPTH;      // 初始化深度值，用于三维重建
extern double MIN_PARALLAX;    // 最小视差，用于判断特征点的有效性
extern int ESTIMATE_EXTRINSIC; // 是否估计外部参数（相机之间的相对位置和方向）

extern int USE_GPU;          // 是否使用GPU加速计算
extern int USE_GPU_ACC_FLOW; // 是否使用GPU进行光流计算
extern int USE_GPU_CERES;    // 是否使用GPU进行Ceres优化

extern double ACC_N, ACC_W; // 加速度计的噪音标准差和偏置
extern double GYR_N, GYR_W; // 陀螺仪的噪音标准差和偏置

extern std::vector<Eigen::Matrix3d> RIC; // 存储相机的旋转矩阵（相机坐标系到IMU坐标系的旋转）
extern std::vector<Eigen::Vector3d> TIC; // 存储相机的平移向量（IMU坐标系到相机坐标系的平移）
extern Eigen::Vector3d G;                // 重力向量

extern double BIAS_ACC_THRESHOLD;        // 加速度计偏置的阈值，用于判断偏置是否显著
extern double BIAS_GYR_THRESHOLD;        // 陀螺仪偏置的阈值
extern double SOLVER_TIME;               // 优化求解器的时间限制
extern int NUM_ITERATIONS;               // 优化的最大迭代次数
extern std::string EX_CALIB_RESULT_PATH; // 外部标定结果的保存路径
extern std::string VINS_RESULT_PATH;     // VINS结果的保存路径
extern std::string OUTPUT_FOLDER;        // 输出文件夹的路径
extern std::string IMU_TOPIC;            // IMU数据的主题名称
extern double TD;                        // 时间延迟，用于同步IMU和相机数据
extern int ESTIMATE_TD;                  // 是否估计时间延迟
extern int ROLLING_SHUTTER;              // 是否使用滚动快门模式
extern int ROW, COL;                     // 图像的行数和列数（分辨率）
extern int NUM_OF_CAM;                   // 相机的数量
extern int STEREO;                       // 是否使用立体相机
extern int USE_IMU;                      // 是否使用IMU数据
extern int MULTIPLE_THREAD;              // 是否使用多线程处理
// pts_gt for debug purpose;
extern map<int, Eigen::Vector3d> pts_gt; // 用于调试目的的地面真实点（ground truth points）

extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC; // 两个相机图像的主题名称
extern std::string FISHEYE_MASK;               // 鱼眼镜头的掩膜，用于图像畸变校正
extern std::vector<std::string> CAM_NAMES;     // 相机名称的列表
extern int MAX_CNT;                            // 最大特征点数量
extern int MIN_DIST;                           // 特征点之间的最小距离
extern double F_THRESHOLD;                     // 特征点匹配的阈值
extern int SHOW_TRACK;                         // 是否显示跟踪结果
extern int FLOW_BACK;                          // 是否进行反向光流计算

void readParameters(std::string config_file);

// 定义了不同状态参数的大小
enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,      // 表示姿态（位置和方向）的参数大小，通常包括三维位置（x,y,z）和四元数（q）表示方向，总共7个参数
    SIZE_SPEEDBIAS = 9, // 表示速度和加速度计偏置的参数大小。通常包括三维速度（vx, vy, vz）和加速度计的偏置（ba）和陀螺仪的偏置(bg)，总共9个参数。
    SIZE_FEATURE = 1    // 特征点的参数大小
};

// 定义了状态参数在状态向量中的顺序
enum StateOrder
{
    O_P = 0,  // 位置参数的起始索引，通常是状态向量的前3个元素
    O_R = 3,  // 旋转参数（四元数）的起始索引，从状态向量的第4个元素开始。
    O_V = 6,  // 速度参数的起始索引，从状态向量的第7个参数开始
    O_BA = 9, // 加速度计偏置的起始索引，从状态向量的第10个元素开始
    O_BG = 12 // 陀螺仪偏置的起始索引，从状态向量的第13个元素开始
};

// 定义了噪声参数在噪声向量中的顺序
enum NoiseOrder
{
    O_AN = 0, // 加速度计噪声的起始索引，通常是噪声向量的前3个元素。     0,1,2
    O_GN = 3, // 陀螺仪噪声的起始索引，从噪声向量的第4个元素开始。      3,4,5
    O_AW = 6, // 加速度计偏置噪声的起始索引，从噪声向量的第7个元素开始。 6,7,8
    O_GW = 9  // 陀螺仪偏置噪声的起始索引，从噪声向量的第10个元素开始。 9,10,11
};
