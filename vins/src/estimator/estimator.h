/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <thread>
#include <mutex>
#include <std_msgs/msg/header.h>
#include <std_msgs/msg/float32.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "parameters.h"
#include "feature_manager.h"
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../initial/solve_5pts.h"
#include "../initial/initial_sfm.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../factor/imu_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../featureTracker/feature_tracker.h"

#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_ERROR RCUTILS_LOG_ERROR

/**
 * @brief 估计器类，处理以下任务
 * 1. 融合视觉和惯性数据（IMU和图像数据）
 * 2. 执行状态估计，包括位置、速度、姿态和传感器偏置等
 * 3. 进行窗口优化（如划窗法）
 * 4. 初始化系统，处理外参、内参和时间偏移
 */
class Estimator
{
public:
    Estimator();         ///< 构造函数，初始化对象时调用，通常用于初始化成员变量
    ~Estimator();        ///< 析构函数，对象销毁时调用，通常用于释放资源
    void setParameter(); ///< 设置参数

    // interface
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);                                                      ///< 初始化第一个位姿（位置P和旋转r）
    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);                  ///< 接收IMU的加速度和角速度数据，并存入缓冲区
    void inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame);     ///< 接收视觉特征点（如点云或关键点）的数据
    void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());                              ///< 接收图像数据，通常用于处理视觉特征点
    void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);   /// 处理IMU数据，进行状态估计
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header); /// 处理图像数据，通常用于处理视觉特征点
    void processMeasurements();                                                                                    /// 处理测量数据，通常用于执行状态估计和窗口优化
    void changeSensorType(int use_imu, int use_stereo);                                                            /// 更改传感器类型，如使用IMU和相机

    // internal
    void clearState();                                                     ///< 清除状态，通常用于重置系统状态
    bool initialStructure();                                               ///< 初始化结构，通常用于初始化系统的初始状态
    bool visualInitialAlign();                                             ///< 视觉初始对齐，通常用于初始化系统的初始状态
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l); ///< 计算相对姿态，通常用于计算两个帧之间的相对姿态
    void slideWindow();                                                    ///< 滑动窗口，通常用于维护系统的窗口大小
    void slideWindowNew();                                                 ///< 滑动窗口，通常用于维护系统的窗口大小
    void slideWindowOld();                                                 ///< 滑动窗口，通常用于维护系统的窗口大小
    void optimization();                                                   ///< 执行优化，通常用于优化系统的状态
    void vector2double();                                                  ///< 将状态转换为双精度数组，通常用于优化过程
    void double2vector();                                                  ///< 将双精度数组转换为状态，通常用于优化过程
    bool failureDetection();                                               ///< 检测故障，通常用于检测系统的故障
    bool getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
                        vector<pair<double, Eigen::Vector3d>> &gyrVector); ///< 从IMU缓冲区中获取指定时间范围内的加速度和角速度数据
    void getPoseInWorldFrame(Eigen::Matrix4d &T);                          ///< 获取当前帧的位姿（位置P和旋转r）
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);               ///< 获取指定帧的位姿（位置P和旋转r）
    void predictPtsInNextFrame();                                          ///< 预测下一帧的特征点位置
    void outliersRejection(set<int> &removeIndex);                         ///< 执行异常值检测，通常用于检测异常值并进行处理
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                             Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                             double depth, Vector3d &uvi, Vector3d &uvj);                                 ///< 计算特征点在两个帧之间的重投影误差
    void updateLatestStates();                                                                            ///< 更新最新状态
    void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity); ///< 快速预测IMU数据
    bool IMUAvailable(double t);                                                                          ///< 检查IMU数据是否可用
    void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector);                              ///< 初始化第一个IMU姿态

    // ==== 枚举类型 ====
    // 表示估计器当前的解算状态
    enum SolverFlag
    {
        INITIAL,   ///< 初始状态
        NON_LINEAR ///< 非线性状态
    };

    // 表示划窗边缘化策略
    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,       ///< 旧的边缘化策略
        MARGIN_SECOND_NEW = 1 ///< 新的边缘化策略
    };

    std::mutex mProcess;   ///< 用于线程同步的互斥锁
    std::mutex mBuf;       ///< 用于线程同步的互斥锁
    std::mutex mPropagate; ///< 用于线程同步的互斥锁

    // ==== 缓冲区 ====
    queue<pair<double, Eigen::Vector3d>> accBuf;                                              ///< 用于存储加速度数据的队列
    queue<pair<double, Eigen::Vector3d>> gyrBuf;                                              ///< 用于存储角速度数据的队列
    queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>>> featureBuf; ///< 用于存储特征数据的队列

    // ==== 状态变量 ====
    double prevTime, curTime; ///< 上一帧和当前帧的时间戳
    bool openExEstimation;    ///< 是否进行外部估计

    std::thread trackThread;   ///< 用于跟踪图像的线程
    std::thread processThread; ///< 用于处理数据的线程

    FeatureTracker featureTracker; ///< 用于跟踪图像的特征跟踪器

    SolverFlag solver_flag;                   ///< 表示解算状态的枚举值
    MarginalizationFlag marginalization_flag; ///< 表示边缘化策略的枚举值
    Vector3d g;

    Matrix3d ric[2]; ///< 表示相机的旋转矩阵
    Vector3d tic[2]; ///< 表示相机的平移向量

    // ==== 状态变量 ====
    // Ps、Vs 和 Rs：窗口中每一帧的平移、速度和旋转矩阵。
    // Bas 和 Bgs：窗口中每一帧的加速度计和陀螺仪的偏差向量。
    Vector3d Ps[(WINDOW_SIZE + 1)];  ///< 表示位置的向量数组
    Vector3d Vs[(WINDOW_SIZE + 1)];  ///< 表示速度的向量数组
    Matrix3d Rs[(WINDOW_SIZE + 1)];  ///< 表示旋转矩阵的向量数组
    Vector3d Bas[(WINDOW_SIZE + 1)]; ///< 表示加速度计的偏差向量数组
    Vector3d Bgs[(WINDOW_SIZE + 1)]; ///< 表示陀螺仪的偏差向量数组
    double td;                       ///< 表示时间偏差

    Matrix3d back_R0, last_R, last_R0; ///< 表示旋转矩阵的向量数组
    Vector3d back_P0, last_P, last_P0; ///< 表示位置的向量数组
    double Headers[(WINDOW_SIZE + 1)]; ///< 表示时间偏差

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)]; ///< 表示时间偏差
    Vector3d acc_0, gyr_0;                                ///< 表示时间偏差

    vector<double> dt_buf[(WINDOW_SIZE + 1)];                    ///< 表示时间偏差
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)]; ///< 表示时间偏差
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];    ///< 表示时间偏差

    int frame_count;                                               ///< 当前滑动窗口中的帧数计数器
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid; ///< 累计的外点（Outliers）数量、累计的向后跟踪的点数量、累计的向前跟踪的点数量、累计的无效点数量
    int inputImageCnt;                                             ///< 累计输入的图像帧数量

    FeatureManager f_manager;              ///< 特征管理器，负责管理滑动窗口内的所有特征点
    MotionEstimator m_estimator;           ///< 运动估计器，用于估计运动状态
    InitialEXRotation initial_ex_rotation; ///< 外参初始旋转估计器

    bool first_imu;        ///< 标志是否接收到第一帧IMU数据
    bool is_valid, is_key; ///< 当前估计是否有效的标志、当前帧是否为关键帧
    bool failure_occur;    ///< 系统失败检测的标志

    vector<Vector3d> point_cloud;  ///< 当前帧或滑动窗口中所有的三维点云
    vector<Vector3d> margin_cloud; ///< 边缘化过程中生成的三维点云
    vector<Vector3d> key_poses;    ///< 关键帧的位置信息
    double initial_timestamp;      ///< 记录系统启动时的初始时间戳

    // ==== 滑窗优化类型 ====
    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];           ///< 存储滑动窗口中每帧的位姿参数
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS]; ///< 存储每帧的速度和IMU的偏置参数
    double para_Feature[NUM_OF_F][SIZE_FEATURE];            ///< 存储特征点的参数
    double para_Ex_Pose[2][SIZE_POSE];                      ///< 存储外参矩阵的参数
    double para_Retrive_Pose[SIZE_POSE];                    ///< 存储一个独立的位姿，用于检索或单独处理
    double para_Td[1][1];                                   ///< 存储时间延迟参数
    double para_Tr[1][1];                                   ///< 存储传感器间的尺度因子（或缩放参数）

    int loop_window_index; ///< 回环检测窗口的索引

    MarginalizationInfo *last_marginalization_info;         ///< 上一轮边缘化的相关信息
    vector<double *> last_marginalization_parameter_blocks; ///< 上一轮边缘化时的参数块

    map<double, ImageFrame> all_image_frame; ///< 存储所有图像帧的集合，按时间戳排序
    IntegrationBase *tmp_pre_integration;    ///< 临时的前积分数据

    Eigen::Vector3d initP; ///< 初始位置向量
    Eigen::Matrix3d initR; ///< 初始旋转矩阵

    double latest_time;                                                                   ///< 最新的时间戳
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0; ///< 系统当前的位置信息、速度、加速度计和陀螺仪的偏置、初始加速度和陀螺仪的零值
    Eigen::Quaterniond latest_Q;                                                          ///< 最新的旋转四元数

    bool initFirstPoseFlag; ///< 标志位，用于指示是否已经初始化了第一个位姿
    bool initThreadFlag;    ///< 标志位，指示是否已初始化线程
};
