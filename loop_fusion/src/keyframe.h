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

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"    // 鱼眼摄像头模型
#include "camodocal/camera_models/PinholeCamera.h" // 针孔相机模型
#include "utility/tic_toc.h"                       // 辅助工具
#include "utility/utility.h"                       // 辅助工具
#include "parameters.h"                            // 系统的参数
#include "ThirdParty/DBoW/DBoW2.h"                 // 第三方库，词袋
#include "ThirdParty/DVision/DVision.h"            // 支持回环检测的描述符库

#define MIN_LOOP_NUM 25 // 定义了最小回环检测的关键帧数量

using namespace Eigen;
using namespace std;
using namespace DVision;

// 用于从图像中提取BRIEF特征描述符的类
class BriefExtractor
{
public:
    // 重载运算符，用于从输入图像提取关键点和对应的BRIEF描述符
    virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
    // 构造函数，用于从文件中加载BRIEF模式
    BriefExtractor(const std::string &pattern_file);

    DVision::BRIEF m_brief; ///< 一个DVision::BRIEF对象，封装了BRIEF特征的提取方法
};

// 用于表示关键帧的类
class KeyFrame
{
public:
    KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
             vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal,
             vector<double> &_point_id, int _sequence);
    KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
             cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1> &_loop_info,
             vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors);
    bool findConnection(KeyFrame *old_kf);
    void computeWindowBRIEFPoint();
    void computeBRIEFPoint();
    // void extractBrief();
    int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);
    bool searchInAera(const BRIEF::bitset window_descriptor,
                      const std::vector<BRIEF::bitset> &descriptors_old,
                      const std::vector<cv::KeyPoint> &keypoints_old,
                      const std::vector<cv::KeyPoint> &keypoints_old_norm,
                      cv::Point2f &best_match,
                      cv::Point2f &best_match_norm);
    void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                          std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm);
    void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                const std::vector<cv::Point2f> &matched_2d_old_norm,
                                vector<uchar> &status);
    void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                   const std::vector<cv::Point3f> &matched_3d,
                   std::vector<uchar> &status,
                   Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
    void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
    void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
    void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
    void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
    void updateLoop(Eigen::Matrix<double, 8, 1> &_loop_info);

    Eigen::Vector3d getLoopRelativeT();
    double getLoopRelativeYaw();
    Eigen::Quaterniond getLoopRelativeQ();

    double time_stamp;                              ///< 时间戳
    int index;                                      ///< 全局索引
    int local_index;                                ///< 局部索引
    Eigen::Vector3d vio_T_w_i;                      ///< 视觉惯性位姿的平移向量
    Eigen::Matrix3d vio_R_w_i;                      ///< 视觉惯性位姿的旋转矩阵
    Eigen::Vector3d T_w_i;                          ///< 全局位姿的平移向量
    Eigen::Matrix3d R_w_i;                          ///< 全局位姿的旋转矩阵
    Eigen::Vector3d origin_vio_T;                   ///< 原始的vio位姿的平移向量
    Eigen::Matrix3d origin_vio_R;                   ///< 原始的vio位姿的旋转矩阵
    cv::Mat image;                                  ///< 图像
    cv::Mat thumbnail;                              ///< 图像缩略图
    vector<cv::Point3f> point_3d;                   ///< 3D点
    vector<cv::Point2f> point_2d_uv;                ///< 2D点
    vector<cv::Point2f> point_2d_norm;              ///< 2D点(归一化)
    vector<double> point_id;                        ///< 点的ID
    vector<cv::KeyPoint> keypoints;                 ///< 关键点
    vector<cv::KeyPoint> keypoints_norm;            ///< 关键点(归一化)
    vector<cv::KeyPoint> window_keypoints;          ///< 窗口内的关键点
    vector<BRIEF::bitset> brief_descriptors;        ///< BRIEF描述符
    vector<BRIEF::bitset> window_brief_descriptors; ///< 窗口内的BRIEF描述符
    bool has_fast_point;                            ///<  是否有快速点
    int sequence;                                   ///<  序列索引

    bool has_loop;                         ///< 是否有回环
    int loop_index;                        ///< 回环索引
    Eigen::Matrix<double, 8, 1> loop_info; ///< 回环信息
};
