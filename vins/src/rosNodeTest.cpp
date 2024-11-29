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

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"

Estimator estimator;

queue<sensor_msgs::msg::Imu::ConstPtr> imu_buf;
queue<sensor_msgs::msg::PointCloud::ConstPtr> feature_buf;
queue<sensor_msgs::msg::Image::ConstPtr> img0_buf;
queue<sensor_msgs::msg::Image::ConstPtr> img1_buf;
std::mutex m_buf;

// header: 1403715278
/**
 * @brief 处理接收到的图像数据
 *
 * 当接收到新的图像消息时，img0_callback就会被调用
 *
 * @param img_msg 接收到的图像消息
 */
void img0_callback(const sensor_msgs::msg::Image::SharedPtr img_msg)
{
    m_buf.lock();
    // std::cout << "Left : " << img_msg->header.stamp.sec << "." << img_msg->header.stamp.nanosec << endl;
    img0_buf.push(img_msg);
    m_buf.unlock();
}

/**
 * @brief 处理接收到的图像数据
 *
 * 当接收到新的图像消息时，img1_callback就会被调用
 *
 * @param img_msg 接收到的图像消息
 */
void img1_callback(const sensor_msgs::msg::Image::SharedPtr img_msg)
{
    m_buf.lock();
    // std::cout << "Right: " << img_msg->header.stamp.sec << "." << img_msg->header.stamp.nanosec << endl;
    img1_buf.push(img_msg);
    m_buf.unlock();
}

// cv::Mat getImageFromMsg(const sensor_msgs::msg::Image::SharedPtr img_msg)
cv::Mat getImageFromMsg(const sensor_msgs::msg::Image::ConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;  // 用于存储转换后的图像数据
    if (img_msg->encoding == "8UC1") // 检查消息的编码格式
    {
        sensor_msgs::msg::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian; // 大端序还是小端序(true表示大端序)
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8); // MONO8表示灰度图
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

// extract images with same timestamp from two topics
/**
 * @brief 同步处理图像数据
 *
 * 该方法从两个缓冲区中提取图像数据，并确保它们具有相同的时间戳。
 * 如果时间戳差异超过0.003s，将丢弃较早的图像。
 *
 * @param STEREO 是否使用立体相机
 *
 */
void sync_process()
{
    while (1)
    {
        if (STEREO)
        {
            cv::Mat image0, image1;
            std_msgs::msg::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                double time0 = img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9);
                double time1 = img1_buf.front()->header.stamp.sec + img1_buf.front()->header.stamp.nanosec * (1e-9);

                // 0.003s sync tolerance
                if (time0 < time1 - 0.003)
                {
                    img0_buf.pop();
                    printf("throw img0\n");
                }
                else if (time0 > time1 + 0.003)
                {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                else
                {
                    time = img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9);
                    header = img0_buf.front()->header;
                    image0 = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image1 = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();
                    // printf("find img0 and img1\n");

                    // std::cout << std::fixed << img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9) << std::endl;
                    // assert(0);
                }
            }
            m_buf.unlock();
            if (!image0.empty())
                estimator.inputImage(time, image0, image1);
        }
        else
        {
            cv::Mat image;
            std_msgs::msg::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty())
            {
                time = img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9);
                header = img0_buf.front()->header;
                image = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
            }
            m_buf.unlock();
            if (!image.empty())
                estimator.inputImage(time, image);
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

/**
 * @brief 用于处理接收到的IMU数据
 *
 * 当接收到新的imu消息时，imu_callback就会被调用
 *
 * @param imu_msg 接收到的IMU消息
 */
void imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu_msg)
{
    // std::cout << "IMU cb" << std::endl;

    double t = imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * (1e-9); // 提取时间戳
    // 提取线性加速度的x、y、z分量
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    // 提取角速度的x、y、z分量
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    // 将线性加速度和角速度封装成Eigen的向量类型
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);

    // std::cout << "got t_imu: " << std::fixed << t << endl;
    // 将数据传递给估计器
    estimator.inputIMU(t, acc, gyr);
    return;
}

/**
 * @brief 处理接收到的特征点数据，然后将这些信息传递给估计器
 *
 * 当接收到新的特征点消息时，feature_callback就会被调用
 *
 * @param feature_msg 接收到的特征点消息
 */
void feature_callback(const sensor_msgs::msg::PointCloud::SharedPtr feature_msg)
{
    // 打印回调函数被调用的消息
    std::cout << "feature cb" << std::endl;
    // 打印接收到的特征点数量
    std::cout << "Feature: " << feature_msg->points.size() << std::endl;

    // 创建一个存储特征点信息的映射表
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    // 遍历接收到的特征点消息
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];    // 获取特征点的ID
        int camera_id = feature_msg->channels[1].values[i];     // 获取特征点所在的相机ID
        double x = feature_msg->points[i].x;                    // 获取特征点的x坐标
        double y = feature_msg->points[i].y;                    // 获取特征点的y坐标
        double z = feature_msg->points[i].z;                    // 获取特征点的z坐标
        double p_u = feature_msg->channels[2].values[i];        // 获取特征点在图像中的u坐标
        double p_v = feature_msg->channels[3].values[i];        // 获取特征点在图像中的v坐标
        double velocity_x = feature_msg->channels[4].values[i]; // 获取特征点的速度在u方向上的分量
        double velocity_y = feature_msg->channels[5].values[i]; // 获取特征点的速度在v方向上的分量
        // 如果消息中包含了更多的通道
        if (feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];   // 获取特征点的重力加速度在u方向上的分量
            double gy = feature_msg->channels[7].values[i];   // 获取特征点的重力加速度在v方向上的分量
            double gz = feature_msg->channels[8].values[i];   // 获取特征点的重力加速度在z方向上的分量
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz); // 将特征点的重力加速度存储到pts_gt映射表中
            // printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        assert(z == 1);                                                    // 检查z坐标是否为1
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;                       // 创建一个7x1的矩阵来存储特征点的信息（位置、速度、像素坐标）
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;      // 存储
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity); // 将特征点的信息存储到featureFrame映射表中
    }
    double t = feature_msg->header.stamp.sec + feature_msg->header.stamp.nanosec * (1e-9); // 获取时间戳
    estimator.inputFeature(t, featureFrame);                                               // 将特征点信息传递给估计器
    return;
}

/**
 * @brief 处理接收到的重启消息，重启估计器
 *
 * 当接收到新的重启消息时，restart_callback就会被调用
 *
 * @param restart_msg 接收到的重启消息
 */
void restart_callback(const std_msgs::msg::Bool::SharedPtr restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();   // 清空估计器的状态
        estimator.setParameter(); // 重新设置估计器的参数
    }
    return;
}

/**
 * @brief 处理接收到的IMU启用/禁用消息，切换IMU的使用状态
 *
 * 当接收到新的IMU启用/禁用消息时，imu_switch_callback就会被调用
 *
 * @param switch_msg 接收到的IMU启动/禁用消息
 */
void imu_switch_callback(const std_msgs::msg::Bool::SharedPtr switch_msg)
{
    if (switch_msg->data == true)
    {
        // ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO); // 启用IMU，并设置传感器类型为STEREO
    }
    else
    {
        // ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO); // 禁用IMU，并设置传感器类型为STEREO
    }
    return;
}

/**
 * @brief 处理接收到的相机启动/禁用消息，切换相机的使用状态
 *
 * 当接收到新的相机启动/禁用消息时，cam_switch_callback就会被调用
 *
 * @param switch_msg 接收到的相机启用 / 禁用消息
 */
void cam_switch_callback(const std_msgs::msg::Bool::SharedPtr switch_msg)
{
    if (switch_msg->data == true)
    {
        // ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    }
    else
    {
        // ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}

int main(int argc, char **argv)
{
    // 初始化ROS
    rclcpp::init(argc, argv);
    // 创建节点
    auto n = rclcpp::Node::make_shared("vins_estimator");
    // 创建日志级别
    // ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    // 检查输入参数个数是否正确
    if (argc != 2)
    {
        printf("please intput: rosrun vins vins_node [config file] \n"
               "for example: rosrun vins vins_node "
               "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 1;
    }

    // 获取配置文件路径
    string config_file = argv[1];
    // 打印配置文件路径
    printf("config_file: %s\n", argv[1]);

    // 读取配置文件
    readParameters(config_file);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu = NULL; // 创建了一个IMU消息类型的订阅者指针 sub_imu,初始值为 NULL
    if (USE_IMU)                                                           // 如果定义为true，则创建IMU订阅者
    {
        sub_imu = n->create_subscription<sensor_msgs::msg::Imu>(IMU_TOPIC, rclcpp::QoS(rclcpp::KeepLast(2000)), imu_callback); // 当订阅者接收到IMU消息时，会调用 imu_callback 函数进行处理
    }
    auto sub_feature = n->create_subscription<sensor_msgs::msg::PointCloud>("/feature_tracker/feature", rclcpp::QoS(rclcpp::KeepLast(2000)), feature_callback); // 订阅feature_tracker/feature主题上的点云数据
    auto sub_img0 = n->create_subscription<sensor_msgs::msg::Image>(IMAGE0_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)), img0_callback);                           // 订阅名为 IMAGE0_TOPIC 的图像主题(sensor_msgs::msg::Image)，用于接收来自相机的图像数据

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img1 = NULL; // 创建了一个图像消息类型的订阅者指针 sub_img1,初始值为 NULL
    if (STEREO)                                                               // 如果宏STEREO被定义为true，则创建第二个图像订阅者
    {
        sub_img1 = n->create_subscription<sensor_msgs::msg::Image>(IMAGE1_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)), img1_callback); // 订阅名为 IMAGE1_TOPIC 的图像主题(sensor_msgs::msg::Image)，用于接收来自相机的图像数据
    }

    auto sub_restart = n->create_subscription<std_msgs::msg::Bool>("/vins_restart", rclcpp::QoS(rclcpp::KeepLast(100)), restart_callback);          // 订阅/vins_restart主题上的布尔类型消息，用于接收重启信号
    auto sub_imu_switch = n->create_subscription<std_msgs::msg::Bool>("/vins_imu_switch", rclcpp::QoS(rclcpp::KeepLast(100)), imu_switch_callback); // 订阅/vins_imu_switch主题上的布尔类型消息，用于接收IMU开关信号
    auto sub_cam_switch = n->create_subscription<std_msgs::msg::Bool>("/vins_cam_switch", rclcpp::QoS(rclcpp::KeepLast(100)), cam_switch_callback); // 订阅/vins_cam_switch主题上的布尔类型消息，用于接收相机开关信号

    std::thread sync_thread{sync_process};
    rclcpp::spin(n);

    return 0;
}
