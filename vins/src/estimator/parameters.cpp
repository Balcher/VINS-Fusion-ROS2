/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC; // 存储相机的旋转矩阵（相机坐标系到IMU坐标系的旋转）
std::vector<Eigen::Vector3d> TIC; // 存储相机的平移向量（IMU坐标系到相机坐标系的平移）

Eigen::Vector3d G{0.0, 0.0, 9.8}; // 重力向量

int USE_GPU;          // 是否使用GPU加速计算
int USE_GPU_ACC_FLOW; // 是否使用GPU进行光流计算
int USE_GPU_CERES;    // 是否使用GPU进行Ceres优化

double BIAS_ACC_THRESHOLD;              // 加速度计偏置的阈值，用于判断偏置是否显著
double BIAS_GYR_THRESHOLD;              // 陀螺仪偏置的阈值
double SOLVER_TIME;                     // 优化求解器的时间限制
int NUM_ITERATIONS;                     // 优化的最大迭代次数
int ESTIMATE_EXTRINSIC;                 // 是否估计外部参数（相机之间的相对位置和方向）
int ESTIMATE_TD;                        // 是否估计时间延迟
int ROLLING_SHUTTER;                    // 是否使用滚动快门模式
std::string EX_CALIB_RESULT_PATH;       // 外部标定结果的保存路径
std::string VINS_RESULT_PATH;           // VINS结果的保存路径
std::string OUTPUT_FOLDER;              // 输出文件夹的路径
std::string IMU_TOPIC;                  // IMU数据的主题名称
int ROW, COL;                           // 图像的行数和列数（分辨率）
double TD;                              // 时间延迟，用于同步IMU和相机数据
int NUM_OF_CAM;                         // 相机的数量
int STEREO;                             // 是否使用立体相机
int USE_IMU;                            // 是否使用IMU数据
int MULTIPLE_THREAD;                    // 是否使用多线程处理
map<int, Eigen::Vector3d> pts_gt;       // 用于调试目的的地面真实点（ground truth points）
std::string IMAGE0_TOPIC, IMAGE1_TOPIC; // 两个相机图像的主题名称
std::string FISHEYE_MASK;               // 鱼眼镜头的掩膜，用于图像畸变校正
std::vector<std::string> CAM_NAMES;     // 相机名称的列表
int MAX_CNT;                            // 最大特征点数量
int MIN_DIST;                           // 特征点之间的最小距离
double F_THRESHOLD;                     // 特征点匹配的阈值
int SHOW_TRACK;                         // 是否显示跟踪结果
int FLOW_BACK;                          // 是否进行反向光流计算

/**
 * @brief 从ROS 2节点中读取指定名称的参数。
 *
 * 该函数尝试从给定的节点中获取参数值，并在成功时输出日志信息。
 *
 * @tparam T 参数的类型，可以是基本数据类型或自定义类型。
 * @param n ROS 2节点的共享指针。
 * @param name 要读取的参数名称。
 * @return T 返回读取到的参数值。
 */
template <typename T>
T readParam(rclcpp::Node::SharedPtr n, std::string name)
{
    T ans;                           // 声明一个变量用于存储读取到的参数值
    if (n->get_parameter(name, ans)) // 尝试从节点中获取参数
    {
        ROS_INFO("Loaded %s: ", name); // 输出加载成功的信息
        std::cout << ans << std::endl; // 打印参数值
    }
    else
    {
        ROS_ERROR("Failed to load %s", name); // 输出错误信息
        rclcpp::shutdown();                   // 关闭ros节点
    }
    return ans; // 返回读取到的参数值
}

/**
 * @brief 用于从配置文件中读取参数并进行初始化
 *
 * @param config_file 配置文件
 */
void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(), "r"); // 尝试打开配置文件
    if (fh == NULL)
    {
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        // ROS_BREAK();
        return;
    }
    fclose(fh); // 打开文件又关闭是为了确保文件的存在性检查

    // 使用opencv的FileStorage读取配置文件
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
        return;
    }

    // 读取参数
    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_GPU = fsSettings["use_gpu"];
    USE_GPU_ACC_FLOW = fsSettings["use_gpu_acc_flow"];
    USE_GPU_CERES = fsSettings["use_gpu_ceres"];

    // IMU相关参数
    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if (USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    // 读取求解器参数
    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    // 输出路径设置
    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    // 外参估计
    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else
    {
        if (ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    // 摄像头数量
    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    if (NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0); // 确保摄像头数量正确
    }

    // 获取配置文件路径
    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    // 摄像头校准文件
    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);

    if (NUM_OF_CAM == 2)
    {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib;
        // printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);

        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    // 初始化参数
    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO("Unsynchronized sensors, online estimate time offset, initial td: %f", TD);
    else
        ROS_INFO("Synchronized sensors, fix time offset: %f", TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    if (!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    fsSettings.release();
}
