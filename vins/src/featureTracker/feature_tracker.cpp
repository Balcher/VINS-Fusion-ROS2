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

#include "feature_tracker.h"

bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double distance(cv::Point2f pt1, cv::Point2f pt2)
{
    // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

/**
 * @brief 从一个整数向量中移除所有标记为 0 的元素
 *
 * @param v
 * @param status
 */
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

/**
 * @brief 从一个整数向量中移除所有标记为 0 的元素
 *
 * @param v
 * @param status
 */
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

/**
 * @brief Construct a new Feature Tracker:: Feature Tracker object
 *
 * 该构造函数用于初始化 FeatureTracker 类的成员变量，
 * 包括 stereo_cam、n_id 和 hasPrediction
 */
FeatureTracker::FeatureTracker()
{
    stereo_cam = 0;        // 初始化 stereo_cam 变量为 0,表示当前未使用立体相机
    n_id = 0;              // 初始化 n_id 变量为 0，用于为跟踪的特征点分配唯一的 ID
    hasPrediction = false; // 初始化 hasPrediction 变量为 false，表示当前没有预测的特征点
}

/**
 * @brief 设置掩码，用于标记已经被特征点占据的区域
 *
 * 这个方法根据特征点的跟踪时间来设置一个掩码，
 * 优先保留那些跟踪时间较长的特征点，并在图像中标记它们的位置。
 */
void FeatureTracker::setMask()
{
    // 创建一个与当前图像大小相同的掩码，初始值为 255（白色）
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    // 在图像中标记跟踪时间较长的特征点
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    // 将特征点的跟踪时间和位置存储到 cnt_pts_id 中
    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    // 按照跟踪次数对特征点进行降序排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         { return a.first > b.first; });

    // 清空当前帧的特征点、ID和跟踪次数向量
    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    // 遍历排序后的特征点信息
    for (auto &it : cnt_pts_id)
    {
        // 如果掩码中对应位置的值为255（白色），则将该特征点添加到当前帧的特征点列表中
        if (mask.at<uchar>(it.second.first) == 255)
        {
            cur_pts.push_back(it.second.first);                 // 将特征点的位置添加到当前帧的特征点列表中
            ids.push_back(it.second.second);                    // 将特征点的ID添加到当前帧的ID列表中
            track_cnt.push_back(it.first);                      // 将特征点的跟踪次数添加到当前帧的跟踪次数列表中
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1); // 在掩码中标记该特征点的位置为黑色（0），表示该位置已被占据
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        cur_pts.push_back(p);   // 当前帧图像中的特征点
        ids.push_back(n_id++);  // 主图像的特征点ID
        track_cnt.push_back(1); // 跟踪计数，用于跟踪特征点的连续出现次数
    }
}

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

/**
 * @brief 跟踪图像中的特征点
 *
 * 该方法使用光流法跟踪图像中的特征点，并返回跟踪结果
 *
 * @param _cur_time 当前时间
 * @param _img 当前图像
 * @param _img1 另一张图像，如果是单目相机，则为空
 * @return map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> 跟踪结果
 */
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    // ==== 变量初始化 ====
    TicToc t_r;               // 计时器，用于测量函数执行时间
    cur_time = _cur_time;     // 当前时间
    cur_img = _img;           // 当前图像
    row = cur_img.rows;       // 图像的行数
    col = cur_img.cols;       // 图像的列数
    cv::Mat rightImg = _img1; // 另一张图像，如果是单目相机，则为空
    /* // 这部分代码用于图像增强，CLAHE（对比度受限自适应直方图均衡）来增强图像的对比度，以提高图像的视觉效果
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    */
    cur_pts.clear(); // 清空当前帧的特征点

    // ==== 光流跟踪 ====
    if (prev_pts.size() > 0) // 如果前一帧的特征点存在，则进行跟踪
    {
        vector<uchar> status;  // 记录每个特征点是否成功跟踪的标记
        if (!USE_GPU_ACC_FLOW) // 不使用GPU加速的光流跟踪
        {
            TicToc t_o; // 计时器，用于测量光流跟踪的执行时间

            vector<float> err; // 用于存储每个特征点的光流跟踪误差
            if (hasPrediction) // 检查是否有预测的特征点
            {
                cur_pts = predict_pts; // 如果有预测，更新当前帧的特征点 cur_pts 为预测的特征点 predict_pts
                // 计算光流跟踪结果，使用金字塔光流法（Pyramid Lucas-Kanade），计算从前一帧图像到当前帧图像的特征点位移
                cv::calcOpticalFlowPyrLK(prev_img,                                                                    // 前一帧图像
                                         cur_img,                                                                     // 当前帧图像
                                         prev_pts,                                                                    // 前一帧特征点
                                         cur_pts,                                                                     // 当前帧特征点
                                         status,                                                                      // 状态标记
                                         err,                                                                         // 光流计算的误差
                                         cv::Size(21, 21),                                                            // 窗口大小
                                         1,                                                                           // 金字塔层数
                                         cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), // 终止条件（最多迭代30次，每次迭代的误差下降小于0.01时停止）
                                         cv::OPTFLOW_USE_INITIAL_FLOW);                                               // 使用初始光流作为初始估计

                int succ_num = 0; // 定义一个变量succ_num，用于统计成功追踪的特征点的数量
                // 遍历 status 向量，统计成功追踪的特征点数量
                for (size_t i = 0; i < status.size(); i++)
                {
                    // status[i]为1 表示成功追踪，为 0 表示失败追踪
                    if (status[i])
                        succ_num++;
                }
                // 如果成功追踪的特征点数量小于10，重新进行光流跟踪，使用更高层数（3层）来重新计算光流，增加计算的鲁棒性
                if (succ_num < 10)
                    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
            }
            else // 如果没有预测特征点，直接使用当前帧和前一帧进行光流计算
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
            // reverse check
            if (FLOW_BACK) // 检查是否需要进行反向光流验证（reverse check）
            {
                vector<uchar> reverse_status;               // 创建一个reverse_status向量，用于存储反向光流验证的结果,表示每个特征点在反向光流验证中是否成功追踪
                vector<cv::Point2f> reverse_pts = prev_pts; // 将 prev_pts 赋值给 reverse_pts，作为反向追踪的输入
                // 执行反向光流计算，即从当前帧图像 cur_img 到前一帧图像 prev_img 计算光流，验证特征点的有效性
                cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1,
                                         cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                // cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3);
                // 遍历所有特征点，检查它们在反向追踪中的状态，进行验证
                for (size_t i = 0; i < status.size(); i++)
                {
                    // 如果正向和反向的追踪都成功，并且两个方向的位移距离小于等于0.5，将该特征点的状态设置为1（成功追踪）
                    if (status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                    {
                        status[i] = 1;
                    }
                    else
                        status[i] = 0; // 否则，将该特征点的状态设置为0（失败追踪）
                }
            }
            // printf("temporal optical flow costs: %fms\n", t_o.toc());
        }
#ifdef GPU_MODE
        else
        {
            TicToc t_og;
            cv::cuda::GpuMat prev_gpu_img(prev_img);
            cv::cuda::GpuMat cur_gpu_img(cur_img);
            cv::cuda::GpuMat prev_gpu_pts(prev_pts);
            cv::cuda::GpuMat cur_gpu_pts(cur_pts);
            cv::cuda::GpuMat gpu_status;
            if (hasPrediction)
            {
                cur_gpu_pts = cv::cuda::GpuMat(predict_pts);
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                    cv::Size(21, 21), 1, 30, true);
                d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);

                vector<cv::Point2f> tmp_cur_pts(cur_gpu_pts.cols);
                cur_gpu_pts.download(tmp_cur_pts);
                cur_pts = tmp_cur_pts;

                vector<uchar> tmp_status(gpu_status.cols);
                gpu_status.download(tmp_status);
                status = tmp_status;

                int succ_num = 0;
                for (size_t i = 0; i < tmp_status.size(); i++)
                {
                    if (tmp_status[i])
                        succ_num++;
                }
                if (succ_num < 10)
                {
                    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                        cv::Size(21, 21), 3, 30, false);
                    d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);

                    vector<cv::Point2f> tmp1_cur_pts(cur_gpu_pts.cols);
                    cur_gpu_pts.download(tmp1_cur_pts);
                    cur_pts = tmp1_cur_pts;

                    vector<uchar> tmp1_status(gpu_status.cols);
                    gpu_status.download(tmp1_status);
                    status = tmp1_status;
                }
            }
            else
            {
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                    cv::Size(21, 21), 3, 30, false);
                d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);

                vector<cv::Point2f> tmp1_cur_pts(cur_gpu_pts.cols);
                cur_gpu_pts.download(tmp1_cur_pts);
                cur_pts = tmp1_cur_pts;

                vector<uchar> tmp1_status(gpu_status.cols);
                gpu_status.download(tmp1_status);
                status = tmp1_status;
            }
            if (FLOW_BACK)
            {
                cv::cuda::GpuMat reverse_gpu_status;
                cv::cuda::GpuMat reverse_gpu_pts = prev_gpu_pts;
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                    cv::Size(21, 21), 1, 30, true);
                d_pyrLK_sparse->calc(cur_gpu_img, prev_gpu_img, cur_gpu_pts, reverse_gpu_pts, reverse_gpu_status);

                vector<cv::Point2f> reverse_pts(reverse_gpu_pts.cols);
                reverse_gpu_pts.download(reverse_pts);

                vector<uchar> reverse_status(reverse_gpu_status.cols);
                reverse_gpu_status.download(reverse_status);

                for (size_t i = 0; i < status.size(); i++)
                {
                    if (status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                    {
                        status[i] = 1;
                    }
                    else
                        status[i] = 0;
                }
            }
            // printf("gpu temporal optical flow costs: %f ms\n",t_og.toc());
        }
#endif

        // 遍历cur_pts向量，包含了当前帧中检测到的特征点，类型是 std::vector<cv::Point2f>,表示每个特征点的2D坐标
        for (int i = 0; i < int(cur_pts.size()); i++)
            // 检查跟踪状态和边界条件，inBorder() 函数用于检查特征点是否在图像边界内
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;
        // 根据跟踪状态更新特征点向量
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        // ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());

        // printf("track cnt %d\n", (int)ids.size());
    }

    // 特征点检测与跟踪
    for (auto &n : track_cnt)
        n++;

    // 特征检测，使用OpenCV提取图像关键点的过程
    // 并提供了两种方式：1. 基于CPU的cv::goodFeaturesToTrack; 2. 基于GPU的cv::cuda::goodFeaturesToTrack
    if (1)
    {
        // rejectWithF();
        ROS_DEBUG("set mask begins"); // 在ROS日志中输出调试信息
        TicToc t_m;                   // 创建一个计时器对象 t_m，用于测量特征检测的执行时间
        setMask();                    // 调用自定义的掩膜设置函数，生成或调整mask掩膜图像
        // ROS_DEBUG("set mask costs %fms", t_m.toc());
        // printf("set mask costs %fms\n", t_m.toc());
        ROS_DEBUG("detect feature begins"); // 日志标志开始特征检测的过程

        // 计算还需检测的最大点数
        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
        if (!USE_GPU) // 非GPU模式
        {
            if (n_max_cnt > 0) // 如果还有剩余点需要检测
            {
                TicToc t_t;       // 计时器
                if (mask.empty()) // 检查mask是否为空
                    cout << "mask is empty " << endl;
                if (mask.type() != CV_8UC1) // 检查mask类型是否正确
                    cout << "mask type wrong " << endl;
                // 使用OpenCV的goodFeaturesToTrack函数检测图像中的角点
                cv::goodFeaturesToTrack(cur_img,                  // 输入图像
                                        n_pts,                    // 输出的角点坐标
                                        MAX_CNT - cur_pts.size(), // 需要检测的角点数量
                                        0.005,                    // 角点检测的质量等级阈值
                                        MIN_DIST,                 // 两个角点之间的最小欧氏距离阈值
                                        mask);                    // 输入的掩膜图像
                // printf("good feature to track costs: %fms\n", t_t.toc());
                std::cout << "n_pts size: " << n_pts.size() << std::endl; // 打印检测到的特征点数
            }
            else
                n_pts.clear(); // 如果不需要更新检测点，则清空特征点集合
            // sum_n += n_pts.size();
            // printf("total point from non-gpu: %d\n",sum_n);
        }
#ifdef GPU_MODE // 使用GPU模式
        // ROS_DEBUG("detect feature costs: %fms", t_t.toc());
        // printf("good feature to track costs: %fms\n", t_t.toc());
        else
        {
            if (n_max_cnt > 0) // 如果还有剩余点需要检测
            {
                if (mask.empty()) // 检查掩膜是否为空
                    cout << "mask is empty " << endl;
                if (mask.type() != CV_8UC1) // 检查掩膜类型
                    cout << "mask type wrong " << endl;
                TicToc t_g;                            // 计时器
                cv::cuda::GpuMat cur_gpu_img(cur_img); // 将当前图像加载到GPU
                cv::cuda::GpuMat d_prevPts;            // 用于存储GPU上的特征点
                TicToc t_gg;
                cv::cuda::GpuMat gpu_mask(mask); // 将掩膜加载到GPU
                // printf("gpumat cost: %fms\n",t_gg.toc());
                // 使用CUDA的createGoodFeaturesToTrackDetector函数创建一个特征检测器对象detector
                cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(cur_gpu_img.type(),       // 图像类型
                                                                                                          MAX_CNT - cur_pts.size(), // 最大特征点数
                                                                                                          0.01,                     // 角点检测的质量阈值
                                                                                                          MIN_DIST);                // 最小距离
                // cout << "new gpu points: "<< MAX_CNT - cur_pts.size()<<endl;
                // 检测特征点
                detector->detect(cur_gpu_img, d_prevPts, gpu_mask);
                // std::cout << "d_prevPts size: "<< d_prevPts.size()<<std::endl;
                if (!d_prevPts.empty())                                // 如果检测到特征点
                    n_pts = cv::Mat_<cv::Point2f>(cv::Mat(d_prevPts)); // 转换为Mat格式
                else
                    n_pts.clear(); // 如果没有检测到特征点，则清空特征点集合
                // sum_n += n_pts.size();
                // printf("total point from gpu: %d\n",sum_n);
                // printf("gpu good feature to track cost: %fms\n", t_g.toc());
            }
            else
                n_pts.clear(); // 如果不需要更新检测点，则清空特征点集合
        }
#endif

        ROS_DEBUG("add feature begins"); // 开始添加新检测到的特征点
        TicToc t_a;                      // 创建计时器
        addPoints();                     // 调用自定义函数，向特征点集合中添加新检测到的特征点
        // ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
        // printf("selectFeature costs: %fms\n", t_a.toc());
    }

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);                            // 对当前帧的特征点坐标进行去畸变操作，得到归一化坐标cur_un_pts，便于后续的三角化或相机模型计算
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map); // 计算特征点在图像帧间的速度，用于预测和跟踪优化，利用当前和上一帧特征点的映射关系计算速度

    // 双目特征点处理
    if (stereo_cam)
        if (!_img1.empty() && stereo_cam) // 如果是双目相机，则会进一步对右目图像提取特征点
        {
            // 处理立体相机中的右图特征点
            ids_right.clear();            // 清空右目特征点的ID集合
            cur_right_pts.clear();        // 清空右目特征点的坐标集合
            cur_un_right_pts.clear();     // 清空右目特征点的归一化坐标集合
            right_pts_velocity.clear();   // 清空右目特征点的速度集合
            cur_un_right_pts_map.clear(); // 清空右目特征点的归一化坐标映射关系
            // 左右图像特征点的光流计算和特征点匹配
            if (!cur_pts.empty())
            {
                // printf("stereo image; track feature on right image\n");

                vector<cv::Point2f> reverseLeftPts;
                vector<uchar> status, statusRightLeft;
                if (!USE_GPU_ACC_FLOW)
                {
                    TicToc t_check;
                    vector<float> err;
                    // cur left ---- cur right
                    // 找到当前帧左图（cur_img）中的特征点cur_pts在右图 rightImage中对应的点 cur_right_pts
                    cv::calcOpticalFlowPyrLK(cur_img,          // 当前帧图像
                                             rightImg,         // 右帧图像
                                             cur_pts,          // 当前帧特征点
                                             cur_right_pts,    // 当前帧特征点的匹配点
                                             status,           // 特征点的跟踪状态，记录每个特征点是否成功匹配
                                             err,              // 特征点的误差，记录每个特征点的误差
                                             cv::Size(21, 21), // 搜索窗口大小
                                             3);               // 金字塔层数
                    // reverse check cur right ---- cur left
                    /**方向检查
                     * 1. 使用光流从右图追踪回左图（reverseLeftPts）
                     * 2. 检查双向光流的一致性
                     * 3. 如果误差较小（distance小于0.5）且点在图像边界内（inBorder），则将状态标记为1
                     */
                    if (FLOW_BACK)
                    {
                        cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
                        for (size_t i = 0; i < status.size(); i++)
                        {
                            if (status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                                status[i] = 1;
                            else
                                status[i] = 0;
                        }
                    }
                    // printf("left right optical flow cost %fms\n",t_check.toc());
                }
#ifdef GPU_MODE
                else
                {
                    TicToc t_og1;
                    cv::cuda::GpuMat cur_gpu_img(cur_img);
                    cv::cuda::GpuMat right_gpu_Img(rightImg);
                    cv::cuda::GpuMat cur_gpu_pts(cur_pts);
                    cv::cuda::GpuMat cur_right_gpu_pts;
                    cv::cuda::GpuMat gpu_status;
                    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                        cv::Size(21, 21), 3, 30, false);
                    d_pyrLK_sparse->calc(cur_gpu_img, right_gpu_Img, cur_gpu_pts, cur_right_gpu_pts, gpu_status);

                    vector<cv::Point2f> tmp_cur_right_pts(cur_right_gpu_pts.cols);
                    cur_right_gpu_pts.download(tmp_cur_right_pts);
                    cur_right_pts = tmp_cur_right_pts;

                    vector<uchar> tmp_status(gpu_status.cols);
                    gpu_status.download(tmp_status);
                    status = tmp_status;

                    if (FLOW_BACK)
                    {
                        cv::cuda::GpuMat reverseLeft_gpu_Pts;
                        cv::cuda::GpuMat status_gpu_RightLeft;
                        cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                            cv::Size(21, 21), 3, 30, false);
                        d_pyrLK_sparse->calc(right_gpu_Img, cur_gpu_img, cur_right_gpu_pts, reverseLeft_gpu_Pts, status_gpu_RightLeft);

                        vector<cv::Point2f> tmp_reverseLeft_Pts(reverseLeft_gpu_Pts.cols);
                        reverseLeft_gpu_Pts.download(tmp_reverseLeft_Pts);
                        reverseLeftPts = tmp_reverseLeft_Pts;

                        vector<uchar> tmp1_status(status_gpu_RightLeft.cols);
                        status_gpu_RightLeft.download(tmp1_status);
                        statusRightLeft = tmp1_status;
                        for (size_t i = 0; i < status.size(); i++)
                        {
                            if (status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                                status[i] = 1;
                            else
                                status[i] = 0;
                        }
                    }
                    // printf("gpu left right optical flow cost %fms\n",t_og1.toc());
                }
#endif
                // 特征点筛选和去畸变
                // 使用 status剔除未匹配成功的特征点，更新右图特征点 cur_right_pts 和 ID ids_right
                ids_right = ids;
                reduceVector(cur_right_pts, status);
                reduceVector(ids_right, status);
                // only keep left-right pts
                /*
                reduceVector(cur_pts, status);
                reduceVector(ids, status);
                reduceVector(track_cnt, status);
                reduceVector(cur_un_pts, status);
                reduceVector(pts_velocity, status);
                */
                cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
                right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
            }
            prev_un_right_pts_map = cur_un_right_pts_map;
        }
    // 绘制跟踪结果
    if (SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    // 更新和记录特征点
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;

    prevLeftPtsMap.clear();
    for (size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    // 生成特征点数据结构
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame; // 左图特征点的最终结构
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i]; // 特征点ID
        double x, y, z; // 无畸变的归一化坐标
        x = cur_un_pts[i].x; 
        y = cur_un_pts[i].y;
        z = 1;
        double p_u, p_v;
        p_u = cur_pts[i].x; // 原始像素坐标 p_u
        p_v = cur_pts[i].y; // 原始像素坐标 p_v
        int camera_id = 0; // 表示左图
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x; // x的速度
        velocity_y = pts_velocity[i].y; // y的速度

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }

    // 右图特征点的组织，对右图特征点执行类似操作，将其加入featureFrame中
    if (!_img1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x, y, z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id = 1; // 表示右图
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
        }
    }

    // printf("feature track whole time %f\n", t_r.toc());
    /**
     * Eigen::Matrix<double, 7, 1>的结构
     * [0]：x 特征点在图像中的横坐标
     * [1]：y 特征点在图像中的纵坐标
     * [2]：z 特征点的深度值（从3D重建或其他传感器获取）
     * [3]：速度u 特征点在x方向上的移动速度
     * [4]：速度v 特征点在y方向上的移动速度
     * [5]：观测时间 特征点的观测时间戳
     * [6]：特征强度 特征点的强度值或置信度
     */
    return featureFrame; // 包含特征点的三维位置、速度、像素坐标等信息，供后续使用
}

void FeatureTracker::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        // cout << trackerData[0].K << endl;
        // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

/**
 * @brief 将像素坐标点转换为归一化平面坐标点
 * @param pts 输入的像素坐标点向量
 * @param cam 相机模型指针
 * @return 转换后的归一化平面坐标点向量
 */
vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts; // 存储转换后的归一化平面坐标点
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y); // 将像素坐标点转换为 Eigen::Vector2d 类型
        Eigen::Vector3d b;
        cam->liftProjective(a, b);                                   // 使用相机模型将像素座标点提升到归一化平面
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z())); // 将归一化平面坐标点转换为 cv::Point2f 类型并添加到结果向量中
    }
    return un_pts; // 返回转换后的归一化平面坐标点向量
}

/**
 * @brief 计算特征点的速度
 *
 * @param ids                     特征点的ID列表
 * @param pts                     特征点的坐标列表
 * @param cur_id_pts              当前帧中特征点的ID和坐标映射
 * @param prev_id_pts             上一帧中特征点的ID和坐标映射
 * @return vector<cv::Point2f>    特征点的速度向量
 */
vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                                map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity; // 存储特征点速度的向量
    cur_id_pts.clear();               // 清空当前帧中特征点的ID和坐标映射
    // 将当前帧中的特征点ID和坐标添加到cur_id_pts映射中
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    // 计算特征点速度
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time; // 计算当前帧和上一帧之间的时间差

        // 遍历当前帧中的特征点
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]); // 在prev_id_pts中查找当前特征点的ID
            if (it != prev_id_pts.end())
            {
                // 计算特征点在x和y方向上的速度
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y)); // 将速度添加到pts_velocity向量中
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0)); // 如果未找到对应ID，则速度为0
        }
    }
    else
    {
        // 如果上一帧的特征点映射为空，则将所有特征点速度设置为0
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity; // 返回特征点的速度向量
}

/**
 * @brief 绘制特征点的跟踪轨迹
 * 
 * @param imLeft           作为输入的左图像
 * @param imRight          右图像
 * @param curLeftIds       当前帧的左图像特征点ID
 * @param curLeftPts       当前帧的左图像特征点坐标
 * @param curRightPts      当前帧的右图像特征点坐标
 * @param prevLeftPtsMap   前一帧左图像中特征点的ID和坐标映射
 */
void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts,
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    // int rows = imLeft.rows;
    int cols = imLeft.cols;
    // 将左图像和右图像拼接在一起
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2RGB); // 将imTrack从灰度图像转换为RGB图像

    // 遍历当前帧左图像中的每个特征点
    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        // 计算每个特征点的跟踪长度比例
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        // 在imTrack上绘制特征点的跟踪轨迹
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    // 如果右图像不为空且是立体相机
    if (!imRight.empty() && stereo_cam)
    {
        // 遍历当前帧右图像中的每个特征点
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            // 将右图像中的特征点坐标平移到平均图像的右半部分
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            // 在imTrack上绘制右图像中的特征点
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            // cv::Point2f leftPt = curLeftPtsTrackRight[i];
            // cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
    
    // 遍历当前帧左图像中的每个特征点ID
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        // 在preLeftPtsMap中查找当前特征点ID
        mapIt = prevLeftPtsMap.find(id);
        // 如果找到了
        if (mapIt != prevLeftPtsMap.end())
        {
            // 在imTrack上绘制从当前特征点到前一帧特征点的箭头
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    // draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    // printf("predict pts size %d \n", (int)predict_pts_debug.size());

    // cv::Mat imCur2Compress;
    // cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}

void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true;
    predict_pts.clear();
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++)
    {
        // printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end())
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]);
    }
}

void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if (itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}

cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}