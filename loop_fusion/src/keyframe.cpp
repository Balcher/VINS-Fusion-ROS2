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

#include "keyframe.h"

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// create keyframe online
/**
 * @brief Construct a new Key Frame:: Key Frame object
 *
 * keyFrame类的一个重载版本，用于初始化一个关键帧对象，
 * 主要作用是将传入的数据进行赋值和初始化，还包括关键点的计算以及缩略图的生成
 *
 * @param _time_stamp       关键帧的时间戳，用于标记该帧的捕获时间
 * @param _index            关键帧的全局索引，用于唯一标识此关键帧
 * @param _vio_T_w_i        表示视觉惯性里程计（VIO）估计的世界坐标系到当前帧的平移向量。
 * @param _vio_R_w_i        表示视觉惯性里程计（VIO）估计的世界坐标系到当前帧的旋转矩阵。
 * @param _image            当前帧的图像数据
 * @param _point_3d         3D空间中的点云信息，通常为特征点的三维坐标
 * @param _point_2d_uv      3D空间中的点云信息，通常为特征点的三维坐标
 * @param _point_2d_norm    归一化相机坐标系下的二维特征点坐标
 * @param _point_id         每个特征点对应的唯一 ID
 * @param _sequence         关键帧所属的序列编号，可能用于标识不同的视频序列或数据流
 */
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
                   vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
                   vector<double> &_point_id, int _sequence)
{
    time_stamp = _time_stamp;                       // 将时间戳赋值给关键帧对象
    index = _index;                                 // 索引复制给关键帧对象
    vio_T_w_i = _vio_T_w_i;                         // 将视觉里程计估计的平移向量赋值给关键帧对象
    vio_R_w_i = _vio_R_w_i;                         // 将视觉里程计估计的旋转矩阵赋值给关键帧对象
    T_w_i = vio_T_w_i;                              // 将视觉里程计估计的平移向量赋值给关键帧对象
    R_w_i = vio_R_w_i;                              // 将视觉里程计估计的旋转矩阵赋值给关键帧对象
    origin_vio_T = vio_T_w_i;                       // 将视觉里程计估计的平移向量赋值给关键帧对象
    origin_vio_R = vio_R_w_i;                       // 将视觉里程计估计的旋转矩阵赋值给关键帧对象
    image = _image.clone();                         // 将图像数据复制给关键帧对象
    cv::resize(image, thumbnail, cv::Size(80, 60)); // 将图像数据缩放到缩略图尺寸
    point_3d = _point_3d;                           // 将特征点的三维坐标复制给关键帧对象
    point_2d_uv = _point_2d_uv;                     // 将特征点的归一化坐标复制给关键帧对象
    point_2d_norm = _point_2d_norm;                 // 将特征点的归一化坐标复制给关键帧对象
    point_id = _point_id;                           // 将特征点的 ID 复制给关键帧对象
    has_loop = false;                               // 将循环检测标志设置为 false
    loop_index = -1;                                // 将循环索引设置为 -1
    has_fast_point = false;                         // 将快速点检测标志设置为 false
    loop_info << 0, 0, 0, 0, 0, 0, 0, 0;            // 将循环信息设置为零矩阵
    sequence = _sequence;                           // 将序列编号复制给关键帧对象
    computeWindowBRIEFPoint();                      // 计算窗口 BRIEF 特征点
    computeBRIEFPoint();                            // 计算 BRIEF 特征点
    if (!DEBUG_IMAGE)
        image.release();
}

// load previous keyframe
/**
 * @brief Construct a new Key Frame:: Key Frame object
 *
 * 用于初始化一个关键帧对象，与之前的构造函数相比，加入了关键点、归一化关键点、BRIEF描述子等特征信息
 *
 * @param _time_stamp           时间戳，标记此关键帧的捕获时间
 * @param _index                关键帧的全局唯一索引
 * @param _vio_T_w_i            视觉惯性里程计估计的平移向量
 * @param _vio_R_w_i            视觉惯性里程计估计的旋转矩阵
 * @param _T_w_i                实际估计的关键帧位姿，用于更新当前帧的世界坐标系平移信息
 * @param _R_w_i                实际估计的关键帧位姿，用于更新当前帧的世界坐标系旋转信息
 * @param _image                当前帧的图像数据
 * @param _loop_index           表示与当前帧相关的循环帧的索引，用于跟踪循环关系
 * @param _loop_info            与回环检测相关的8维向量，保存回环检测的附加信息（如位姿变换参数等）
 * @param _keypoints            关键帧的关键点集合，用于特征匹配和描述子计算
 * @param _keypoints_norm       关键帧的归一化关键点集合，用于特征匹配和描述子计算
 * @param _brief_descriptors    关键帧的 BRIEF 描述子集合，用于特征匹配和描述子计算
 */
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
                   cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1> &_loop_info,
                   vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors)
{
    // 将传入的时间戳和索引值直接赋给关键帧对象
    time_stamp = _time_stamp;
    index = _index;
    // vio_T_w_i = _vio_T_w_i;
    // vio_R_w_i = _vio_R_w_i;
    // 位姿初始化，将vio_T_w_i和vio_R_w_i初始化为实际估计的位姿势(_T_w_i和_R_w_i)
    // 同时 T_w_i 和 R_w_i 保存世界坐标系下的平移和旋转，便于后续操作
    vio_T_w_i = _T_w_i;
    vio_R_w_i = _R_w_i;
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
    // 当开启调试模式（DEBUG_IMAGE）时，才会深拷贝图像数据并生成缩略图以节省内存
    if (DEBUG_IMAGE)
    {
        image = _image.clone();
        cv::resize(image, thumbnail, cv::Size(80, 60));
    }
    // 回环检测
    if (_loop_index != -1) // 如果 _loop_index 不等于 -1，说明当前帧有回环
        has_loop = true;
    else
        has_loop = false;
    loop_index = _loop_index;
    loop_info = _loop_info;
    has_fast_point = false; // 快速点检测
    sequence = 0;           // 序列初始化，序列号固定为0,可能是因为不涉及多序列处理
    // 关键点和描述子赋值
    keypoints = _keypoints;
    keypoints_norm = _keypoints_norm;
    brief_descriptors = _brief_descriptors;
}

/**
 * @brief 计算图像窗口区域的BRIEF特征点
 *
 * 该函数用于在给定的图像窗口区域内提取BRIEF特征点和描述子。
 * 它使用BriefExtractor类来计算特征点和描述子，并将结果存储在
 * window_keypoints和window_brief_descriptors向量中。
 *
 * @param image 输入图像
 * @param point_2d_uv 图像窗口区域的2D点坐标向量
 * @param window_keypoints 存储提取的BRIEF特征点的向量
 * @param window_brief_descriptors 存储提取的BRIEF描述子的向量
 */
void KeyFrame::computeWindowBRIEFPoint()
{
    // 创建一个BriefExtractor对象，用于提取BRIEF特征
    BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
    // 遍历point_2d_uv中的每个2D点
    for (int i = 0; i < (int)point_2d_uv.size(); i++)
    {
        // 创建一个cv::KeyPoint对象，表示图像中的一个特征点
        cv::KeyPoint key;
        // 设置特征点的位置为point_2d_uv中的当前点
        key.pt = point_2d_uv[i];
        // 将当前特征点添加到window_keypoints向量中
        window_keypoints.push_back(key);
    }
    // 使用BriefExtractor对象从图像中提取BRIEF特征点和描述子
    extractor(image, window_keypoints, window_brief_descriptors);
}

/**
 * @brief 计算图像的BRIEF特征点
 * 
 * 该函数用于在给定的图像中提取BRIEF特征点和描述子。
 * 它使用BriefExtractor类来计算特征点和描述子，并将结果存储在
 * keypoints和brief_descriptors向量中。此外，它还计算了特征点的归一化坐标，
 * 并将这些坐标存储在keypoints_norm向量中。
 * 
 * @param image 输入图像
 * @param fast_th FAST角点检测的阈值
 * @param m_camera 相机模型，用于将图像坐标转换为归一化坐标
 */
void KeyFrame::computeBRIEFPoint()
{
    BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
    const int fast_th = 20; // corner detector response threshold
    if (1)
        cv::FAST(image, keypoints, fast_th, true);
    else
    {
        vector<cv::Point2f> tmp_pts;
        cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
        for (int i = 0; i < (int)tmp_pts.size(); i++)
        {
            cv::KeyPoint key;
            key.pt = tmp_pts[i];
            keypoints.push_back(key);
        }
    }
    extractor(image, keypoints, brief_descriptors);
    for (int i = 0; i < (int)keypoints.size(); i++)
    {
        Eigen::Vector3d tmp_p;
        m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
        cv::KeyPoint tmp_norm;
        tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
        keypoints_norm.push_back(tmp_norm);
    }
}

/**
 * @brief 重载的运算符函数，用于计算图像的BRIEF特征描述子
 * 
 * 这个函数是BriefExtractor类的一部分，它被设计为一个函数对象，可以像函数一样被调用。
 * 它接受一个OpenCV图像对象、一个关键点向量和一个描述子向量作为参数。
 * 函数内部调用了m_brief对象的compute方法来计算图像中每个关键点的BRIEF描述子。
 * 
 * @param im 输入的OpenCV图像对象
 * @param keys 图像中的关键点向量
 * @param descriptors 计算得到的BRIEF描述子向量
 */
void BriefExtractor::operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
    m_brief.compute(im, keys, descriptors);
}

/**
 * @brief 在指定区域内搜索与给定描述符最匹配的描述符
 * 
 * 该函数用于在给定的区域内搜索与当前窗口描述符最匹配的描述符，并返回最佳匹配的关键点和归一化关键点。
 * 它通过计算 Hamming 距离来衡量描述符之间的相似度，并选择距离最小的描述符作为最佳匹配。
 * 
 * @param window_descriptor     当前窗口的描述符
 * @param descriptors_old       旧的描述符集合
 * @param keypoints_old         旧的关键点集合
 * @param keypoints_old_norm    旧的归一化关键点集合
 * @param best_match            最佳匹配的关键点
 * @param best_match_norm       最佳匹配的归一化关键点
 * @return 如果找到匹配的描述符，则返回 true，否则返回 false
 */
bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm)
{
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
    for (int i = 0; i < (int)descriptors_old.size(); i++)
    {
        // 计算当前窗口描述符与旧描述符之间的Hamming距离
        int dis = HammingDis(window_descriptor, descriptors_old[i]);
        if (dis < bestDist)
        {
            bestDist = dis;
            bestIndex = i;
        }
    }
    // printf("best dist %d", bestDist);
    if (bestIndex != -1 && bestDist < 80)
    {
        // 如果找到匹配的描述符，更新最佳匹配点和归一化匹配点
        best_match = keypoints_old[bestIndex].pt;
        best_match_norm = keypoints_old_norm[bestIndex].pt;
        return true;
    }
    else
        return false;
}
/**
 * @brief 在指定区域内搜索与给定描述符最匹配的描述符
 * 
 * 该函数用于在给定的区域内搜索与当前窗口描述符最匹配的描述符，并返回最佳匹配的关键点和归一化关键点。
 * 它通过计算 Hamming 距离来衡量描述符之间的相似度，并选择距离最小的描述符作为最佳匹配。
 * 
 * @param window_descriptor 当前窗口的描述符
 * @param descriptors_old 旧的描述符集合
 * @param keypoints_old 旧的关键点集合
 * @param keypoints_old_norm 旧的归一化关键点集合
 * @param best_match 最佳匹配的关键点
 * @param best_match_norm 最佳匹配的归一化关键点
 * @return 如果找到匹配的描述符，则返回 true，否则返回 false
 */
void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                                std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm)
{
    for (int i = 0; i < (int)window_brief_descriptors.size(); i++)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
            status.push_back(1);
        else
            status.push_back(0);
        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }
}

/**
 * @brief 使用RANSAC算法计算两帧图像之间的基本矩阵
 * 
 * 该函数通过RANSAC算法来估计两帧图像之间的基本矩阵，从而找到匹配的特征点对。
 * 它首先将归一化的图像坐标转换为像素坐标，然后使用OpenCV的findFundamentalMat函数来计算基本矩阵。
 * 
 * @param matched_2d_cur_norm 当前帧的归一化2D点
 * @param matched_2d_old_norm 旧帧的归一化2D点
 * @param status 存储每个点对是否为内点的状态向量
 */
void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                      const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      vector<uchar> &status)
{
    int n = (int)matched_2d_cur_norm.size();
    for (int i = 0; i < n; i++)
        status.push_back(0);
    if (n >= 8)
    {
        vector<cv::Point2f> tmp_cur(n), tmp_old(n);
        for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
        {
            double FOCAL_LENGTH = 460.0;
            double tmp_x, tmp_y;
            tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
            tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

            tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
        }
        cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
    }
}

/**
 * @brief 使用RANSAC算法求解PnP问题，估计相机位姿
 * 
 * 该函数通过RANSAC算法来估计相机的位姿，即旋转矩阵和平移向量。
 * 它使用匹配的3D点和2D点来计算相机的位姿，并返回估计的旋转矩阵和平移向量。
 * 
 * @param matched_2d_old_norm 归一化的2D点匹配对
 * @param matched_3d 对应的3D点
 * @param status 存储每个点对是否为内点的状态向量
 * @param PnP_T_old 估计的平移向量
 * @param PnP_R_old 估计的旋转矩阵
 */
void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
    // for (int i = 0; i < matched_3d.size(); i++)
    //	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
    // printf("match size %d \n", matched_3d.size());
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    Matrix3d R_inital;
    Vector3d P_inital;
    Matrix3d R_w_c = origin_vio_R * qic;
    Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);

    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;
    TicToc t_pnp_ransac;

    if (CV_MAJOR_VERSION < 3)
        solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
    else
    {
        if (CV_MINOR_VERSION < 2)
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
        else
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);
    }

    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(0);

    for (int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }

    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);

    PnP_R_old = R_w_c_old * qic.transpose();
    PnP_T_old = T_w_c_old - PnP_R_old * tic;
}

// 实现了两个关键帧之间寻找连接点的功能，
// 主要步骤涉及特征点匹配、基本矩阵估计和可视化
bool KeyFrame::findConnection(KeyFrame *old_kf)
{
    TicToc tmp_t;
    // printf("find Connection\n");
    vector<cv::Point2f> matched_2d_cur, matched_2d_old;
    vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
    vector<cv::Point3f> matched_3d;
    vector<double> matched_id;
    vector<uchar> status;

    matched_3d = point_3d;
    matched_2d_cur = point_2d_uv;
    matched_2d_cur_norm = point_2d_norm;
    matched_id = point_id;

    TicToc t_match;
#if 0
		if (DEBUG_IMAGE)    
	    {
	        cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        cv::hconcat(image, old_img, gray_img);
	        cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        ostringstream path;
	        path << "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif
    // printf("search by des\n");
    searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_cur_norm, status);
    reduceVector(matched_2d_old_norm, status);
    reduceVector(matched_3d, status);
    reduceVector(matched_id, status);
    // printf("search by des finish\n");

#if 0 
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap);
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path, path1, path2;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	        /*
	        path1 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_1.jpg";
	        cv::imwrite( path1.str().c_str(), image);
	        path2 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_2.jpg";
	        cv::imwrite( path2.str().c_str(), old_img);	        
	        */
	        
	    }
#endif
    status.clear();
/*
FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
reduceVector(matched_2d_cur, status);
reduceVector(matched_2d_old, status);
reduceVector(matched_2d_cur_norm, status);
reduceVector(matched_2d_old_norm, status);
reduceVector(matched_3d, status);
reduceVector(matched_id, status);
*/
#if 0
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap) ;
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "2fundamental_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif
    Eigen::Vector3d PnP_T_old;
    Eigen::Matrix3d PnP_R_old;
    Eigen::Vector3d relative_t;
    Quaterniond relative_q;
    double relative_yaw;
    if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
    {
        status.clear();
        PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
        reduceVector(matched_2d_cur, status);
        reduceVector(matched_2d_old, status);
        reduceVector(matched_2d_cur_norm, status);
        reduceVector(matched_2d_old_norm, status);
        reduceVector(matched_3d, status);
        reduceVector(matched_id, status);
#if 1
        if (DEBUG_IMAGE)
        {
            int gap = 10;
            cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
            for (int i = 0; i < (int)matched_2d_cur.size(); i++)
            {
                cv::Point2f cur_pt = matched_2d_cur[i];
                cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
            }
            for (int i = 0; i < (int)matched_2d_old.size(); i++)
            {
                cv::Point2f old_pt = matched_2d_old[i];
                old_pt.x += (COL + gap);
                cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
            }
            for (int i = 0; i < (int)matched_2d_cur.size(); i++)
            {
                cv::Point2f old_pt = matched_2d_old[i];
                old_pt.x += (COL + gap);
                cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
            }
            cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
            putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

            putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
            cv::vconcat(notation, loop_match_img, loop_match_img);

            /*
            ostringstream path;
            path <<  "/home/tony-ws1/raw_data/loop_image/"
                    << index << "-"
                    << old_kf->index << "-" << "3pnp_match.jpg";
            cv::imwrite( path.str().c_str(), loop_match_img);
            */
            if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
            {
                /*
                cv::imshow("loop connection",loop_match_img);
                cv::waitKey(10);
                */
                cv::Mat thumbimage;
                cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
                // sensor_msgs::msg::ImagePtr
                sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", thumbimage).toImageMsg();

                int sec_ts = (int)time_stamp;
                uint nsec_ts = (uint)((time_stamp - sec_ts) * 1e9);
                msg->header.stamp.sec = sec_ts;
                msg->header.stamp.nanosec = nsec_ts;

                pub_match_img->publish(*msg);
            }
        }
#endif
    }

    if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
    {
        relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
        relative_q = PnP_R_old.transpose() * origin_vio_R;
        relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
        // printf("PNP relative\n");
        // cout << "pnp relative_t " << relative_t.transpose() << endl;
        // cout << "pnp relative_yaw " << relative_yaw << endl;
        if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)
        {

            has_loop = true;
            loop_index = old_kf->index;
            loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
                relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
                relative_yaw;
            // cout << "pnp relative_t " << relative_t.transpose() << endl;
            // cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
            return true;
        }
    }
    // printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(), t_match.toc());
    return false;
}

/**
 * 计算两个 BRIEF 描述符之间的汉明距离
 * @param a 第一个 BRIEF 描述符
 * @param b 第二个 BRIEF 描述符
 * @return 两个描述符之间的汉明距离
 */
int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}
/**
 * 获取关键帧的视觉惯性里程计（VIO）姿态
 * @param _T_w_i 输出参数，存储关键帧的位置向量
 * @param _R_w_i 输出参数，存储关键帧的旋转矩阵
 */
void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}
/**
 * 获取关键帧的姿态
 * @param _T_w_i 输出参数，存储关键帧的位置向量
 * @param _R_w_i 输出参数，存储关键帧的旋转矩阵
 */
void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

/**
 * 更新关键帧的姿态
 * @param _T_w_i 新的位置向量
 * @param _R_w_i 新的旋转矩阵
 */
void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}
/**
 * 更新关键帧的视觉惯性里程计（VIO）姿态
 * @param _T_w_i 新的位置向量
 * @param _R_w_i 新的旋转矩阵
 */
void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    vio_T_w_i = _T_w_i;
    vio_R_w_i = _R_w_i;
    T_w_i = vio_T_w_i;
    R_w_i = vio_R_w_i;
}
/**
 * 获取关键帧的循环相对平移向量
 * @return 循环相对平移向量
 */
Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}
/**
 * 获取关键帧的循环相对旋转四元数
 * @return 循环相对旋转四元数
 */
Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}
/**
 * 获取关键帧的循环相对偏航角（yaw）
 * @return 循环相对偏航角（yaw）
 */
double KeyFrame::getLoopRelativeYaw()
{
    return loop_info(7);
}
/**
 * 更新关键帧的循环信息
 * @param _loop_info 包含循环信息的 8 维向量
 */
void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1> &_loop_info)
{
    if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
    {
        // printf("update loop info\n");
        loop_info = _loop_info;
    }
}
/**
 * 构造函数，用于初始化 BRIEF 特征描述子提取器
 * @param pattern_file 模式文件的路径，该文件包含了用于生成 BRIEF 描述子的点对信息
 */
BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
    // The DVision::BRIEF extractor computes a random pattern by default when
    // the object is created.
    // We load the pattern that we used to build the vocabulary, to make
    // the descriptors compatible with the predefined vocabulary

    // loads the pattern
    cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
    if (!fs.isOpened())
        throw string("Could not open file ") + pattern_file;

    vector<int> x1, y1, x2, y2;
    fs["x1"] >> x1;
    fs["x2"] >> x2;
    fs["y1"] >> y1;
    fs["y2"] >> y2;

    m_brief.importPairs(x1, y1, x2, y2);
}
