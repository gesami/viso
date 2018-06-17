//
// Created by sage on 16.06.18.
//

#ifndef VISO_INITIALIZER_H
#define VISO_INITIALIZER_H

#include <common.h>
#include <keyframe.h>
#include <map.h>

class Initializer {
public:
    Initializer();
    bool InitializeMap(Keyframe::Ptr cur_frame, Map* map, const cv::Mat& display);

private:
    void OpticalFlowSingleLevel(const cv::Mat& img1, const cv::Mat& img2,
        const std::vector<cv::KeyPoint>& kp1,
        std::vector<cv::KeyPoint>& kp2,
        std::vector<bool>& success, int& good_cnt, bool inverse);

    void OpticalFlowMultiLevel(
        Keyframe* const ref_frame,
        Keyframe* const cur_frame,
        std::vector<cv::KeyPoint>& kp1,
        std::vector<cv::KeyPoint>& kp2,
        std::vector<bool>& success, int& good_cnt,
        bool inverse);

    double CalculateDisparity2(const std::vector<cv::KeyPoint>& kp1,
        const std::vector<cv::KeyPoint>& kp2,
        const std::vector<bool>& success);

    // |points1| and |points2| are observations on the image plane,
    // meaning that the inverse intrinsic matrix has already been applied to
    // the respective pixel coordinates.
    void PoseEstimation(
        const std::vector<cv::Point2f>& p1,
        const std::vector<cv::Point2f>& p2,
        M3d& R, V3d& T,
        std::vector<bool>& success,
        int& good_cnt);

    // Triangulation, see paper "Triangulation", Section 5.1, by Richard I. Hartley, Peter Sturm
    void Triangulate(const M34d& Pi1, const M34d& Pi2, const cv::Point2f& x1, const cv::Point2f& x2, V3d& P);

    void Reconstruct(
        const std::vector<cv::Point2f>& p1,
        const std::vector<cv::Point2f>& p2,
        const M3d& R, const V3d& T, std::vector<bool>& success,
        int& good_cnt, std::vector<V3d>& points3d);

    void NormalizeDepth(V3d& T, std::vector<V3d>& points3d);

    void Visualize(const cv::Mat& display, const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2, const std::vector<bool>& success, const int& good_cnt);

    Keyframe::Ptr ref_frame_;
    std::vector<cv::KeyPoint> cur_kp_;
    std::vector<cv::KeyPoint> ref_kp_;
    std::vector<bool> track_success_;
    std::vector<V3d> points3d_;
    M3d K_; // the current's frame intrinsic parameters
    int frame_cnt_;

    // Parameters
    int reset_after_;
    int iterations_;
    int half_patch_size_;
    double photometric_thresh_;
    int min_good_cnt_;
    double disparity2_thresh_;
    double reprojection_thresh_;
};

#endif //VISO_INITIALIZER_H
