
#ifndef VISO_VISO_H
#define VISO_VISO_H

#include "bundle_adjuster.h"
#include "config.h"
#include "frame_sequence.h"
#include "keyframe.h"
#include "ring_buffer.h"
#include "slam_map.h"
#include <atomic>
#include <initializer.h>
#include <sophus/se3.hpp>
#include <thread>
#include <tuple>

class Viso : public FrameSequence::FrameHandler {
private:
    enum State {
        kInitialization = 0,
        kRunning = 1,
        kFinished = 2
    };

    int lk_half_patch_size = Config::get<int>("lk_half_patch_size");
    double lk_photometric_thresh = (lk_half_patch_size * 2) * (lk_half_patch_size * 2) * Config::get<double>("lk_photometric_thresh") * Config::get<double>("lk_photometric_thresh");
    ;
    double lk_d2_factor = Config::get<double>("lk_d2_factor"); // deviation of median disparity
    int BA_iteration = Config::get<int>("BA_iteration");
    double ba_outlier_thresh = Config::get<double>("ba_outlier_thresh"); // deviation of median disparity

    const int max_feature = Config::get<int>("max_feature");
    const double qualityLevel = Config::get<double>("qualityLevel");
    const double minDistance = Config::get<double>("minDistance");

    const double new_kf_dist_thresh = Config::get<double>("new_kf_dist_thresh");
    const double new_kf_angle_thresh = Config::get<double>("new_kf_angle_thresh");
    const int new_kf_nr_tracked_points = Config::get<int>("new_kf_nr_tracked_points");
    const int new_kf_nr_frames_inbtw = Config::get<int>("new_kf_nr_frames_inbtw");
    const int add_ba = Config::get<int>("do_bundle_adjustment");
    const int vis = Config::get<int>("visualize_tracking");
    const int add_mba = Config::get<int>("do_motion_only_bundle_adjustment");
    const double chi2_thresh =  Config::get<int>("chi2_thresh");

    //const int lk_half_patch_size = 5;
    //const double lk_photometric_thresh = (lk_half_patch_size * 2) * (lk_half_patch_size * 2) * 15 * 15;
    //const double lk_d2_factor = 1.5 * 1.5; // deviation of median disparity
    //const int BA_iteration = 1000;
    //const double ba_outlier_thresh = 1;

    M3d K;

    Initializer initializer;
    viso::Map map_;
    State state_;
    Sophus::SE3d k2f; //keyframe-to-frame motion
    Sophus::SE3d f2f; //last frame-to-frame motion
    Sophus::SE3d lf; //last frame motion
    Sophus::SE3d lkf; //last keyframe motion

    cv::Ptr<cv::GFTTDetector> featureDetector = cv::GFTTDetector::create(max_feature, qualityLevel, minDistance);

    std::thread ba_thread_;
    std::atomic<bool> do_ba_;
    std::mutex update_map_;

public:
    Viso(double fx, double fy, double cx, double cy)
    {
        K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        state_ = kInitialization;
        running = true;
    }

    std::atomic<bool> running;

    std::vector<Sophus::SE3d> poses;
    std::vector<Sophus::SE3d> poses_opt;
    std::vector<V3d> points_opt;
    std::vector<std::string> frame_time;
    std::vector<int> ref_key;
    std::vector<Sophus::SE3d> ref_pose;

    Keyframe::Ptr last_frame;

    ~Viso() = default;

    void OnNewFrame(Keyframe::Ptr cur_frame);

    inline std::vector<V3d> GetPoints()
    {
        std::vector<V3d> points;
        for (const auto& p : map_.GetPoints()) {
            points.push_back(p->GetWorldPos());
        }
        return points;
    }

    inline viso::Map* GetMap() { return &map_; }

private:
    void DirectPoseEstimationSingleLayer(int level, Keyframe::Ptr current_frame, Sophus::SE3d& T21);
    void DirectPoseEstimationMultiLayer(Keyframe::Ptr current_frame, Sophus::SE3d& T21);

    struct AlignmentPair {
        Keyframe::Ptr ref_frame;
        Keyframe::Ptr cur_frame;
        V2d uv_ref;
        V2d uv_cur;
    };

    void LKAlignment(Keyframe::Ptr current_frame, std::vector<V2d>& kp_before, std::vector<V2d>& kp_after, std::vector<int>& tracked_points, std::vector<AlignmentPair>& alignment_pairs);
    void LKAlignmentSingle(std::vector<AlignmentPair>& pairs, std::vector<bool>& success, std::vector<V2d>& kp, int level);
    void BA(bool map_only, int fix_cnt);
    void BA_KEY();
    bool IsKeyframe(Keyframe::Ptr keyframe, int nr_tracked_points);

    double CalculateVariance2(const double& nu, const Sophus::SE3d& T21,
        const std::vector<int>& tracked_points,
        const std::vector<AlignmentPair>& alignment_pairs);
    double RemoveOutliers(const Sophus::SE3d& T21,
        std::vector<int>& tracked_points,
        std::vector<AlignmentPair>& alignment_pairs);

    void MotionOnlyBA(Sophus::SE3d& T21, std::vector<int>& tracked_points, std::vector<AlignmentPair>& alignment_pairs);
};

#endif
