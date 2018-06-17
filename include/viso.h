
#ifndef VISO_VISO_H
#define VISO_VISO_H

#include "bundle_adjuster.h"
#include "frame_sequence.h"
#include "keyframe.h"
#include "map.h"
#include "ring_buffer.h"
#include <initializer.h>
#include <sophus/se3.hpp>
#include <tuple>

class Viso : public FrameSequence::FrameHandler {
private:
    enum State {
        kInitialization = 0,
        kRunning = 1,
        kFinished = 2
    };

    const int lk_half_patch_size = 5;
    const double lk_photometric_thresh = (lk_half_patch_size * 2) * (lk_half_patch_size * 2) * 15 * 15;
    const double lk_d2_factor = 1.5 * 1.5; // deviation of median disparity
    const int BA_iteration = 1000;
    const double ba_outlier_thresh = 1;

    M3d K;

    Initializer initializer;
    Map map_;
    State state_;

public:
    Viso(double fx, double fy, double cx, double cy)
    {
        K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        state_ = kInitialization;
    }

    std::vector<Sophus::SE3d> poses;
    std::vector<Sophus::SE3d> poses_opt;
    std::vector<V3d> points_opt;

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

    inline Map* GetMap() { return &map_; }

private:
    void DirectPoseEstimationSingleLayer(int level, Keyframe::Ptr current_frame, Sophus::SE3d& T21);
    void DirectPoseEstimationMultiLayer(Keyframe::Ptr current_frame, Sophus::SE3d& T21);

    struct AlignmentPair {
        Keyframe::Ptr ref_frame;
        Keyframe::Ptr cur_frame;
        V2d uv_ref;
        V2d uv_cur;
    };

    void LKAlignment(Keyframe::Ptr current_frame, std::vector<V2d>& kp_before, std::vector<V2d>& kp_after, std::vector<int>& tracked_points);
    void LKAlignmentSingle(std::vector<AlignmentPair>& pairs, std::vector<bool>& success, std::vector<V2d>& kp, int level);
    void BA(bool map_only, Keyframe::Ptr current_frame, const std::vector<V2d>& kp, const std::vector<int>& tracked_points);
};

#endif
