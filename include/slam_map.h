#ifndef VISO_MAP_H
#define VISO_MAP_H

#include "keyframe.h"
#include "map_point.h"
#include "ring_buffer.h"
#include "types.h"

#include <mutex>

#define LOCK() std::lock_guard<std::mutex> lock(mut_)

namespace viso {
class Map {
private:
    std::vector<Keyframe::Ptr> keyframes_;
    std::vector<MapPoint::Ptr> points_;
    std::mutex mut_;

    ring_buffer<5, Keyframe::Ptr> last_keyframes_;
    ring_buffer<5, int> key_index;
    Keyframe::Ptr current_frame_;

public:
    Map() = default;

    ~Map() = default;

    inline void AddKeyframe(Keyframe::Ptr keyframe)
    {
        LOCK();
        keyframes_.push_back(keyframe);
        last_keyframes_.push(keyframe);
        key_index.push(keyframes_.size()-1);
    }

    inline void AddPoint(MapPoint::Ptr map_point)
    {
        LOCK();
        points_.push_back(map_point);
    }

    inline std::vector<Keyframe::Ptr> Keyframes()
    {
        LOCK();
        return keyframes_;
    }

    inline std::vector<Keyframe::Ptr> LastKeyframes()
    {
        LOCK();
        return last_keyframes_.to_vector();
    }

    inline std::vector<int> Keyindex()
    {
        LOCK();
        return key_index.to_vector();
    }

    inline std::vector<Sophus::SE3d> GetLastPoses()
    {
        LOCK();
        std::vector<Sophus::SE3d> poses;
        poses.reserve(last_keyframes_.size());
        for (int i = 0; i < last_keyframes_.size(); ++i) {
            poses.push_back(last_keyframes_[i]->GetPose());
        }
        return poses;
    }

    inline std::vector<MapPoint::Ptr> GetPoints()
    {
        LOCK();
        return points_;
    }

    inline std::vector<V3d> GetPoints3d()
    {
        LOCK();
        std::vector<V3d> points3d;
        points3d.reserve(points_.size());
        for (int i = 0; i < points_.size(); ++i) {
            points3d.push_back(points_[i]->GetWorldPos());
        }
        return points3d;
    }

    inline std::vector<Sophus::SE3d> GetPoses()
    {
        LOCK();
        std::vector<Sophus::SE3d> poses;
        poses.reserve(keyframes_.size());
        for (const auto& kf : keyframes_) {
            poses.push_back(kf->GetPose());
        }
        return poses;
    }

    inline Sophus::SE3d GetLastPose()
    {
        LOCK();
        if (last_keyframes_.size() == 0) {
            return Sophus::SE3d();
        }

        return last_keyframes_.last()->GetPose();
    }

    inline void SetCurrent(Keyframe::Ptr cur_frame)
    {
        current_frame_ = cur_frame;
    }

    inline Keyframe::Ptr GetCurrent()
    {
        return current_frame_;
    }

    inline int GetKeyid()
    {
        return (keyframes_.size()-1);
    }
};
}

#endif
