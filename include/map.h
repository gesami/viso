#ifndef VISO_MAP_H
#define VISO_MAP_H

#include "keyframe.h"
#include "map_point.h"
#include "types.h"
#include <mutex>

#define LOCK() std::lock_guard<std::mutex> lock(mut_)

class Map {
private:
    std::vector<Keyframe::Ptr> keyframes_;
    std::vector<MapPoint::Ptr> points_;
    std::mutex mut_;

public:
    Map() = default;
    ~Map() = default;

    inline void AddKeyframe(Keyframe::Ptr keyframe)
    {
        LOCK();
        keyframes_.push_back(keyframe);
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
};

#endif