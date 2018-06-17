
#include "keyframe.h"
#include "map_point.h"
#include "types.h"

#ifndef VISO_MAP_H
#define VISO_MAP_H

class Map {
private:
    std::vector<Keyframe::Ptr> keyframes_;
    std::vector<MapPoint::Ptr> points_;

public:
    Map() = default;
    ~Map() = default;

    inline void AddKeyframe(Keyframe::Ptr keyframe) { keyframes_.push_back(keyframe); }
    inline void AddPoint(MapPoint::Ptr map_point) { points_.push_back(map_point); }

    inline std::vector<Keyframe::Ptr> Keyframes() { return keyframes_; }
    inline std::vector<MapPoint::Ptr> GetPoints() { return points_; }
    inline std::vector<V3d> GetPoints3d()
    {
        std::vector<V3d> points3d;
        points3d.reserve(points_.size());
        for (int i = 0; i < points_.size(); ++i) {
            points3d.push_back(points_[i]->GetWorldPos());
        }
        return points3d;
    }
};

#endif