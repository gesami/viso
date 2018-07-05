//
// Created by sage on 09.06.18.
//

#ifndef VISO_MAP_POINT_H
#define VISO_MAP_POINT_H

#include "types.h"
#include <tuple>

class Keyframe;
using KeyframePtr = std::shared_ptr<Keyframe>;

class MapPoint {
public:
    using Ptr = std::shared_ptr<MapPoint>;

    MapPoint(V3d Pw)
        : Pw_(Pw)
    {
    }

    inline void AddObservation(KeyframePtr keyframe, int idx)
    {
        observations_.push_back({ keyframe, idx });
    }

    inline const std::vector<std::pair<KeyframePtr, int> >& GetObservations()
    {
        return observations_;
    }

    inline void EraseObservation(int idx)
    {
        observations_.erase(observations_.begin() + idx);
    }
    
    // Position in world coordinates.
    inline V3d GetWorldPos() { return Pw_; }
    inline void SetWorldPos(V3d pos) { Pw_ = pos; }

    // Direction in world coordinates.
    inline void SetDirection(V3d dir) { dir_ = dir; }
    inline V3d GetDirection() { return dir_; }
private:
    V3d Pw_;
    std::vector<std::pair<KeyframePtr, int> > observations_;
    V3d dir_;
};

#endif //VISO_MAP_POINT_H
