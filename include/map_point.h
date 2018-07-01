//
// Created by sage on 09.06.18.
//

#ifndef VISO_MAP_POINT_H
#define VISO_MAP_POINT_H

#include "keyframe.h"
#include "types.h"
#include <tuple>

class MapPoint {
public:
    using Ptr = std::shared_ptr<MapPoint>;

    MapPoint(V3d Pw)
        : Pw_(Pw)
    {
    }

    inline void AddObservation(Keyframe::Ptr keyframe, int idx)
    {
        observations_.push_back({ keyframe, idx });
    }

    inline const std::vector<std::pair<Keyframe::Ptr, int> >& GetObservations()
    {
        return observations_;
    }

    // Position in world coordinates.
    inline V3d GetWorldPos() { return Pw_; }
    inline void SetWorldPos(V3d pos) { Pw_ = pos; }

    // Direction in world coordinates.
    inline void SetDirection(V3d dir) { dir_ = dir; }
    inline V3d GetDirection() { return dir_; }
private:
    V3d Pw_;
    std::vector<std::pair<Keyframe::Ptr, int> > observations_;
    V3d dir_;
};

#endif //VISO_MAP_POINT_H
