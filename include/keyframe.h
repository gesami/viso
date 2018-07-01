

#ifndef VISO_KEYFRAME_H
#define VISO_KEYFRAME_H

#include "config.h"
#include "types.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

class Keyframe {
private:
    static long next_id_;
    long id_;

    std::vector<cv::KeyPoint> keypoints_;
    std::vector<cv::KeyPoint> keypoints_df_;

    std::string time_;

    cv::Mat mat_;
    M3d R_, relatR_; //relative transformation between keyframe
    V3d T_, relatT_;
    M3d K_;
    int relatkey_;

    const int nr_pyramids_ = 4;
    const double pyramid_scale_ = 0.5;
    const double scales_[4] = { 1.0, 0.5, 0.25, 0.125 };
    // Some constants.
    int grid_row_;
    int grid_col_;
    int grid_size_ = Config::get<int>("grid_size");
    std::vector<bool> grid_occupy_; //record occupied grid
    std::vector<cv::Mat> pyramids_;

public:
    using Ptr = std::shared_ptr<Keyframe>;

    static constexpr const int nr_pyramids = 4;
    static constexpr const double pyramid_scale = 0.5;
    static const double scales[];

    Keyframe(cv::Mat mat, std::string timestamp)
        : mat_(mat)
        , time_(timestamp)
    {
        id_ = next_id_;
        ++next_id_;

        R_ = M3d::Identity();
        T_ = V3d::Zero();

        // Pyramids
        pyramids_.push_back(mat_);

        for (int i = 1; i < Keyframe::nr_pyramids; i++) {
            cv::Mat pyr;
            cv::pyrDown(pyramids_[i - 1], pyr,
                cv::Size(pyramids_[i - 1].cols * Keyframe::pyramid_scale, pyramids_[i - 1].rows * Keyframe::pyramid_scale));
            pyramids_.push_back(pyr);
        }

        //Set grid
        grid_col_ = ceil(mat_.cols / (double)grid_size_);
        grid_row_ = ceil(mat_.rows / (double)grid_size_);
        grid_occupy_.resize(grid_col_ * grid_row_, false);
    }

    ~Keyframe() = default;

    inline double GetPixelValue(const double& x, const double& y, int level = 0)
    {
        U8* data = &pyramids_[level].data[int(y) * pyramids_[level].step + int(x)];
        double xx = x - floor(x);
        double yy = y - floor(y);
        return double(
            (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[pyramids_[level].step] + xx * yy * data[pyramids_[level].step + 1]);
    }

    inline V2d GetGradient(const double& u, const double& v, int level = 0)
    {
        double dx = 0.5 * (GetPixelValue(u + 1, v, level) - GetPixelValue(u - 1, v, level));
        double dy = 0.5 * (GetPixelValue(u, v + 1, level) - GetPixelValue(u, v - 1, level));
        return V2d(dx, dy);
    }

    inline long GetId()
    {
        return id_;
    }

    inline bool IsInside(const V3d& point, int level = 0)
    {
        V2d uv = Project(point, level);
        return IsInside(uv.x(), uv.y(), level);
    }

    inline V3d PixelToImagePlane(const V2d& uv, int level)
    {
        double x = (uv.x() / scales[level] - K_(0, 2)) / K_(0, 0);
        double y = (uv.y() / scales[level] - K_(1, 2)) / K_(1, 1);
        return V3d{ x, y, 1.0 };
    }

    inline V3d WorldToCamera(const V3d& Pw)
    {
        V3d Pc = R_ * Pw + T_;
        return Pc;
    }

    inline V3d CameraToWorld(const V3d& Pc)
    {
        V3d Pw = R_.transpose() * (Pc - T_);
        return Pw;
    }

    inline bool IsInside(const double& u, const double& v, int level = 0)
    {
        return u >= 0 && u < pyramids_[level].cols && v >= 0 && v < pyramids_[level].rows;
    }

    inline V2d Project(const V3d& point, int level)
    {
        V3d uv1 = R_ * point + T_;
        uv1 /= uv1.z();
        double u = scales[level] * (uv1.x() * K_(0, 0) + K_(0, 2));
        double v = scales[level] * (uv1.y() * K_(1, 1) + K_(1, 2));
        return V2d{ u, v };
    }

    // Calculate the viewing angle, i.e. the angle between the vector from the
    // origin of the camera to the point and the camera's Z-axis.
    inline double ViewingAngle(const V3d& Pw)
    {
        V3d Pc = R_ * Pw + T_;
        Pc.normalize();
        return std::acos(Pc.z());
    }

    inline std::vector<cv::KeyPoint>& Keypoints() { return keypoints_; }
    inline std::vector<cv::KeyPoint>& GetKeypointsDF() { return keypoints_df_; }

    inline const cv::Mat& Mat() { return mat_; }
    inline Sophus::SE3d GetPose() { return Sophus::SE3d(R_, T_); }
    inline Sophus::SE3d SetPose(Sophus::SE3d pose)
    {
        R_ = pose.rotationMatrix();
        T_ = pose.translation();
    }
    inline M3d GetR() { return R_; }
    inline V3d GetT() { return T_; }
    inline void SetT(V3d T) { T_ = T; }
    inline void SetR(M3d R) { R_ = R; }
    inline void SetrelatT(V3d T) { relatT_ = T; }
    inline void SetrelatR(M3d R) { relatR_ = R; }
    inline void SetrelatKey(int a) { relatkey_ = a; }
    inline M3d GetK() { return K_; }
    inline void SetK(M3d K) { K_ = K; }
    inline const std::string& GetTime() { return time_; }
    inline double GetScale(int level) { return Keyframe::scales[level]; }

    inline const std::vector<cv::Mat>& GetPyramids() { return pyramids_; }

    static long GetNextId()
    {
        return next_id_;
    }

    inline int AddKeypoint(cv::KeyPoint kp)
    {
        keypoints_.push_back(kp);
        return keypoints_.size() - 1;
    }

    inline int AddKeypointForDepthFiler(cv::KeyPoint kp)
    {
        keypoints_df_.push_back(kp);
        return keypoints_df_.size();
    }

    void SetOccupied(std::vector<V3d> mp);
    void AddNewFeatures(std::vector<cv::KeyPoint> newfts);
};

#endif
