//
// Created by sage on 04.06.18.
//

#ifndef FINAL_COMMON_H
#define FINAL_COMMON_H

#include "types.h"

inline M4d MakeSE3(const M3d& R, const V3d& T)
{
    M4d result = M4d::Zero();
    result.block<3, 3>(0, 0) = R;
    result.block<3, 1>(0, 3) = T;
    result(3, 3) = 1;
    return result;
}

inline M34d MakePI0()
{
    M34d result = M34d::Zero();
    result.block<3, 3>(0, 0) = M3d::Identity();
    return result;
}

inline M3d Hat(V3d v)
{
    M3d result;
    result << 0.0, -v(2), v(1),
        v(2), 0.0, -v(0),
        -v(1), v(0), 0.0;
    return result;
}

inline double GetPixelValue(const cv::Mat& mat, const double& x, const double& y)
{
    const int y_int = (int)y;
    const int x_int = (int)x;

    assert(x_int >= 0 && x_int < mat.cols);
    assert(y_int >= 0 && y_int < mat.rows);

    U8* data = &mat.data[y_int * mat.step + x_int];
    double xx = x - floor(x);
    double yy = y - floor(y);
    return double(
        (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[mat.step] + xx * yy * data[mat.step + 1]);
}

inline V2d GetImageGradient(const cv::Mat& mat, const double& u, const double& v)
{
    double dx = 0.5 * (GetPixelValue(mat, u + 1, v) - GetPixelValue(mat, u - 1, v));
    double dy = 0.5 * (GetPixelValue(mat, u, v + 1) - GetPixelValue(mat, u, v - 1));
    return V2d(dx, dy);
}

inline double GetImageGradientX(const cv::Mat& mat, const double& u, const double& v)
{
    return 0.5 * (GetPixelValue(mat, u + 1, v) - GetPixelValue(mat, u - 1, v));
}

inline double GetImageGradientY(const cv::Mat& mat, const double& u, const double& v)
{
    return 0.5 * (GetPixelValue(mat, u, v + 1) - GetPixelValue(mat, u, v - 1));
}

template <typename T>
inline T CalculateMedian(std::vector<T> vec)
{
    std::size_t size = vec.size();
    if (size == 0) {
        return T();
    }
    std::sort(vec.begin(), vec.end());
    T median = T();

    if (size % 2 == 0) {
        median = (vec[size / 2 - 1] + vec[size / 2]) / 2;
    } else {
        median = vec[size / 2];
    }

    return median;
}

#endif //FINAL_COMMON_H
