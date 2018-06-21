#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/timer.hpp>
#include <fstream>
#include <iostream>
#include <sophus/se3.hpp>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "keyframe.h"
#include "slam_map.h"
#include "types.h"
#include <common.h>

using namespace cv;
using namespace Eigen;
using namespace Sophus;
using namespace std;

class depth_filter {
    // parameters
    const int boarder = 20; // boarder length
    const int width = 640; // width
    const int height = 480; // height

    // intrinstic parameter for rgb dataset

    const int ncc_window_size = 7; // NCC half window size
    const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC window area
    const double min_cov = 0.1; // convergence determination: minimum variance
    const double max_cov = 10; // divergence determination: maximum variance

    const int max_iterations = 5;

    double fx;
    double fy;
    double cx;
    double cy;

private:
    Keyframe::Ptr ref_frame_;
    std::vector<cv::KeyPoint> kp_;
    std::vector<double> depths_;
    std::vector<double> depths_cov_;
    std::vector<int> status_;
    int current_iteration_;

public:
    depth_filter(Keyframe::Ptr ref_frame)
        : ref_frame_(ref_frame)
        , kp_(ref_frame_->GetKeypointsDF())
    {
        M3d K = ref_frame_->GetK();

        fx = K(0, 0); // focal length x
        fy = K(1, 1); // focal length y
        cx = K(0, 2); // optical center x
        cy = K(1, 2); // optical center y

        int nr_keypoints = kp_.size();
        depths_.resize(nr_keypoints);
        depths_cov_.resize(nr_keypoints);
        status_.resize(nr_keypoints);

        for (int i = 0; i < nr_keypoints; ++i) {
            depths_[i] = 3.0;
            depths_cov_[i] = 5.0;
            status_[i] = 0;
        }

        current_iteration_ = 0;
    }

    bool IsDone()
    {
        return current_iteration_ >= max_iterations;
    }

    // This should be called, after the filter IsDone() == true
    void UpdateMap(viso::Map* map)
    {
        for (int i = 0; i < kp_.size(); ++i) {
            if (status_[i] == 1) {
                V3d P = ref_frame_->GetPose().inverse() * (depths_[i] * px2cam({ kp_[i].pt.x, kp_[i].pt.y }));
                int kp_idx = ref_frame_->AddKeypoint(kp_[i]);
                MapPoint::Ptr map_point = std::make_shared<MapPoint>(P);
                map_point->AddObservation(ref_frame_, kp_idx);
                map->AddPoint(map_point);

                status_[i] = 2; // added to map
            }
        }
    }

    void Update(Keyframe::Ptr cur_frame)
    {
        if (IsDone()) {
            return;
        }

        ++current_iteration_;

        SE3d pose_curr_TWC = cur_frame->GetPose();
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * ref_frame_->GetPose(); // change world coordinate： T_C_W * T_W_R = T_C_R
        // plot the search process for each point
        for (int i = 0; i < kp_.size(); ++i) {
            if (status_[i] != 0) {
                continue;
            }
            int x = kp_[i].pt.x;
            int y = kp_[i].pt.y;
            // set last parameter to one to show polar line search process
            status_[i] = update(ref_frame_->Mat(), cur_frame->Mat(), pose_T_C_R, depths_[i], depths_cov_[i], x, y, false);
        }
    }

private:
    // bilinear grayscale interpolation
    inline double getBilinearInterpolatedValue(const Mat& img, const Vector2d& pt)
    {
        uchar* d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
        double xx = pt(0, 0) - floor(pt(0, 0));
        double yy = pt(1, 0) - floor(pt(1, 0));
        return ((1 - xx) * (1 - yy) * double(d[0]) + xx * (1 - yy) * double(d[1]) + (1 - xx) * yy * double(d[img.step]) + xx * yy * double(d[img.step + 1])) / 255.0;
    }

    // pixel to camera coordinate
    inline Vector3d px2cam(const Vector2d px)
    {
        return Vector3d(
            (px(0, 0) - cx) / fx,
            (px(1, 0) - cy) / fy,
            1);
    }

    // camera coordinate to pixel
    inline Vector2d cam2px(const Vector3d p_cam)
    {
        return Vector2d(
            p_cam(0, 0) * fx / p_cam(2, 0) + cx,
            p_cam(1, 0) * fy / p_cam(2, 0) + cy);
    }

    // check if a point is in image boundary
    inline bool inside(const Vector2d& pt)
    {
        return pt(0, 0) >= boarder && pt(1, 0) >= boarder
            && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
    }

    // write depth information
    // update depth mat
    int update(const Mat& ref, const Mat& curr, const SE3d& T_C_R, double& depth, double& depth_cov, int x, int y, bool show)
    {
        // uncommend to iterate for whole mat
        /* #pragma omp parallel for
    for ( int x=boarder; x<width-boarder; x++ )
#pragma omp parallel for
        for ( int y=boarder; y<height-boarder; y++ )
        { */

        // int x = 250; int y = 250;
        // judge the covariance of the point
        if (depth_cov < min_cov) {
            cout << " depth converged " << endl;
            return 1;
        };
        if (depth_cov > max_cov) {
            cout << " depth  divergered " << endl;
            return -1;
        }
        // continue;
        // search the match in polar line
        Vector2d pt_curr;
        bool ret = epipolarSearch(
            ref,
            curr,
            T_C_R,
            Vector2d(x, y),
            depth,
            sqrt(depth_cov),
            pt_curr,
            show);

        if (ret == false) { // matching fails
            cout << " matching fails " << endl;
            return -1;
        }
        // continue;

        // uncomment to show match
        if (show == 1)
            showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);

        // match successful, update the depth filter
        updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, depth, depth_cov);
        cout << " after updating depth is " << depth << endl;
        cout << " after updating convariance is " << depth_cov << endl;
        // }

        return 0;
    }

    // polar line search
    bool epipolarSearch(
        const Mat& ref, const Mat& curr,
        const SE3d& T_C_R, const Vector2d& pt_ref,
        const double& depth_mu, const double& depth_cov,
        Vector2d& pt_curr, bool show)
    {
        Vector3d f_ref = px2cam(pt_ref);
        f_ref.normalize();
        Vector3d P_ref = f_ref * depth_mu; // the p vecotr in reference frame

        Vector2d px_mean_curr = cam2px(T_C_R * P_ref); // the projected pixel according the depth average
        double d_min = depth_mu - 4 * depth_cov, d_max = depth_mu + 4 * depth_cov;
        if (d_min < 0.1)
            d_min = 0.1;
        Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min)); // the projected pixel according the smallest depth value
        Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max)); // the projected pixel according the smallest depth value

        Vector2d epipolar_line = px_max_curr - px_min_curr; // epipolar line
        Vector2d epipolar_direction = epipolar_line; // direction of epipolar line
        epipolar_direction.normalize();
        double half_length = 0.5 * epipolar_line.norm(); // half length of epipolar line
        if (half_length > 100)
            half_length = 100; // dont search too much

        // show epipolar line
        if (show == 1)
            showEpipolarLine(ref, curr, pt_ref, px_min_curr, px_max_curr);

        // search around the epipolar line
        double best_ncc = -1.0;
        Vector2d best_px_curr;
        for (double l = -half_length; l <= half_length; l += 0.7) // l+=sqrt(2)
        {
            Vector2d px_curr = px_mean_curr + l * epipolar_direction; // points to be matched
            if (!inside(px_curr))
                continue;
            // calculated NCC
            double ncc = NCC(ref, curr, pt_ref, px_curr);
            if (ncc > best_ncc) {
                best_ncc = ncc;
                best_px_curr = px_curr;
            }
        }
        if (best_ncc < 0.95f) // only choose large ncc values
            return false;
        pt_curr = best_px_curr;
        return true;
    }

    double NCC(
        const Mat& ref, const Mat& curr,
        const Vector2d& pt_ref, const Vector2d& pt_curr)
    {
        // average should be zero
        double mean_ref = 0, mean_curr = 0;
        vector<double> values_ref, values_curr; // average value of reference frame and current frame
        for (int x = -ncc_window_size; x <= ncc_window_size; x++)
            for (int y = -ncc_window_size; y <= ncc_window_size; y++) {
                double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
                mean_ref += value_ref;

                double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y));
                mean_curr += value_curr;

                values_ref.push_back(value_ref);
                values_curr.push_back(value_curr);
            }

        mean_ref /= ncc_area;
        mean_curr /= ncc_area;

        // calculate Zero mean NCC
        double numerator = 0, demoniator1 = 0, demoniator2 = 0;
        for (int i = 0; i < values_ref.size(); i++) {
            double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
            numerator += n;
            demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
            demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
        }
        return numerator / sqrt(demoniator1 * demoniator2 + 1e-10); // numerical stability
    }

    bool updateDepthFilter(
        const Vector2d& pt_ref,
        const Vector2d& pt_curr,
        const SE3d& T_C_R,
        double& depth,
        double& depth_cov)
    {
        // depth calculation using triangulation
        SE3d T_R_C = T_C_R.inverse();
        Vector3d f_ref = px2cam(pt_ref);
        f_ref.normalize();
        Vector3d f_curr = px2cam(pt_curr);
        f_curr.normalize();

        // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
        // => [ f_ref^T f_ref, -f_ref^T f_cur ] [d_ref] = [f_ref^T t]
        //    [ f_cur^T f_ref, -f_cur^T f_cur ] [d_cur] = [f_cur^T t]

        Vector3d t = T_R_C.translation();
        Vector3d f2 = T_R_C.rotationMatrix() * f_curr;
        //Vector3d f2 = T_R_C.rotation() * f_curr;
        Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));
        double A[4];
        A[0] = f_ref.dot(f_ref);
        A[2] = f_ref.dot(f2);
        A[1] = -A[2];
        A[3] = -f2.dot(f2);
        double d = A[0] * A[3] - A[1] * A[2];
        Vector2d lambdavec = Vector2d(A[3] * b(0, 0) - A[1] * b(1, 0),
                                 -A[2] * b(0, 0) + A[0] * b(1, 0))
            / d;
        Vector3d xm = lambdavec(0, 0) * f_ref;
        Vector3d xn = t + lambdavec(1, 0) * f2;
        Vector3d d_esti = (xm + xn) / 2.0; // depth vector calculated by triangulation
        double depth_estimation = d_esti.norm(); // depth value

        // uncertainty calculation (one pixel as error)
        Vector3d p = f_ref * depth_estimation;
        Vector3d a = p - t;
        double t_norm = t.norm();
        double a_norm = a.norm();
        double alpha = acos(f_ref.dot(t) / t_norm);
        double beta = acos(-a.dot(t) / (a_norm * t_norm));
        double beta_prime = beta + atan(1 / fx);
        double gamma = M_PI - alpha - beta_prime;
        double p_prime = t_norm * sin(beta_prime) / sin(gamma);
        double d_cov = p_prime - depth_estimation;
        double d_cov2 = d_cov * d_cov;

        // gaussian fusion
        double mu = depth;
        double sigma2 = depth_cov;

        double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
        double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

        depth = mu_fuse;
        depth_cov = sigma_fuse2;

        return true;
    }

    void showEpipolarMatch(const Mat& ref, const Mat& curr, const Vector2d& px_ref, const Vector2d& px_curr)
    {
        Mat ref_show, curr_show;
        cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
        cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

        cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
        cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

        imshow("ref", ref_show);
        imshow("curr", curr_show);
        waitKey(0);
    }

    void showEpipolarLine(const Mat& ref, const Mat& curr, const Vector2d& px_ref, const Vector2d& px_min_curr, const Vector2d& px_max_curr)
    {

        Mat ref_show, curr_show;
        cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
        cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

        cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
        cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
        cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
        cv::line(curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), Scalar(0, 255, 0), 1);

        imshow("ref", ref_show);
        imshow("curr", curr_show);
        waitKey(0);
    }
};
