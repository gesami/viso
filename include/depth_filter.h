#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/timer.hpp>
#include <fstream>
#include <iostream>
#include <sophus/se3.hpp>
#include <vector>
#include <cmath>
#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "keyframe.h"
#include "slam_map.h"
#include "types.h"
#include <common.h>
#include <config.h>

using namespace cv;
using namespace Eigen;
using namespace Sophus;
using namespace std;

class depth_filter {
    // parameters
    const int boarder = 20; // boarder length
    // const int width = 640; // width
    //const int height = 480; // height
    const int ncc_window_size = 10; // NCC half window size
    const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC window area
    // const double min_cov = 0.1; // convergence determination: minimum variance
    const double max_cov = 10; // divergence determination: maximum variance
    double z_range = 5;
    double init_depth = 1; 
    double init_cov2 = 5;
    double min_cov = z_range/200.0;
    
    double photometric_thresh = Config::get<double>("photometric_thresh"); 
    double width = Config::get<double>("image_width");
    double height = Config::get<double>("image_height");
    double init_a = Config::get<double>("init_a");
    double init_b = Config::get<double>("init_b");
    int df_window_size = Config::get<int>("df_window_size");
    double photo_area = (double) (2 * df_window_size + 1) * (2 * df_window_size + 1); 
    const double reprojection_thresh_ = Config::get<double>("reprojection_thresh");
    
    
    // intrinstic parameter for rgb dataset
    double fx;
    double fy;
    double cx;
    double cy;

private:
    Keyframe::Ptr ref_frame_;
    std::vector<cv::KeyPoint> kp_;
    std::vector<double> depths_;      // inverse gaussian coordinate
    std::vector<double> depths_cov_;  
    std::vector<double> beta_a;       // beta distribution
	std::vector<double> beta_b;
    std::vector<int> status_;
    int current_iteration_;

    Keyframe::Ptr cur_frame_;

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
        beta_a.resize(nr_keypoints);
        beta_b.resize(nr_keypoints);
        status_.resize(nr_keypoints);

        for (int i = 0; i < nr_keypoints; ++i) {
            depths_[i] = 1.0/init_depth;
            depths_cov_[i] = init_cov2;
            beta_a[i] = init_a;
            beta_b[i] = init_b;
            status_[i] = 0;
        }
        current_iteration_ = 0;
    }

    // This should be called, after the filter IsDone() == true
    void UpdateMap(viso::Map* map, Keyframe::Ptr cur_frame)
    {
        Mat ref = ref_frame_->GetMat();    // reference image
        std::vector<Keyframe::Ptr> lkf = map->LastKeyframes();
        // Mat curr = cur_frame->GetMat();
        // double photo_error = 0;
        for (int i = 0; i < kp_.size(); ++i) {
            if (status_[i] == 1) {
                V3d P = ref_frame_->GetPose().inverse() * (depths_[i] * px2cam({ kp_[i].pt.x, kp_[i].pt.y }));
                int bad_observation=0; 
                // use only last two keyframes
                for(int j=lkf.size()-2; j<lkf.size();j++){
                //for(int j=0; j<lkf.size();j++){
                    double photo_error = 0;
                    Mat curr = lkf[j]->GetMat();
                    SE3d pose_curr_TWC = lkf[j]->GetPose();
                    V3d pc = pose_curr_TWC * P;
                    V2d p = cam2px(pc);
                    if (p(0)<boarder || p(0)>(width-boarder) || p(1)<boarder || p(1)>(height-boarder) )
                        continue;
                    for (int u=-df_window_size; u<df_window_size; u++){
                        for (int v=-df_window_size; v<df_window_size; v++){
                                photo_error += abs( GetPixelValue(ref, kp_[i].pt.x + u, kp_[i].pt.y + v) - GetPixelValue(curr, p(0)+u, p(1)+v) );
                        }
                    }
                    if ( photo_error > photometric_thresh*photo_area)
                        bad_observation++;
                }
                cout << " bad observation number " << bad_observation << endl;
                if( bad_observation > 0){
                    cout <<  "deleting outlier !!!" << endl;
                    status_[i] = 0;
                    continue;
                }
                assert(depths_[i]<10 && depths_[i]>0);
                int kp_idx = ref_frame_->AddKeypoint(kp_[i]);
                MapPoint::Ptr map_point = std::make_shared<MapPoint>(P);
                map_point->AddObservation(ref_frame_, kp_idx);
                map->AddPoint(map_point);
                status_[i] = 2; // added to map
            }
        }
    }
    
    // TODO clear the outliers
    void Clear_outliers();
    
    void Update(Keyframe::Ptr cur_frame)
    {
        /*
        if (IsDone()) {
            return;
        } */

        cur_frame_ = cur_frame;

        SE3d pose_curr_TWC = cur_frame->GetPose();
        SE3d pose_T_C_R = pose_curr_TWC * ref_frame_->GetPose().inverse(); // change world coordinate： T_C_W * T_W_R = T_C_R
        // plot the search process for each point
        for (int i = 0; i < kp_.size(); ++i) {
            if (status_[i] != 0) {
                continue;
            }
            int x = kp_[i].pt.x;
            int y = kp_[i].pt.y;
            // set last parameter to one to show polar line search process
            status_[i] = update(ref_frame_->Mat(), cur_frame->Mat(), pose_T_C_R, depths_[i], depths_cov_[i], beta_a[i], beta_b[i], x, y, false);
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

    // update depth mat
    int update(const Mat& ref, const Mat& curr, const SE3d& T_C_R, double& depth, double& depth_cov, double& beta_a, double& beta_b, int x, int y, bool show)
    {
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
            //cout << " matching fails " << endl;
            // return -1;
            return 0;
        }
        // continue;

        // uncomment to show match
        if (show == 1)
            showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);

        // match successful, update the depth filter
        updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, depth, depth_cov, beta_a, beta_b);
//        cout << " after updating depth is " << depth << endl;
//        cout << " after updating convariance is " << depth_cov << endl;

        if (depth_cov < min_cov) {
            V3d P = ref_frame_->GetPose().inverse() * (depth * px2cam({x, y }));
            V2d proj_to_cur = cur_frame_->Project(P, 0);

            double dx = proj_to_cur.x() - pt_curr.x();
            double dy = proj_to_cur.y() - pt_curr.y();
            double proj_error = std::sqrt(dx * dx + dy * dy);

            if(proj_error > 3 * reprojection_thresh_) {
              cout << " depth outlier " << endl;
              return -1;
            }
            cout << " depth converged " << endl;
            return 1;
        };
        
        if (depth_cov > max_cov) {
            cout << " depth  divergered " << endl;
            return -1;
        }
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
        // double d_min = depth_mu - 4 * depth_cov, d_max = depth_mu + 4 * depth_cov;
        // if (d_min < 0.1) d_min = 0.1;
        double d_min = 1.0/( depth_mu + depth_cov );
        double d_max = 1.0/std::max( depth_mu - depth_cov, (double)0.0000001f);
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
        double& depth_cov,
        double& beta_a, 
        double& beta_b)
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
        Vector2d db = Vector2d(t.dot(f_ref), t.dot(f2));
        double A[4];
        A[0] = f_ref.dot(f_ref);
        A[2] = f_ref.dot(f2);
        A[1] = -A[2];
        A[3] = -f2.dot(f2);
        double d = A[0] * A[3] - A[1] * A[2];
        Vector2d lambdavec = Vector2d(A[3] * db(0, 0) - A[1] * db(1, 0),
                                 -A[2] * db(0, 0) + A[0] * db(1, 0))
            / d;
        Vector3d xm = lambdavec(0, 0) * f_ref;
        Vector3d xn = t + lambdavec(1, 0) * f2;
        Vector3d d_esti = (xm + xn) / 2.0; // depth vector calculated by triangulation
        double depth_estimation = d_esti.norm(); // depth value

        // uncertainty calculation (one pixel as error)
        Vector3d p = f_ref * depth_estimation;
        Vector3d da = p - t;
        double t_norm = t.norm();
        double a_norm = da.norm();
        double alpha = acos(f_ref.dot(t) / t_norm);
        double beta = acos(-da.dot(t) / (a_norm * t_norm));
        double beta_prime = beta + atan(1 / fx);
        double gamma = M_PI - alpha - beta_prime;
        double p_prime = t_norm * sin(beta_prime) / sin(gamma);
        double d_cov = p_prime - depth_estimation;
        // double d_cov2 = d_cov * d_cov;
        // gaussian fusion
        // double mu = depth;
        // double sigma2 = depth_cov;
        // parameters for fusion
        double mu = depth;
        double sigma2 = depth_cov;
        double tau2 = d_cov*d_cov; 
        double a = beta_a;
	    double b = beta_b;
        // fusion
        double norm_scale = sqrt(sigma2 + tau2);
        if(std::isnan(norm_scale))
     	  return false;
        boost::math::normal_distribution<float> nd(mu, norm_scale);
    	double s2 = 1./(1./sigma2 + 1./tau2);
    	double m = s2*(mu/sigma2 + depth_estimation/tau2);
    	double C1 = a/(a + b) * boost::math::pdf(nd, depth_estimation);
    	double C2 = b/(a + b) * 1./z_range;
    	double normalization_constant = C1 + C2;
    	C1 /= normalization_constant;
    	C2 /= normalization_constant;
    	double f = C1*(a+1.)/(a+b+1.) + C2*a/(a+b+1.);
    	float e = C1*(a+1.)*(a+2.)/((a+b+1.)*(a+b+2.))
        	  + C2*a*(a+1.0f)/((a+b+1.0f)*(a+b+2.0f));
    	// update parameters
    	double mu_fuse = C1*m+C2*mu;
    	double sigma_fuse2 = C1*(s2 + m*m) + C2*(sigma2 + mu*mu) - mu_fuse*mu_fuse;
        depth = mu_fuse;
        depth_cov = sigma_fuse2;
    	beta_a = (e-f)/(f-e/f);
    	beta_b =  a*(1.0f-f)/f;

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

