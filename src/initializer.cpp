//
// Created by sage on 16.06.18.
//

#include <initializer.h>
#include <opencv2/core/eigen.hpp>

Initializer::Initializer()
{
    frame_cnt_ = 0;
    reset_after_ = 20;
    iterations_ = 10;
    half_patch_size_ = 10;
    photometric_thresh_ = (half_patch_size_ * 2) * (half_patch_size_ * 2) * 15 * 15; // squared error for the whole patch, 15 gray values per pixel
    min_good_cnt_ = 100;
    disparity2_thresh_ = 35 * 35; // squared, 15 pixels
    reprojection_thresh_ = 0.3;
}

bool Initializer::InitializeMap(Keyframe::Ptr cur_frame, Map* map, const cv::Mat& display)
{
    if (frame_cnt_ > 0 && frame_cnt_ <= reset_after_) {
        ++frame_cnt_;

        int good_cnt = 0;
        OpticalFlowMultiLevel(ref_frame_.get(),
            cur_frame.get(), ref_kp_,
            cur_kp_, track_success_, good_cnt,
            true);

        Visualize(display, ref_kp_, cur_kp_, track_success_, good_cnt);

        if (good_cnt < min_good_cnt_) {
            return false;
        }

        double disparity2 = CalculateDisparity2(ref_kp_, cur_kp_, track_success_);

        if (disparity2 == 0 || disparity2 < disparity2_thresh_) {
            return false;
        }

        K_ = cur_frame->GetK();
        const double fx = K_(0, 0);
        const double fy = K_(1, 1);
        const double cx = K_(0, 2);
        const double cy = K_(1, 2);

        std::vector<cv::Point2f> p1;
        std::vector<cv::Point2f> p2;
        std::vector<cv::KeyPoint> ref_kp_new;
        std::vector<cv::KeyPoint> cur_kp_new;

        p1.reserve(good_cnt);
        p2.reserve(good_cnt);
        ref_kp_new.reserve(good_cnt);
        cur_kp_new.reserve(good_cnt);

        for (int i = 0; i < ref_kp_.size(); ++i) {
            if (track_success_[i]) {
                p1.push_back({ (ref_kp_[i].pt.x - (float)cx) / (float)fx, (ref_kp_[i].pt.y - (float)cy) / (float)fy });
                p2.push_back({ (cur_kp_[i].pt.x - (float)cx) / (float)fx, (cur_kp_[i].pt.y - (float)cy) / (float)fy });
                ref_kp_new.push_back(ref_kp_[i]);
                cur_kp_new.push_back(cur_kp_[i]);
            }
        }

        ref_kp_ = ref_kp_new;
        cur_kp_ = cur_kp_new;

        good_cnt = 0;
        M3d R;
        V3d T;
        PoseEstimation(p1, p2, R, T, track_success_, good_cnt);

        if (good_cnt < min_good_cnt_) {
            return false;
        }

        points3d_.reserve(good_cnt);

        Reconstruct(p1, p2, R, T, track_success_, good_cnt, points3d_);

        if (good_cnt < min_good_cnt_) {
            return false;
        }

        NormalizeDepth(T, points3d_);

        map->AddKeyframe(ref_frame_);
        map->AddKeyframe(cur_frame);

        cur_frame->SetR(R);
        cur_frame->SetT(T);

        std::cout << "Rotation: \n"
                  << R << "\n";
        std::cout << "Translation: \n"
                  << T << "\n";

        std::cout << "Points: \n";

        for (const auto& p : points3d_) {
            std::cout << p << "\n";
        }

        int cnt = 0;
        for (int i = 0; i < p1.size(); ++i) {
            if (track_success_[i]) {
                ref_frame_->AddKeypoint(ref_kp_[i]);
                cur_frame->AddKeypoint(cur_kp_[i]);
                MapPoint::Ptr map_point = std::make_shared<MapPoint>(points3d_[cnt]);
                map_point->AddObservation(ref_frame_, cnt);
                map_point->AddObservation(cur_frame, cnt);
                map->AddPoint(map_point);
                ++cnt;
            }
        }

        return true;
    } else {
        ref_kp_.clear();
        cur_kp_.clear();
        track_success_.clear();
        //cv::FAST(cur_frame->Mat(), init_.kp1, fast_thresh);
        cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.01, 10);
        detector->detect(cur_frame->Mat(), ref_kp_);
        cur_kp_ = ref_kp_;
        ref_frame_ = cur_frame;
        frame_cnt_ = 1;

        return false;
    }
}

void Initializer::OpticalFlowSingleLevel(const cv::Mat& img1, const cv::Mat& img2,
    const std::vector<cv::KeyPoint>& kp1,
    std::vector<cv::KeyPoint>& kp2,
    std::vector<bool>& success, int& good_cnt, bool inverse)
{
    bool have_initial = !kp2.empty();
    success.clear();
    success.reserve(kp1.size());
    good_cnt = 0;

    for (size_t i = 0; i < kp1.size(); i++) {
        cv::KeyPoint kp = kp1[i];
        double dx = 0, dy = 0;

        if (have_initial) {
            dx = kp2[i].pt.x - kp1[i].pt.x;
            dy = kp2[i].pt.y - kp1[i].pt.y;
        }

        double cost = 0, last_cost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations_; ++iter) {
            M2d H = M2d::Zero();
            V2d b = V2d::Zero();
            cost = 0;

            // +1 and -1 because we also need to calculate gradients
            if ((int)(kp.pt.x - half_patch_size_ - std::abs(dx) - 1) < 0 || (int)(kp.pt.x + half_patch_size_ + std::abs(dx) + 1) >= img1.cols || (int)(kp.pt.y - half_patch_size_ - std::abs(dy) - 1) < 0 || (int)(kp.pt.y + half_patch_size_ + std::abs(dy) + 1) >= img1.rows) {
                succ = false;
                break;
            }

            // compute cost and jacobian
            for (int x = -half_patch_size_; x < half_patch_size_; ++x) {
                for (int y = -half_patch_size_; y < half_patch_size_; ++y) {
                    double error = 0;
                    V2d J;

                    // Inverse Jacobian
                    J = -GetImageGradient(img1, kp.pt.x + x, kp.pt.y + y);

                    error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) - GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);

                    // compute H, b and set cost;
                    H += J * J.transpose();
                    b += -J * error;
                    cost += error * error;
                }
            }

            V2d update = H.inverse() * b;

            if (std::isnan(update[0])) {
                succ = false;
                break;
            }

            if (iter > 0 && cost > last_cost) {
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            last_cost = cost;

            if (last_cost > photometric_thresh_) {
                succ = false;
            } else {
                succ = true;
            }
        }

        success.push_back(succ);
        good_cnt += succ;

        // set kp2
        if (have_initial) {
            kp2[i].pt = kp.pt + cv::Point2f((float)dx, (float)dy);
        } else {
            cv::KeyPoint tracked = kp;
            tracked.pt += cv::Point2f((float)dx, (float)dy);
            kp2.push_back(tracked);
        }
    }
}

void Initializer::OpticalFlowMultiLevel(
    Keyframe* const ref_frame,
    Keyframe* const cur_frame,
    std::vector<cv::KeyPoint>& kp1,
    std::vector<cv::KeyPoint>& kp2,
    std::vector<bool>& success, int& good_cnt, bool inverse)
{
    // Scale kp1 to the highest pyramid scale.
    for (int j = 0; j < kp1.size(); ++j) {
        kp1[j].pt *= Keyframe::scales[Keyframe::nr_pyramids - 1];
    }

    // Scale the initial guess for kp2.
    for (int j = 0; j < kp2.size(); ++j) {
        kp2[j].pt *= Keyframe::scales[Keyframe::nr_pyramids - 1];
    }

    for (int i = Keyframe::nr_pyramids - 1; i >= 0; --i) {
        OpticalFlowSingleLevel(ref_frame->GetPyramids()[i], cur_frame->GetPyramids()[i], kp1, kp2, success, good_cnt, inverse);
        assert(kp1.size() == kp2.size());

        if (i != 0) {
            for (int j = 0; j < kp1.size(); ++j) {
                kp1[j].pt /= Keyframe::pyramid_scale;
                kp2[j].pt /= Keyframe::pyramid_scale;
            }
        }
    }
}

double Initializer::CalculateDisparity2(const std::vector<cv::KeyPoint>& kp1,
    const std::vector<cv::KeyPoint>& kp2, const std::vector<bool>& success)
{
    assert(kp1.size() == kp2.size());
    assert(kp2.size() == success.size());

    if (kp1.size() == 0) {
        return 0;
    }

    int good_cnt = 0;
    double disparity_squared = 0;
    for (int i = 0; i < kp1.size(); ++i) {
        if (success[i]) {
            double dx = kp2[i].pt.x - kp1[i].pt.x;
            double dy = kp2[i].pt.y - kp1[i].pt.y;
            disparity_squared += dx * dx + dy * dy;
            ++good_cnt;
        }
    }

    if (good_cnt > 0) {
        disparity_squared /= good_cnt;
    }

    return disparity_squared;
}

void Initializer::PoseEstimation(
    const std::vector<cv::Point2f>& p1,
    const std::vector<cv::Point2f>& p2,
    M3d& R, V3d& T, std::vector<bool>& success,
    int& good_cnt)
{
    std::vector<M3d> rotations;
    std::vector<V3d> translations;

    good_cnt = 0;
    success.clear();
    success.reserve(p1.size());

    std::cout << "Tracking : " << p1.size() << "\n";

    const double thresh = reprojection_thresh_ / std::sqrt(K_(0, 0) * K_(0, 0) + K_(1, 1) * K_(1, 1));
    cv::Mat outlier_mask_essential;
    cv::Mat essential = cv::findEssentialMat(
        p1, p2, 1.0, { 0.0, 0.0 }, CV_FM_RANSAC, 0.99, thresh, outlier_mask_essential);

    if (essential.data != NULL) {
        rotations.push_back(M3d::Identity());
        translations.push_back(V3d::Zero());

        // This method does the depth check. Only users points which are not masked
        // out by
        // the outlier mask.
        cv::Mat R_ess, T_ess;
        cv::recoverPose(essential, p1, p2, R_ess, T_ess, 1.0, {}, outlier_mask_essential);
        cv::cv2eigen(R_ess, rotations[0]);
        cv::cv2eigen(T_ess, translations[0]);
    }

    for (int i = 0; i < p1.size(); ++i) {
        if (outlier_mask_essential.at<bool>(i, 1)) {
            ++good_cnt;
            success.push_back(true);
        } else {
            success.push_back(false);
        }
    }

    R = rotations[0];
    T = translations[0];

#if 0
  cv::Mat outlier_mask_homography;
    cv::Mat homography = cv::findHomography(p1_, p2_, CV_RANSAC, thresh, outlier_mask_homography, 2000, 0.99);

    if (homography.data != NULL) {
        std::vector<cv::Mat> rotations_homo, translations_homo, normals;
        cv::decomposeHomographyMat(homography, cv::Mat::eye(3, 3, CV_64F), rotations_homo, translations_homo, normals);

        for (int i = 0; i < rotations_homo.size(); ++i) {
            rotations.push_back(M3d::Identity());
            translations.push_back(V3d::Zero());
            cv::cv2eigen(rotations_homo[i], rotations[rotations.size() - 1]);
            cv::cv2eigen(translations_homo[i], translations[translations.size() - 1]);
        }
    }
#endif
    //SelectMotion(p1, p2, rotations, translations, R, T, inliers, nr_inliers, points3d);
}

// We want to find the 4d-coorindates of point P = (P1, P2, P3, 1)^T.
// lambda1 * x1 = Pi1 * P
// lambda2 * x2 = Pi2 * P
//
// I:   lambda1 * x11 = Pi11 * P
// II:  lambda1 * x12 = Pi12 * P
// III: lambda1       = Pi13 * P
//
// III in I and II
// I':   Pi13 * P * x11 = Pi11 * P
// II':  Pi13 * P * x12 = Pi12 * P
//
// I'':   (x11 * Pi13  - Pi11) * P = 0
// II'':  (x12 * Pi13  - Pi12) * p = 0
//
// We get another set of two equations from lambda2 * x2 = Pi2 * P.
// Finally we can construct a 4x4 matrix A such that A * P = 0
// A = [[x11 * Pi13  - Pi11];
//      [x12 * Pi13  - Pi12];
//      [x21 * Pi23  - Pi21];
//      [x22 * Pi13  - Pi22]];
void Initializer::Triangulate(const M34d& Pi1, const M34d& Pi2, const cv::Point2f& x1,
    const cv::Point2f& x2, V3d& P)
{
    M4d A = M4d::Zero();
    A.row(0) = x1.x * Pi1.row(2) - Pi1.row(0);
    A.row(1) = x1.y * Pi1.row(2) - Pi1.row(1);
    A.row(2) = x2.x * Pi2.row(2) - Pi2.row(0);
    A.row(3) = x2.y * Pi2.row(2) - Pi2.row(1);

    Eigen::JacobiSVD<M4d> svd(A, Eigen::ComputeFullV);
    M4d V = svd.matrixV();

    // The solution is the last column V. This gives us a homogeneous
    // point, so we need to normalize.
    P = V.col(3).block<3, 1>(0, 0) / V(3, 3);
}

void Initializer::Reconstruct(
    const std::vector<cv::Point2f>& p1,
    const std::vector<cv::Point2f>& p2,
    const M3d& R, const V3d& T, std::vector<bool>& success,
    int& good_cnt, std::vector<V3d>& points3d)
{
    assert(good_cnt > 0);
    assert(p1.size() == p2.size());
    assert(p2.size() == success.size());

    const M34d Pi1 = MakePI0();
    const M34d Pi2 = MakePI0() * MakeSE3(R, T);

#if 0
    V3d O1 = V3d::Zero();
    V3d O2 = -R*T;
#endif

    points3d.clear();

    for (int i = 0; i < p1.size(); ++i) {
        if (success[i]) {
            V3d P1;
            Triangulate(Pi1, Pi2, p1[i], p2[i], P1);

#if 0
            // parallax
            V3d n1 = P1 - O1;
            V3d n2 = P1 - O2;
            double d1 = n1.norm();
            double d2 = n2.norm();

            double parallax = (n1.transpose () * n2);
            parallax /=  (d1 * d2);
            parallax = acos(parallax)*180/CV_PI;
            if (parallax > parallax_thresh)
            {
                inliers[i] = false;
                nr_inliers--;
                continue;
            }
#endif
            // projection error
            V3d P1_proj = P1 / P1.z();
            double dx = (P1_proj.x() - p1[i].x) * K_(0, 0);
            double dy = (P1_proj.y() - p1[i].y) * K_(1, 1);
            double projection_error1 = std::sqrt(dx * dx + dy * dy);

            if (projection_error1 > reprojection_thresh_) {
                success[i] = false;
                --good_cnt;
                continue;
            }

            V3d P2 = R * P1 + T;
            V3d P2_proj = P2 / P2.z();
            dx = (P2_proj.x() - p2[i].x) * K_(0, 0);
            dy = (P2_proj.y() - p2[i].y) * K_(1, 1);
            double projection_error2 = std::sqrt(dx * dx + dy * dy);

            if (projection_error2 > reprojection_thresh_) {
                success[i] = false;
                --good_cnt;
                continue;
            }

            points3d.push_back(P1);
        }
    }

    assert(good_cnt == points3d.size());
}

void Initializer::NormalizeDepth(V3d& T, std::vector<V3d>& points3d)
{
    assert(points3d.size() > 0);

    double mean_depth = 0;
    for (const auto& p : points3d) {
        mean_depth += p.z();
    }

    mean_depth /= points3d.size();

    for (auto& p : points3d) {
        p /= mean_depth;
    }

    T /= mean_depth;
}

void Initializer::Visualize(
    const cv::Mat& display,
    const std::vector<cv::KeyPoint>& kp1,
    const std::vector<cv::KeyPoint>& kp2,
    const std::vector<bool>& success,
    const int& good_cnt)
{
    for (int i = 0; i < kp1.size(); i++) {
        if (success[i]) {
            cv::line(display, kp1[i].pt, kp2[i].pt, cv::Scalar(0, 255, 0));
        }
    }
}
