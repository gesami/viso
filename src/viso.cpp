#include "viso.h"
#include "common.h"
#include "timer.h"

#include <Eigen/Geometry>
#include <depth_filter.h>
#include <map>
#include <opencv2/core/eigen.hpp>

using namespace std;

void Viso::OnNewFrame(Keyframe::Ptr cur_frame)
{
    static depth_filter* filter = nullptr;

    // set intrinstic parameters
    cur_frame->SetK(K);

    switch (state_) {
    case kInitialization: {
        // if not yet initialized
        cv::Mat img;
        cv::cvtColor(cur_frame->Mat(), img, CV_GRAY2BGR);
        std::string first;
        bool initialized = initializer.InitializeMap(cur_frame, &map_, img, first);
        // Visualization
        if (vis) {
            cv::imshow("Tracked", img);
            cv::waitKey(10);
        }
        // if initialization successful
        if (initialized) {
            state_ = kRunning;
            opt.BA(&map_, true, 1, K);  // do the batch optinmiztion
            do_ba_ = false;
            // record pose
            k2f = Sophus::SE3d(M3d::Identity(), V3d::Zero());
            f2f = Sophus::SE3d(cur_frame->GetR(), cur_frame->GetT());
            ///for trajectory record
            frame_time.push_back(first);
            ref_key.push_back(0);
            ref_pose.push_back(k2f);
            frame_time.push_back(cur_frame->GetTime());
            ref_key.push_back(1);
            ref_pose.push_back(k2f);

            if (add_ba) {
                ba_thread_ = std::thread([&]() {
                    while (running.load()) {
                        if (do_ba_.load()) {
                            do_ba_ = false;
                            global_ba();
                            //opt.BA(&map_,true, 2, K);
                            usleep(15000);
                        }
                    }
                });
            } else if (add_lba) {
                ba_thread_ = std::thread([&]() {
                    while (running.load()) {
                        if (do_ba_.load()) {
                            do_ba_ = false;
                            local_ba();
                            //opt.BA_LOCAL(&map_, K);
                            usleep(15000);
                        }
                    }
                });
            }

            lkf = cur_frame->GetPose();

            if (better_tracker_.use) {
                better_tracker_.ref_frame = cur_frame;
                better_tracker_.success = std::vector<bool>(cur_frame->Keypoints().size(), true);
                better_tracker_.good_cnt = better_tracker_.success.size();
                better_tracker_.last_frame = cur_frame;
            }
        }
    } break;

    // if already initialized
    case kRunning: {
        std::lock_guard<std::mutex> lock(update_map_);
        // modified tracking
        if (better_tracker_.use) {

            // Use last frame's keypoints and pose as starting values.
            cur_frame->CopyKeypointsFrom(better_tracker_.last_frame);
            cur_frame->SetPose(better_tracker_.last_frame->GetPose());

            if (Track(better_tracker_.last_frame, cur_frame, better_tracker_.success, better_tracker_.good_cnt) == false) {
                state_ = kLostTrack;
                break;
            }

            k2f = cur_frame->GetPose() * lkf.inverse();
            //f2f = X*lf.inverse();
            map_.SetCurrent(cur_frame);

            frame_time.push_back(cur_frame->GetTime());
            ref_key.push_back(map_.GetKeyid());
            ref_pose.push_back(k2f);

            // show image
            cv::Mat display;
            cv::cvtColor(cur_frame->Mat(), display, CV_GRAY2BGR);

            const auto& kps = cur_frame->Keypoints();
            for (int i = 0; i < kps.size(); ++i) {
                auto color = cv::Scalar(0, 255, 0);
                if (better_tracker_.success[i]) {
                    cv::rectangle(display, cv::Point2f(kps[i].pt.x - 4, kps[i].pt.y - 4),
                        cv::Point2f(kps[i].pt.x + 4, kps[i].pt.y + 4),
                        color);
                }
            }

            cv::imshow("Better tracker", display);
            cv::waitKey(0);

            if (IsKeyframe(cur_frame, cur_frame->Keypoints().size())) {

                map_.AddKeyframe(cur_frame);

                { // Delete unsuccessful keypoints
                    auto kp_iter = cur_frame->Keypoints().begin();
                    auto mp_iter = cur_frame->MapPoints().begin();

                    for (int i = 0; i < better_tracker_.success.size(); ++i) {
                        if (!better_tracker_.success[i]) {
                            kp_iter = cur_frame->Keypoints().erase(kp_iter);
                            mp_iter = cur_frame->MapPoints().erase(mp_iter);
                        } else {
                            ++kp_iter;
                            ++mp_iter;
                        }
                    }
                }

                { // Add observations
                    // At this point, all observations are valid.
                    auto& kps = cur_frame->Keypoints();
                    auto& mps = cur_frame->MapPoints();

                    int i = 0;
                    for (auto& mp : mps) {
                        mp->AddObservation(cur_frame, i);
                        ++i;
                    }
                }

                better_tracker_.ref_frame = cur_frame;
                better_tracker_.success = std::vector<bool>(cur_frame->Keypoints().size(), true);
                better_tracker_.good_cnt = better_tracker_.success.size();

                std::vector<cv::KeyPoint> kp;
                featureDetector->detect(cur_frame->Mat(), kp);
                vector<V3d> wp = map_.GetPoints3d();
                cur_frame->SetOccupied(wp);
                cur_frame->AddNewFeatures(kp);

                if (df_on) {
                    if (filter != nullptr) {
                        delete filter;
                    }

                    filter = new depth_filter(cur_frame);
                }

                do_ba_ = true;
                k2f = Sophus::SE3d(M3d::Identity(), V3d::Zero());
                lkf = cur_frame->GetPose();

            } else if (df_on && filter != nullptr && GetMotion(cur_frame) >= 0.05) {
                filter->Update(cur_frame);
                filter->UpdateMap(&map_, cur_frame);
            }

            better_tracker_.last_frame = cur_frame;
        } else {
            // original tracking method
            Sophus::SE3d X = Sophus::SE3d(last_frame->GetR(), last_frame->GetT()); //Keyframe pose
            
            // pose estimation using direct method
            DirectPoseEstimationMultiLayer(cur_frame, X);
            cur_frame->SetR(X.rotationMatrix());
            cur_frame->SetT(X.translation());

            // LKAlignment
            std::vector<V2d> kp_before, kp_after;
            std::vector<int> tracked_points;
            std::vector<AlignmentPair> alignment_pairs;
            LKAlignment(cur_frame, kp_before, kp_after, tracked_points, alignment_pairs);

            if (add_mba) {
                if (tracked_points.size() > 9) {
                    MotionOnlyBA(X, alignment_pairs);
                    cur_frame->SetR(X.rotationMatrix());
                    cur_frame->SetT(X.translation());
                }
            }

            k2f = cur_frame->GetPose() * lkf.inverse();
            //f2f = X*lf.inverse();
            map_.SetCurrent(cur_frame);

            frame_time.push_back(cur_frame->GetTime());
            ref_key.push_back(map_.GetKeyid());
            ref_pose.push_back(k2f);

            // show image
            if (vis) {
                // show image
                cv::Mat display;
                cv::cvtColor(cur_frame->Mat(), display, CV_GRAY2BGR);

                for (int i = 0; i < kp_after.size(); ++i) {
                    cv::rectangle(display, cv::Point2f(kp_after[i].x() - 4, kp_after[i].y() - 4), cv::Point2f(kp_after[i].x() + 4, kp_after[i].y() + 4),
                        cv::Scalar(0, 255, 0));
                }

                cv::imshow("Tracked", display);
                cv::waitKey(10);
            }

            // determine if current frame should be a keyframe
            if (IsKeyframe(cur_frame, tracked_points.size())) {

                map_.AddKeyframe(cur_frame);

                assert(cur_frame->Keypoints().size() == 0);
                vector<MapPoint::Ptr> map_points = map_.GetPoints();
                for (int i = 0; i < tracked_points.size(); ++i) {
                    cv::KeyPoint kp;
                    kp.pt.x = kp_after[i].x();
                    kp.pt.y = kp_after[i].y();
                    int kp_idx = cur_frame->AddKeypoint(kp, map_points[tracked_points[i]]); // add the keypoints one by one
                    map_points[tracked_points[i]]->AddObservation(cur_frame, kp_idx); // saved in map: world point coordinate, corresponding frame, position of point kp array
                }

                std::vector<cv::KeyPoint> kp;
                featureDetector->detect(cur_frame->Mat(), kp);
                vector<V3d> wp = map_.GetPoints3d();
                cur_frame->SetOccupied(wp);
                cur_frame->AddNewFeatures(kp);

                if (df_on) {
                    if (filter != nullptr) {
                        delete filter;
                    }

                    filter = new depth_filter(cur_frame);
                }

                do_ba_ = true;
                k2f = Sophus::SE3d(M3d::Identity(), V3d::Zero());
                lkf = cur_frame->GetPose();
            } else if (df_on && filter != nullptr && GetMotion(cur_frame) >= 0.05) {
                filter->Update(cur_frame);
                filter->UpdateMap(&map_, cur_frame);
            }
        }

    } break;

    case kLostTrack: {
        std::cout << "Lost track\n";
    } break;

    default:
        break;
    }

    lf = Sophus::SE3d(cur_frame->GetR(), cur_frame->GetT());
    last_frame = cur_frame;
}

#if 0
void Viso::SelectMotion(const std::vector<V3d>& p1,
    const std::vector<V3d>& p2,
    const std::vector<M3d>& rotations,
    const std::vector<V3d>& translations,
    M3d& R_out,
    V3d& T_out,
    std::vector<bool>& inliers,
    int& nr_inliers,
    std::vector<V3d>& points3d)
{
    assert(rotations.size() == translations.size());

    int best_nr_inliers = 0;
    int best_motion = -1;
    std::vector<bool> best_inliers;
    std::vector<V3d> best_points;

    for (int m = 0; m < rotations.size(); ++m) {
        M3d R = rotations[m];
        V3d T = translations[m];
        M34d Pi1 = MakePI0();
        M34d Pi2 = MakePI0() * MakeSE3(R, T);
        V3d O1 = V3d::Zero();
        V3d O2 = -R * T;

        inliers.clear();
        inliers.reserve(p1.size());
        points3d.clear();

        int j = 0;
        for (int i = 0; i < p1.size(); ++i) {
            inliers.push_back(false);

            V3d P1;
            Triangulate(Pi1, Pi2, p1[i], p2[i], P1);

            // depth test
            if (P1.z() < 0) {
                continue;
            }

            // parallax
            V3d n1 = P1 - O1;
            V3d n2 = P1 - O2;
            double d1 = n1.norm();
            double d2 = n2.norm();

            double parallax = (n1.transpose() * n2);
            parallax /= (d1 * d2);
            parallax = acos(parallax) * 180 / CV_PI;
            if (parallax > parallax_thresh) {
                continue;
            }

            // projection error
            V3d P1_proj = P1 / P1.z();
            double dx = (P1_proj.x() - p1[i].x()) * K(0, 0);
            double dy = (P1_proj.y() - p1[i].y()) * K(1, 1);
            double projection_error1 = std::sqrt(dx * dx + dy * dy);

            if (projection_error1 > projection_error_thresh) {
                continue;
            }

            V3d P2 = R * P1 + T;

            // depth test
            if (P2.z() < 0) {
                continue;
            }

            // projection error
            V3d P2_proj = P2 / P2.z();
            dx = (P2_proj.x() - p2[i].x()) * K(0, 0);
            dy = (P2_proj.y() - p2[i].y()) * K(1, 1);
            double projection_error2 = std::sqrt(dx * dx + dy * dy);

            if (projection_error2 > projection_error_thresh) {
                continue;
            }

            inliers[i] = true;
            points3d.push_back(P1);
        }

        if (points3d.size() > best_nr_inliers) {
            best_nr_inliers = points3d.size();
            best_inliers = inliers;
            best_motion = m;
            best_points = points3d;
        }
    }

    nr_inliers = best_nr_inliers;
    points3d = best_points;
    inliers = best_inliers;

    if (best_motion != -1) {
        R_out = rotations[best_motion];
        T_out = translations[best_motion];
    }

    // Depth normalization
    double mean_depth = 0;
    for (const auto& p : points3d) {
        mean_depth += p.z();
    }

    if (mean_depth != 0) {
        mean_depth /= points3d.size();

        for (auto& p : points3d) {
            p /= mean_depth;
        }

        T_out /= mean_depth;
    }
}
#endif

M26d dPixeldXi(const M3d& K, const M3d& R, const V3d& T, const V3d& P,
    const double& scale)
{
    V3d Pc = R * P + T;
    double x = Pc.x();
    double y = Pc.y();
    double z = Pc.z();
    double fx = K(0, 0) * scale;
    double fy = K(1, 1) * scale;
    double zz = z * z;
    double xy = x * y;

    M26d result;
    result << fx / z, 0, -fx * x / zz, -fx * xy / zz, fx + fx * x * x / zz,
        -fx * y / z, 0, fy / z, -fy * y / zz, -fy - fy * y * y / zz, fy * xy / zz,
        fy * x / z;

    return result;
}

// TODO: Move this to a separate class.
void Viso::DirectPoseEstimationSingleLayer(int level,
    Keyframe::Ptr current_frame,
    Sophus::SE3d& T21)
{
    const double scale = Keyframe::scales[level];
    const double delta_thresh = 0.005;

    // parameters
    int iterations = 100;

    double cost = 0, lastCost = 0;
    int nGood = 0; // good projections

    Sophus::SE3d best_T21 = T21;

    for (int iter = 0; iter < iterations; iter++) {
        nGood = 0;

        // Define Hessian and bias
        M6d H = M6d::Zero(); // 6x6 Hessian
        V6d b = V6d::Zero(); // 6x1 bias

        current_frame->SetR(T21.rotationMatrix());
        current_frame->SetT(T21.translation());

        V3d look_dir = T21.inverse() * V3d{ 0, 0, 1 };

        for (size_t i = 0; i < map_.GetPoints().size(); i++) {

            V3d P1 = map_.GetPoints()[i]->GetWorldPos();

            Keyframe::Ptr frame = last_frame;
            V2d uv_ref = frame->Project(P1, level);
            double u_ref = uv_ref.x();
            double v_ref = uv_ref.y();

            V2d uv_cur = current_frame->Project(P1, level);
            double u_cur = uv_cur.x();
            double v_cur = uv_cur.y();

            bool hasNaN = uv_cur.array().hasNaN() || uv_ref.array().hasNaN();
            assert(!hasNaN);

            bool good = frame->IsInside(u_ref - lk_half_patch_size,
                            v_ref - lk_half_patch_size, level)
                && frame->IsInside(u_ref + lk_half_patch_size,
                       v_ref + lk_half_patch_size, level)
                && current_frame->IsInside(u_cur - lk_half_patch_size,
                       v_cur - lk_half_patch_size, level)
                && current_frame->IsInside(u_cur + lk_half_patch_size,
                       v_cur + lk_half_patch_size, level);

            if (!good) {
                continue;
            }

            double cos_angle = look_dir.transpose() * map_.GetPoints()[i]->GetDirection();

            /*if (cos_angle < 0.86602540378) { // 30 degree
                std::cout << "DirectPoseEstimationSingleLayer: cos_angle " << cos_angle << "\n";
                continue;
            }*/

            nGood++;

            M26d J_pixel_xi = dPixeldXi(K, T21.rotationMatrix(), T21.translation(),
                P1, scale); // pixel to \xi in Lie algebra

            for (int x = -lk_half_patch_size; x < lk_half_patch_size; x++) {
                for (int y = -lk_half_patch_size; y < lk_half_patch_size; y++) {
                    double error = frame->GetPixelValue(u_ref + x, v_ref + y, level) - current_frame->GetPixelValue(u_cur + x, v_cur + y, level);
                    V2d J_img_pixel = current_frame->GetGradient(u_cur + x, v_cur + y, level);
                    V6d J = -J_img_pixel.transpose() * J_pixel_xi;
                    H += J * J.transpose();
                    b += -error * J;
                    cost += error * error;
                }
            }
        }

        //if(nGood < 6) return;

        // solve update and put it into estimation
        V6d update = H.inverse() * b;

        T21 = Sophus::SE3d::exp(update) * T21;

        cost /= nGood;

        if (std::isnan(update[0])) {
            T21 = best_T21;
            break;
        }

        if (iter > 0 && cost > lastCost) {
            T21 = best_T21;
            break;
        }

        if ((1 - cost / (double)lastCost) < delta_thresh) {
            break;
        }

        best_T21 = T21;
        lastCost = cost;
    }
}

void Viso::DirectPoseEstimationMultiLayer(Keyframe::Ptr current_frame,
    Sophus::SE3d& T21)
{
    for (int level = 3; level >= 0; level--) {
        DirectPoseEstimationSingleLayer(level, current_frame, T21);
    }
}

void Viso::LKAlignment(Keyframe::Ptr current_frame, std::vector<V2d>& kp_before, std::vector<V2d>& kp_after, std::vector<int>& tracked_points,
    std::vector<AlignmentPair>& alignment_pairs)
{
    const double max_angle = 180.0; // 180 means basically no restriction on the angle (for now)
    assert(alignment_pairs.size() == 0);

    for (size_t i = 0; i < map_.GetPoints().size(); ++i) {

        MapPoint::Ptr map_point = map_.GetPoints()[i];
        V3d Pw = map_point->GetWorldPos();

        if (!current_frame->IsInside(Pw, /*level=*/0)) {
            continue;
        }

        V3d look_dir = current_frame->GetPose().inverse() * V3d{ 0, 0, 1 };
        double cos_angle = look_dir.transpose() * map_.GetPoints()[i]->GetDirection();

        /*if (cos_angle < 0.86602540378) { // 30 degree
            std::cout << "LKAlignment: cos_angle " << cos_angle << "\n";
            continue;
        }*/

        // Find frame with best viewing angle.
        double best_angle = 180.0;
        int best_frame_idx = -1;
        Keyframe::Ptr best_keyframe;
        V2d best_uv_ref;

        const std::vector<std::pair<Keyframe::Ptr, int> >& observations = map_point->GetObservations();

        for (int j = 0; j < observations.size(); ++j) {
            Keyframe::Ptr frame = observations[j].first;

            double angle = std::abs(frame->ViewingAngle(Pw) / CV_PI * 180);
            if (angle > max_angle || angle > best_angle) {
                continue;
            }

            best_angle = angle;
            best_frame_idx = j;
            best_uv_ref = V2d{ frame->Keypoints()[observations[j].second].pt.x, frame->Keypoints()[observations[j].second].pt.y };
            best_keyframe = frame;
        }

        if (best_frame_idx == -1) {
            continue;
        }

        AlignmentPair pair;
        pair.ref_frame = best_keyframe;
        pair.cur_frame = current_frame;
        pair.uv_ref = best_uv_ref;
        pair.uv_cur = current_frame->Project(Pw, /*level=*/0);
        pair.point3d = map_point->GetWorldPos();

        alignment_pairs.push_back(pair);
        tracked_points.push_back(i);
    }

    for (int i = 0; i < alignment_pairs.size(); ++i) {
        kp_before.push_back(alignment_pairs[i].uv_cur);
    }

    std::vector<bool> success(alignment_pairs.size(), false);
    kp_after = std::vector<V2d>(kp_before.size(), V2d{ 0.0, 0.0 });

    for (int level = Keyframe::nr_pyramids - 1; level >= 0; --level) {
        LKAlignmentSingle(alignment_pairs, success, kp_after, level);
    }

    // reduce tracking outliers
    std::vector<double> d2;
    for (int i = 0; i < kp_before.size(); ++i) {
        if (success[i]) {
            double dx = kp_after[i].x() - kp_before[i].x();
            double dy = kp_after[i].y() - kp_before[i].y();
            d2.push_back(dx * dx + dy * dy);
        }
    }

    double median_d2 = CalculateMedian(d2);

    int j = 0;
    for (int i = 0; i < kp_before.size(); ++i) {
        if (success[i]) {
            if (d2[j] > median_d2 * lk_d2_factor) {
                success[i] = false;
            }
            ++j;
        }
    }

    assert(success.size() == kp_before.size());

    int i = 0;
    auto iter2 = kp_after.begin();
    auto iter3 = tracked_points.begin();
    auto iter4 = alignment_pairs.begin();
    for (auto iter1 = kp_before.begin(); iter1 != kp_before.end(); ++i) {
        if (!success[i]) {
            iter1 = kp_before.erase(iter1);
            iter2 = kp_after.erase(iter2);
            iter3 = tracked_points.erase(iter3);
            iter4 = alignment_pairs.erase(iter4);
        } else {
            ++iter1;
            ++iter2;
            ++iter3;
            ++iter4;
        }
    }
}

// kp
void Viso::LKAlignmentSingle(std::vector<AlignmentPair>& pairs, std::vector<bool>& success, std::vector<V2d>& kp, int level)
{
    // parameters
    const bool inverse = false;
    const int iterations = 100;

    assert(pairs.size() == success.size());
    assert(success.size() == kp.size());

    for (size_t i = 0; i < pairs.size(); i++) {
        AlignmentPair& pair = pairs[i];

        double dx = 0, dy = 0; // dx,dy need to be estimated
        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; iter++) {
            M2d A = M2d::Identity();
            if (affine_warping) {
                A = GetAffineWarpingMatrix(pair.ref_frame, pair.cur_frame, pair.point3d, pair.uv_ref);
            }

            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (
                !pair.ref_frame->IsInside(pair.uv_ref.x() * Keyframe::scales[level] - std::abs(dx) - lk_half_patch_size - 1, pair.uv_ref.y() * Keyframe::scales[level] - std::abs(dy) - lk_half_patch_size - 1, level) || !pair.ref_frame->IsInside(pair.uv_ref.x() * Keyframe::scales[level] + std::abs(dx) + lk_half_patch_size + 1, pair.uv_ref.y() * Keyframe::scales[level] + std::abs(dy) + lk_half_patch_size + 1, level)) {
                succ = false;
                break;
            }

            {
                V2d test_uv_1 = A * V2d{ -lk_half_patch_size, -lk_half_patch_size } + V2d{ pair.uv_cur.x() * Keyframe::scales[level] - std::abs(dx) - 1, pair.uv_cur.y() * Keyframe::scales[level] - std::abs(dy) - 1 };
                V2d test_uv_2 = A * V2d{ lk_half_patch_size, lk_half_patch_size } + V2d{ pair.uv_cur.x() * Keyframe::scales[level] + std::abs(dx) + 1, pair.uv_cur.y() * Keyframe::scales[level] + std::abs(dy) + 1 };

                if (!pair.cur_frame->IsInside(test_uv_1.x(), test_uv_1.y()) || !pair.cur_frame->IsInside(test_uv_2.x(), test_uv_2.y())) {
                    succ = false;
                    break;
                }
            }

            double error = 0;
            // compute cost and jacobian
            for (int x = -lk_half_patch_size; x < lk_half_patch_size; x++) {
                for (int y = -lk_half_patch_size; y < lk_half_patch_size; y++) {
                    V2d uv_cur_warped = A * V2d{ x, y } + V2d{ pair.uv_cur.x() * Keyframe::scales[level] + dx, pair.uv_cur.y() * Keyframe::scales[level] + dy };
                    V2d J;
                    if (!inverse) {
                        J = -pair.cur_frame->GetGradient(pair.uv_cur.x() * Keyframe::scales[level] + x + dx, pair.uv_cur.y() * Keyframe::scales[level] + y + dy, level);
                    } else {
                        J = -pair.ref_frame->GetGradient(pair.uv_ref.x() * Keyframe::scales[level] + x, pair.uv_ref.y() * Keyframe::scales[level] + y, level);
                    }
                    error = pair.ref_frame->GetPixelValue(pair.uv_ref.x() * Keyframe::scales[level] + x, pair.uv_ref.y() * Keyframe::scales[level] + y, level) - pair.cur_frame->GetPixelValue(pair.uv_cur.x() * Keyframe::scales[level] + x + dx, pair.uv_cur.y() * Keyframe::scales[level] + y + dy, level);

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

            if (iter > 0 && cost > lastCost) {
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;

            if (lastCost > lk_photometric_thresh) {
                succ = false;
            } else {
                succ = true;
            }
        }

        success[i] = success[i] || succ;

        // Only update if succeeded. Otherwise we want to keep the old value.
        if (succ) {
            pair.uv_cur += V2d{ dx / Keyframe::scales[level], dy / Keyframe::scales[level] };
        }
    }

    for (int i = 0; i < pairs.size(); ++i) {
        kp[i] = pairs[i].uv_cur;
    }
}

bool Viso::Track(Keyframe::Ptr ref_frame, Keyframe::Ptr cur_frame, std::vector<bool>& success, int& good_cnt)
{
    const auto& ref_kps = ref_frame->Keypoints();
    const auto& ref_mps = ref_frame->MapPoints();

    auto& cur_kps = cur_frame->Keypoints();

    assert(ref_kps.size() == success.size());
    assert(ref_kps.size() == ref_mps.size());
    assert(ref_kps.size() == cur_kps.size());
    assert(cur_kps.size() == success.size());

    std::vector<AlignmentPair> alignment_pairs;
    alignment_pairs.reserve(good_cnt);

    for (int i = 0; i < ref_kps.size(); ++i) {
        bool add_pair = false;
        AlignmentPair pair;

        // Keep track...
        if (success[i]) {
            pair.uv_cur = V2d{ cur_kps[i].pt.x, cur_kps[i].pt.y };
            add_pair = true;
            // or try to track lost map points.
        }

        //        else {
        //            V2d uv = cur_frame->Project(ref_mps[i]->GetWorldPos(), /*level =*/0);
        //            if (cur_frame->IsInside(uv.x(), uv.y(), /*level =*/0)) {
        //                pair.uv_cur = uv;
        //                success[i] = true;
        //                ++good_cnt;
        //                add_pair = true;
        //            }
        //        }

        if (add_pair) {
            pair.ref_frame = ref_frame;
            pair.cur_frame = cur_frame;
            pair.uv_ref = V2d{ ref_kps[i].pt.x, ref_kps[i].pt.y };
            pair.point3d = ref_mps[i]->GetWorldPos();
            alignment_pairs.push_back(pair);
        }
    }

    assert(alignment_pairs.size() == good_cnt);
    std::vector<bool> success_before = success;

    for (int level = Keyframe::nr_pyramids - 1; level >= 0; --level) {
        std::vector<bool> success_tmp = success;
        TrackSingle(alignment_pairs, success_tmp, level);

        if (level == 0) {
            success = success_tmp;
        }
    }

    assert(success.size() == success_before.size());

    int j = 0;
    // update keypoints
    for (int i = 0; i < success.size(); ++i) {
        if (success_before[i]) {
            cur_kps[i].pt = cv::Point2f{ (float)alignment_pairs[j].uv_cur.x(), (float)alignment_pairs[j].uv_cur.y() };
            ++j;
        }
    }

    assert(j == good_cnt);

    // delete keypoints

    int good_cnt_before = good_cnt;

    {
        auto ap_iter = alignment_pairs.begin();
        for (int i = 0; i < success.size(); ++i) {
            if (success_before[i]) {
                if (success_before[i] != success[i]) {
                    ap_iter = alignment_pairs.erase(ap_iter);
                    --good_cnt;
                } else {
                    ++ap_iter;
                }
            }
        }

        assert(ap_iter == alignment_pairs.end());
    }

    assert(good_cnt == alignment_pairs.size());

#if 1
    if (good_cnt < 9) {
        return false;
    }

    Sophus::SE3d pose
        = cur_frame->GetPose();
    MotionOnlyBA(pose, alignment_pairs);
    cur_frame->SetPose(pose);
#endif

#if 0
    std::cout << "good_cnt before pnp: " << good_cnt << "\n";
    {
        cv::Mat camera_matrix = (cv::Mat_<float>(3, 3) << K(0, 0), 0.0, K(0, 2), 0.0, K(1, 1), K(1, 2), 0.0, 0.0, 1.0);

        cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion
        std::vector<cv::Point3f> p3d;
        std::vector<cv::Point2f> p2d;

        {
            int j = 0;
            for (int i = 0; i < success.size(); ++i) {
                if (success[i]) {
                    p3d.push_back({ (float)alignment_pairs[j].point3d[0], (float)alignment_pairs[j].point3d[1], (float)alignment_pairs[j].point3d[2] });
                    p2d.push_back({ (float)alignment_pairs[j].uv_cur[0], (float)alignment_pairs[j].uv_cur[1] });
                    ++j;
                }
            }
        }

        if (p3d.size() < 4) {
            std::cout << "Not enough points\n";
            return false;
        }

        Eigen::AngleAxisd angle_axis(cur_frame->GetR());
        V3d rotation_vector_eig = angle_axis.axis() * angle_axis.angle();
        cv::Mat rotation_vector;
        cv::eigen2cv(rotation_vector_eig, rotation_vector);

        cv::Mat translation_vector;
        cv::eigen2cv(cur_frame->GetT(), translation_vector);

        cv::Mat inliers;
        if (
            solvePnPRansac(p3d, p2d, camera_matrix, dist_coeffs, rotation_vector, translation_vector,
                /* use extrinsic guess */ true, /* iterations */ 100, /* reproj. error */ 1.0, /*confidence*/ 0.99, inliers)
            == false) {
            std::cout << "solvePnPRansac failed\n";
            return false;
        };

        {
            auto ap_iter = alignment_pairs.begin();
            int j = 0;
            for (int i = 0; i < success.size(); ++i) {
                if (success[i]) {
                    if (!inliers.at<bool>(j)) {
                        success[i] = false;
                        ap_iter = alignment_pairs.erase(ap_iter);
                        --good_cnt;
                    } else {
                        ++ap_iter;
                    }
                    ++j;
                }
            }
        }

        V3d translation_vector_eigen = V3d::Zero();
        M3d R_eig = M3d::Zero();
        cv::Mat R;
        cv::Rodrigues(rotation_vector, R);

        cv::cv2eigen(R, R_eig);
        cv::cv2eigen(translation_vector, translation_vector_eigen);

        cur_frame->SetT(translation_vector_eigen);
        cur_frame->SetR(R_eig);
    }
#endif

    std::cout << "good_cnt: " << good_cnt << "\n";

    return true;
}

void Viso::TrackSingle(std::vector<AlignmentPair>& pairs, std::vector<bool>& success, int level)
{
    const int iterations = 100;
    const bool inverse = true;

    int j = 0;
    for (size_t i = 0; i < success.size(); ++i) {

        if (!success[i]) {
            continue;
        }

        AlignmentPair& pair = pairs[j];

        double dx = 0, dy = 0; // dx,dy need to be estimated
        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; ++iter) {

            M2d A = M2d::Identity();
            if (affine_warping) {
                A = GetAffineWarpingMatrix(pair.ref_frame, pair.cur_frame, pair.point3d, pair.uv_ref);
            }

            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (
                !pair.ref_frame->IsInside(pair.uv_ref.x() * Keyframe::scales[level] - std::abs(dx) - lk_half_patch_size - 1, pair.uv_ref.y() * Keyframe::scales[level] - std::abs(dy) - lk_half_patch_size - 1, level) || !pair.ref_frame->IsInside(pair.uv_ref.x() * Keyframe::scales[level] + std::abs(dx) + lk_half_patch_size + 1, pair.uv_ref.y() * Keyframe::scales[level] + std::abs(dy) + lk_half_patch_size + 1, level)) {
                succ = false;
                break;
            }

            {
                V2d test_uv_1 = A * V2d{ -lk_half_patch_size, -lk_half_patch_size } + V2d{ pair.uv_cur.x() * Keyframe::scales[level] - std::abs(dx) - 1, pair.uv_cur.y() * Keyframe::scales[level] - std::abs(dy) - 1 };
                V2d test_uv_2 = A * V2d{ lk_half_patch_size, lk_half_patch_size } + V2d{ pair.uv_cur.x() * Keyframe::scales[level] + std::abs(dx) + 1, pair.uv_cur.y() * Keyframe::scales[level] + std::abs(dy) + 1 };

                if (!pair.cur_frame->IsInside(test_uv_1.x(), test_uv_1.y()) || !pair.cur_frame->IsInside(test_uv_2.x(), test_uv_2.y())) {
                    succ = false;
                    break;
                }
            }

            double error = 0;
            // compute cost and jacobian
            for (int x = -lk_half_patch_size; x < lk_half_patch_size; x++) {
                for (int y = -lk_half_patch_size; y < lk_half_patch_size; y++) {
                    V2d uv_cur_warped = A * V2d{ x, y } + V2d{ pair.uv_cur.x() * Keyframe::scales[level] + dx, pair.uv_cur.y() * Keyframe::scales[level] + dy };
                    V2d J;
                    if (!inverse) {
                        J = -pair.cur_frame->GetGradient(pair.uv_cur.x() * Keyframe::scales[level] + x + dx, pair.uv_cur.y() * Keyframe::scales[level] + y + dy, level);
                    } else {
                        J = -pair.ref_frame->GetGradient(pair.uv_ref.x() * Keyframe::scales[level] + x, pair.uv_ref.y() * Keyframe::scales[level] + y, level);
                    }
                    error = pair.ref_frame->GetPixelValue(pair.uv_ref.x() * Keyframe::scales[level] + x, pair.uv_ref.y() * Keyframe::scales[level] + y, level) - pair.cur_frame->GetPixelValue(pair.uv_cur.x() * Keyframe::scales[level] + x + dx, pair.uv_cur.y() * Keyframe::scales[level] + y + dy, level);

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

            if (iter > 0 && cost > lastCost) {
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;

            if (lastCost > lk_photometric_thresh) {
                succ = false;
            } else {
                succ = true;
            }
        }

        success[i] = succ;

        pair.uv_cur += V2d{ dx / Keyframe::scales[level], dy / Keyframe::scales[level] };

        ++j;
    }
}

M2d Viso::GetAffineWarpingMatrix(Keyframe::Ptr ref_frame, Keyframe::Ptr cur_frame, V3d Pw, V2d uv_ref)
{

    V3d Pref = ref_frame->WorldToCamera(Pw);
    V3d Pdu_ref = ref_frame->PixelToImagePlane(uv_ref + V2d{ lk_half_patch_size, 0 }, 0);
    V3d Pdv_ref = ref_frame->PixelToImagePlane(uv_ref + V2d{ 0, lk_half_patch_size }, 0);

    Pdu_ref *= Pref[2];
    Pdv_ref *= Pref[2];

    const V2d uv_cur = cur_frame->Project(Pw, 0);
    const Vector2d px_du(cur_frame->Project(ref_frame->CameraToWorld(Pdu_ref), 0));
    const Vector2d px_dv(cur_frame->Project(ref_frame->CameraToWorld(Pdv_ref), 0));

    M2d A = M2d::Zero();
    A.col(0) = (px_du - uv_cur) / lk_half_patch_size;
    A.col(1) = (px_dv - uv_cur) / lk_half_patch_size;

    return A;
}

bool Viso::IsKeyframe(Keyframe::Ptr cur_frame, int nr_tracked_points)
{
    if (nr_tracked_points < 10) {
        return false;
    }
    V3d last_T = map_.GetLastPose().translation();
    V3d cur_T = cur_frame->GetT();
    V3d delta_T = (cur_T - last_T);

    double distance = delta_T.norm();

    if (distance > new_kf_dist_thresh) {
        std::cout << "IsKeyframe distance: " << distance << "\n";
        return true;
    }

    M3d last_R = map_.GetLastPose().rotationMatrix();
    M3d cur_R = cur_frame->GetR();
    M3d delta_R = cur_R.transpose() * last_R;

    double angle = std::abs(std::acos((delta_R(0, 0) + delta_R(1, 1) + delta_R(2, 2) - 1) * 0.5));

    if (angle > new_kf_angle_thresh) {
        std::cout << "IsKeyframe angle: " << angle << "\n";
        return true;
    }

    if (distance + angle_combined_ratio * angle > combined_thresh) {
        std::cout << " combined_thresh selection " << std::endl;
        return true;
    }

    return false;
}

double Viso::GetMotion(Keyframe::Ptr cur_frame)
{
    V3d last_T = map_.GetLastPose().translation();
    V3d cur_T = cur_frame->GetT();
    V3d delta_T = (cur_T - last_T);
    double distance = delta_T.norm();

    M3d last_R = map_.GetLastPose().rotationMatrix();
    M3d cur_R = cur_frame->GetR();
    M3d delta_R = cur_R.transpose() * last_R;
    double angle = std::abs(std::acos((delta_R(0, 0) + delta_R(1, 1) + delta_R(2, 2) - 1) * 0.5));

    return (distance + angle_combined_ratio * angle);
}

double Viso::CalculateVariance2(const double& nu, const Sophus::SE3d& T21,
    const std::vector<AlignmentPair>& alignment_pairs)
{
    const double n = alignment_pairs.size();
    const int max_iterations = 100;
    const double eps = 0.0001;

    double sigma2 = 0.0;
    double old_sigma2 = 1.0;

    std::vector<MapPoint::Ptr> map_points = map_.GetPoints();

    int iter = 0;
    for (; iter < max_iterations; ++iter) {
        sigma2 = 0.0;

        for (int i = 0; i < alignment_pairs.size(); ++i) {
            V3d global = alignment_pairs[i].point3d;
            V3d local = T21 * global;
            V2d uv = V2d{ local[0] * K(0, 0) / local[2] + K(0, 2), local[1] * K(1, 1) / local[2] + K(1, 2) };
            V2d error = (V2d)(alignment_pairs[i].uv_cur - uv);
            double chi2 = error.transpose() * error;
            sigma2 += chi2 * (nu + 1) / (nu + chi2 / old_sigma2);
        }

        sigma2 /= n;

        if (iter > 0 && std::abs(sigma2 - old_sigma2) / old_sigma2 < eps) {
            break;
        }

        old_sigma2 = sigma2;
    }

    //    cout << "Sigma : " << sigma2 << ", iterations: " << (iter + 1) << "\n";

    return sigma2;
}

double Viso::RemoveOutliers(const Sophus::SE3d& T21,
    std::vector<int>& tracked_points,
    std::vector<AlignmentPair>& alignment_pairs)
{
    std::vector<MapPoint::Ptr> map_points = map_.GetPoints();
    auto iter1 = alignment_pairs.begin();
    for (auto iter = tracked_points.begin(); iter != tracked_points.end();) {

        V3d global = map_points[*iter]->GetWorldPos();
        V3d local = T21 * global;
        V2d uv = V2d{ local[0] * K(0, 0) / local[2] + K(0, 2), local[1] * K(1, 1) / local[2] + K(1, 2) };
        V2d error = (V2d)((*iter1).uv_cur - uv);
        double chi2 = error.transpose() * error;

        if (chi2 > chi2_thresh) {
            iter1 = alignment_pairs.erase(iter1);
            iter = tracked_points.erase(iter);
        } else {
            ++iter;
            ++iter1;
        }
    }
}

void Viso::MotionOnlyBA(Sophus::SE3d& T21, std::vector<AlignmentPair>& alignment_pairs)
{
    assert(alignment_pairs.size() > 0);

    //    std::cout << "MBA start\n";
    const int mba_max_iterations = 100;
    const double nu = 5.0;
    std::vector<MapPoint::Ptr> map_points = map_.GetPoints();

    double last_cost = 0;
    double initial_cost = 0;
    int iter = 0;

    double chi2_min = 99999999;
    double chi2_max = 0;

    //    const M6d Sigma_inv = 0 * M6d::Identity();
    //    V6d last_tangent = last_frame->GetPose().log();

    for (; iter < mba_max_iterations; ++iter) {
        M6d H = M6d::Zero();
        V6d b = V6d::Zero();
        double cost = 0;
        double sigma2 = CalculateVariance2(nu, T21, alignment_pairs);

        for (int i = 0; i < alignment_pairs.size(); ++i) {
            V3d global = alignment_pairs[i].point3d;
            V3d local = T21 * global;
            V2d uv = V2d{ local[0] * K(0, 0) / local[2] + K(0, 2), local[1] * K(1, 1) / local[2] + K(1, 2) };
            V2d error = (V2d)(alignment_pairs[i].uv_cur - uv);
            double chi2 = error.transpose() * error;
            chi2_min = std::min(chi2, chi2_min);
            chi2_max = std::max(chi2, chi2_max);
            double w = (nu + 1) / (nu + chi2 / sigma2);

            M26d J = -dPixeldXi(K, T21.rotationMatrix(), T21.translation(), global, 1.0);
            cost += w * chi2;

            H += w * J.transpose() * J;
            b += -w * error.transpose() * J;
        }

        //        H += Sigma_inv;
        //        b += Sigma_inv * (last_tangent - T21.log());

        if (iter == 0) {
            initial_cost = cost;
        } else if (cost > last_cost) {
            break;
        } else {
            V6d update = H.inverse() * b;
            if(std::isnan(update[0])) {
              std::cout << "Motion only ba: std::isnan(update[0])\n";
              return;
            }
            T21 = Sophus::SE3d::exp(update) * T21;
        }

        last_cost = cost;
    }

    assert(!T21.matrix().hasNaN());

    //    std::cout << "MBA cost from " << initial_cost << " to " << last_cost << " in " << (iter + 1) << " iterations, chi2_min: " << chi2_min << ", chi2_max: " << chi2_max << "\n";
}

void Viso::global_ba()
{
    std::lock_guard<std::mutex> lock(update_map_);
    opt.BA(&map_, true, 2, K);
}

void Viso::local_ba()
{
    std::lock_guard<std::mutex> lock(update_map_);
    opt.BA_LOCAL(&map_, K);
}
