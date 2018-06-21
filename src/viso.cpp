#include "viso.h"
#include "common.h"
#include "timer.h"

#include <depth_filter.h>
#include <map>
#include <opencv2/core/eigen.hpp>

void Viso::OnNewFrame(Keyframe::Ptr cur_frame)
{
    static depth_filter* filter = nullptr;

    // TODO: Clean this up.
    cur_frame->SetK(K);

    switch (state_) {
    case kInitialization: {
        // Visualization
        cv::Mat img;
        cv::cvtColor(cur_frame->Mat(), img, CV_GRAY2BGR);
        bool initialized = initializer.InitializeMap(cur_frame, &map_, img);
        cv::imshow("Optical flow", img);
        cv::waitKey(10);
        if (initialized) {
            state_ = kRunning;
            BA(true, 1, {}, {}, {});
        }
    } break;

    case kRunning: {
        Sophus::SE3d X = Sophus::SE3d(last_frame->GetR(), last_frame->GetT());
        DirectPoseEstimationMultiLayer(cur_frame, X);

        cur_frame->SetR(X.rotationMatrix());
        cur_frame->SetT(X.translation());

        map_.SetCurrent(cur_frame);

        std::vector<V2d> kp_before, kp_after;
        std::vector<int> tracked_points;
        LKAlignment(cur_frame, kp_before, kp_after, tracked_points);

        cv::Mat display;
        cv::cvtColor(cur_frame->Mat(), display, CV_GRAY2BGR);

        for (int i = 0; i < kp_after.size(); ++i) {
            int r = (i % 3) == 0;
            int g = (i % 3) == 1;
            int b = (i % 3) == 2;
            cv::rectangle(display, cv::Point2f(kp_before[i].x() - 4, kp_before[i].y() - 4), cv::Point2f(kp_before[i].x() + 4, kp_before[i].y() + 4),
                cv::Scalar(255 * b, 255 * g, 255 * r));

            cv::rectangle(display, cv::Point2f(kp_after[i].x() - 4, kp_after[i].y() - 4), cv::Point2f(kp_after[i].x() + 4, kp_after[i].y() + 4),
                cv::Scalar(255 * b, 255 * g, 255 * r));
        }

        cv::imshow("Tracked", display);
        cv::waitKey(10);

        // for now there is only one active filter
        if (filter != nullptr) {
            if (filter->IsDone()) {
                delete filter;
                filter = nullptr;
            } else {
                filter->Update(cur_frame);
                //filter->UpdateMap(&map_);
            }
        }

        if (IsKeyframe(cur_frame)) {
            assert(cur_frame->Keypoints().size() == 0);

            std::vector<cv::KeyPoint> kp;
            map_.AddKeyframe(cur_frame);
            featureDetector->detect(cur_frame->Mat(), kp);
            cur_frame->SetOccupied();
            cur_frame->AddNewFeatures(kp);

            vector<MapPoint::Ptr> map_points = map_.GetPoints();

            for (int i = 0; i < tracked_points.size(); ++i) {
                cv::KeyPoint kp;
                kp.pt.x = kp_after[i].x();
                kp.pt.y = kp_after[i].y();
                int kp_idx = cur_frame->AddKeypoint(kp);
                map_points[tracked_points[i]]->AddObservation(cur_frame, kp_idx);
            }

            // TODO: What else do we have to do here?
            if (filter == nullptr) {
                filter = new depth_filter(cur_frame);
            }
            std::cout << "New keyframe added!\n";

            BA(true, 2, {}, {}, {});
        }

    } break;

    default:
        break;
    }

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

void Viso::LKAlignment(Keyframe::Ptr current_frame, std::vector<V2d>& kp_before, std::vector<V2d>& kp_after, std::vector<int>& tracked_points)
{
    std::vector<AlignmentPair> alignment_pairs;

    const double max_angle = 180.0; // 180 means basically no restriction on the angle (for now)

    for (size_t i = 0; i < map_.GetPoints().size(); i++) {

        MapPoint::Ptr map_point = map_.GetPoints()[i];
        V3d Pw = map_point->GetWorldPos();

        if (!current_frame->IsInside(Pw, /*level=*/0)) {
            continue;
        }

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

        alignment_pairs.push_back(pair);
        tracked_points.push_back(i);
    }

    std::vector<bool> success;

    for (int i = 0; i < alignment_pairs.size(); ++i) {
        kp_before.push_back(alignment_pairs[i].uv_cur);
    }

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
    for (auto iter1 = kp_before.begin(); iter1 != kp_before.end(); ++i) {
        if (!success[i]) {
            iter1 = kp_before.erase(iter1);
            iter2 = kp_after.erase(iter2);
            iter3 = tracked_points.erase(iter3);
        } else {
            ++iter1;
            ++iter2;
            ++iter3;
        }
    }
}

void Viso::LKAlignmentSingle(std::vector<AlignmentPair>& pairs, std::vector<bool>& success, std::vector<V2d>& kp, int level)
{
    // parameters
    const bool inverse = true;
    const int iterations = 100;

    success.clear();
    kp.clear();

    for (size_t i = 0; i < pairs.size(); i++) {
        AlignmentPair& pair = pairs[i];

        double dx = 0, dy = 0; // dx,dy need to be estimated
        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; iter++) {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (
                !pair.ref_frame->IsInside(pair.uv_ref.x() * Keyframe::scales[level] + dx - lk_half_patch_size, pair.uv_ref.y() * Keyframe::scales[level] + dy - lk_half_patch_size, level) || !pair.ref_frame->IsInside(pair.uv_ref.x() * Keyframe::scales[level] + dx + lk_half_patch_size, pair.uv_ref.y() * Keyframe::scales[level] + dy + lk_half_patch_size, level)) {
                succ = false;
                break;
            }

            double error = 0;
            // compute cost and jacobian
            for (int x = -lk_half_patch_size; x < lk_half_patch_size; x++) {
                for (int y = -lk_half_patch_size; y < lk_half_patch_size; y++) {
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

        success.push_back(succ);
        pair.uv_cur += V2d{ dx / Keyframe::scales[level], dy / Keyframe::scales[level] };
    }

    for (int i = 0; i < pairs.size(); ++i) {
        kp.push_back(pairs[i].uv_cur);
    }
}

void Viso::BA(bool map_only, int fix_cnt, Keyframe::Ptr current_frame, const std::vector<V2d>& kp, const std::vector<int>& tracked_points)
{
    using KernelType = g2o::RobustKernelHuber;

    std::cout << "start BA" << std::endl;
    // build optimization problem
    // setup g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3> > Block; // pose is 6x1, landmark is 3x1
    std::unique_ptr<Block::LinearSolverType> linearSolver(
        new g2o::LinearSolverDense<Block::PoseMatrixType>()); // linear solver

    // use levernberg-marquardt here (or you can choose gauss-newton)
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<Block>(std::move(linearSolver)));
    g2o::SparseOptimizer optimizer; // graph optimizer
    optimizer.setAlgorithm(solver); // solver
    optimizer.setVerbose(false); // open the output

    std::vector<VertexSBAPointXYZ*> points_v;
    std::vector<VertexSophus*> cameras_v;
    int id = 0;
    int tracked_idx = 0;

    for (size_t i = 0; i < map_.GetPoints().size(); i++, id++) { //for each mappoint
        MapPoint::Ptr mp = map_.GetPoints()[i];
        VertexSBAPointXYZ* p = new VertexSBAPointXYZ();
        p->setId(id);
        p->setMarginalized(true);
        p->setEstimate(mp->GetWorldPos());
        optimizer.addVertex(p);
        points_v.push_back(p);
    }

    std::vector<Keyframe::Ptr> keyframes = map_.Keyframes();
    std::map<int, int> keyframe_indices;

    for (size_t i = 0; i < keyframes.size(); i++, id++) {
        VertexSophus* cam = new VertexSophus();
        cam->setId(id);
        if (i < fix_cnt) {
            cam->setFixed(true); //fix the pose of the first frame
        }
        cam->setEstimate(keyframes[i]->GetPose());
        optimizer.addVertex(cam);
        cameras_v.push_back(cam);
        keyframe_indices.insert({ keyframes[i]->GetId(), i });
    }

    if (!map_only) {
        VertexSophus* cam = new VertexSophus();
        cam->setId(id);
        cam->setEstimate(current_frame->GetPose());
        optimizer.addVertex(cam);
        cameras_v.push_back(cam);
    }

    for (size_t i = 0; i < map_.GetPoints().size(); i++) {
        //std::cout << i << "th mappint" << std::endl;
        MapPoint::Ptr mp = map_.GetPoints()[i];
        for (size_t j = 0; j < mp->GetObservations().size(); j++, id++) { //for each observation
            std::pair<Keyframe::Ptr, int> obs = mp->GetObservations()[j];
            EdgeObservation* e = new EdgeObservation(K);
            e->setVertex(0, points_v[i]);
            e->setVertex(1, cameras_v[keyframe_indices.find(obs.first->GetId())->second]);
            e->setInformation(Eigen::Matrix2d::Identity()); //intensity is a scale?
            int idx = obs.second;
            V2d xy(obs.first->Keypoints()[idx].pt.x, obs.first->Keypoints()[idx].pt.y);
            e->setMeasurement(xy);
            e->setId(id);
            KernelType* robustKernel = new KernelType();
            robustKernel->setDelta(ba_outlier_thresh);
            e->setRobustKernel(robustKernel);
            optimizer.addEdge(e);
        }

        if (!map_only && tracked_idx < tracked_points.size() && tracked_points[tracked_idx] == i) {
            EdgeObservation* e = new EdgeObservation(K);
            e->setVertex(0, points_v[i]);
            e->setVertex(1, cameras_v[cameras_v.size() - 1]);
            e->setInformation(Eigen::Matrix2d::Identity()); //intensity is a scale?
            e->setMeasurement(kp[tracked_idx]);
            e->setId(id);
            KernelType* robustKernel = new KernelType();
            robustKernel->setDelta(ba_outlier_thresh);
            e->setRobustKernel(robustKernel);
            optimizer.addEdge(e);

            ++tracked_idx;
        }
    }

    // perform optimization
    std::cout << "optimize!" << std::endl;
    optimizer.initializeOptimization(0);
    optimizer.optimize(BA_iteration);
    std::cout << "end!" << std::endl;

    poses_opt.clear();
    poses.clear();
    for (int i = 0; i < keyframes.size(); i++) {
        VertexSophus* pose = dynamic_cast<VertexSophus*>(optimizer.vertex(map_.GetPoints().size() + i));
        Sophus::SE3d p_opt = pose->estimate();

        poses.push_back(keyframes[i]->GetPose());
        keyframes[i]->SetPose(p_opt);
        poses_opt.push_back(p_opt);
    }

    if (!map_only) {
        VertexSophus* pose = cameras_v[cameras_v.size() - 1];
        poses.push_back(current_frame->GetPose());
        current_frame->SetPose(pose->estimate());
        poses_opt.push_back(pose->estimate());
    }

    points_opt.clear();
    for (int i = 0; i < map_.GetPoints().size(); i++) {
        VertexSBAPointXYZ* point = dynamic_cast<VertexSBAPointXYZ*>(optimizer.vertex(i));
        V3d point_opt = point->estimate();
        map_.GetPoints()[i]->SetWorldPos(point_opt);
        //points_opt.push_back(point_opt);
    }
}

bool Viso::IsKeyframe(Keyframe::Ptr cur_frame)
{
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

    return false;
}
