//
// Created by sage on 06.06.18.
//

#include "keyframe.h"

long Keyframe::next_id_ = 0;
const double Keyframe::scales[] = { 1.0, 0.5, 0.25, 0.125 };
void Keyframe::SetOccupied()
{
    std::cout << "count occupied grid!" << std::endl;
    std::cout << "grid size: " << grid_size_ << std::endl;

    for (auto& kp : keypoints_) {
        grid_occupy_.at(
            static_cast<int>(kp.pt.y / grid_size_) * grid_col_
            + static_cast<int>(kp.pt.x / grid_size_))
            = true;
    }
}

void Keyframe::AddNewFeatures(std::vector<cv::KeyPoint> newfts)
{
    //    std::cout << "add new feature!" << std::endl;
    //    cv::Mat display;
    //    cv::cvtColor(mat_, display, CV_GRAY2BGR);
    //    for (auto& kp : keypoints_) {
    //        cv::circle(display, kp.pt, 2, cv::Scalar(0, 250, 0), 2);
    //    }

    int cnt = 0;
    for (auto& kp : newfts) {
        if (grid_occupy_.at(
                static_cast<int>(kp.pt.y / grid_size_) * grid_col_
                + static_cast<int>(kp.pt.x / grid_size_))
            == false) {
            cnt++;
            AddKeypointForDepthFiler(kp);
            //cv::circle(display, kp.pt, 2, cv::Scalar(0, 0, 250), 2);
        }
    }

    //		std::cout << "cnt new feature: " << cnt << std::endl;
    //		cv::imshow("add new feature", display);
    //    cv::waitKey(0);
}
