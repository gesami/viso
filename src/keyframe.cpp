//
// Created by sage on 06.06.18.
//

#include "keyframe.h"

long Keyframe::next_id_ = 0;
const double Keyframe::scales[] = { 1.0, 0.5, 0.25, 0.125 };

// use map points
//TODO use all map points
void Keyframe::SetOccupied(vector<V3d>& wp)
{
    std::cout << "count occupied grid!" << std::endl;
    std::cout << "grid size: " << grid_size_ << std::endl;

    for (auto& pt_w: wp){
        V3d pt_c = R_*pt_w + T_;
        V3d pt = K_*pt_c;
        double px = pt(0)/pt(2);
        double py = pt(1)/pt(2);
        if (px>0 && px<640 && py>0 && py<480){
            for (int u = -2; u < 2; u++) {
                for (int v = -2; v < 2; v++) { 
                    int x = static_cast<int>(px / grid_size_)+u;
                    int y = static_cast<int>(py / grid_size_)+v;
                    if (x<grid_col_&& x>=0 && y<grid_row_ && y>=0){
                      grid_occupy_.at( y * grid_col_ + x) = true;
                    }
                }
            }
        }
    }
}

void Keyframe::SetOccupied()
{
    std::cout << "count occupied grid!" << std::endl;
    std::cout << "grid size: " << grid_size_ << std::endl;

    for (auto& kp : keypoints_) {
        int x = static_cast<int>(kp.pt.x / grid_size_);
        int y = static_cast<int>(kp.pt.y / grid_size_);
        grid_occupy_.at( y * grid_col_ + x  ) = true;
        // for (int u = -1; u < 2; u++) {
        //    for (int v = -1; v < 2; v++) { 
                // std::cout << (y + v) * grid_col_ + (x + u) -1 << std::endl;
                //grid_occupy_.at(((y + v -1) * grid_col_ + (x + u) )) = true;
          //  }
        //}
        grid_occupy_.at(
                static_cast<int>(kp.pt.y / grid_size_) * grid_col_
                + static_cast<int>(kp.pt.x / grid_size_))
            = true;
    } 
}

void Keyframe::AddNewFeatures(std::vector<cv::KeyPoint> newfts)
{
    std::cout << "add new feature!" << std::endl;
    cv::Mat display;
    cv::cvtColor(mat_, display, CV_GRAY2BGR);
    for (auto& kp : keypoints_) {
        cv::circle(display, kp.pt, 2, cv::Scalar(0, 250, 0), 2);
    }

    int cnt = 0;
    for (auto& kp : newfts) {
        if (grid_occupy_.at(
                static_cast<int>(kp.pt.y / grid_size_) * grid_col_
                + static_cast<int>(kp.pt.x / grid_size_))
            == false) {
            cnt++;
            AddKeypointForDepthFiler(kp);
            grid_occupy_.at(static_cast<int>(kp.pt.y / grid_size_) * grid_col_ + static_cast<int>(kp.pt.x / grid_size_)) = true;
            cv::circle(display, kp.pt, 2, cv::Scalar(0, 0, 250), 2);
        }
    }

    std::cout << "cnt new feature: " << cnt << std::endl;
    cv::imshow("add new feature", display);
    cv::waitKey(0);
}
