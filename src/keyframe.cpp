//
// Created by sage on 06.06.18.
//

#include "keyframe.h"

long Keyframe::next_id_ = 0;
const double Keyframe::scales[] = { 1.0, 0.5, 0.25, 0.125 };


void Keyframe::SetOccupied(std::vector<V3d> mp)
{
    //std::cout << "count occupied grid!" << std::endl;
    //std::cout << "grid size: " << grid_size_ << std::endl;
    //std::cout << "grid occupy size: " << grid_occupy_.size() << endl;
    for (int i = 0; i < mp.size(); i++) {
        //V3d Pw = mp->GetWorldPos();
        V2d uv = this->Project(mp[i], 0);
        int x = static_cast<int>(floor(uv[0] / grid_size_));
        int y = static_cast<int>(floor(uv[1] / grid_size_));
        for (int u = -1; u < 2; u++) {
            for (int v = -1; v < 2; v++) {
                if ((x + u) < 0 || (x + u + 1) > grid_col_)
                    continue;
                if ((y + v) < 0 || (y + v + 1) > grid_row_)
                    continue;
//                cout << x + u << "," << y + v << endl;
//                cout << (y + v) * grid_col_ + (x + u) << endl;
                grid_occupy_.at((y + v) * grid_col_ + (x + u)) = true;
            }
        }
        
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
        int x = static_cast<int>(floor(kp.pt.x / grid_size_));
        int y = static_cast<int>(floor(kp.pt.y / grid_size_));
        //cout << kp.pt.x << " " << kp.pt.y << " " << grid_col_ << " " << x << " " << y<< endl;
        //cout << y  * grid_col_ + x << endl;
        if (grid_occupy_.at(y * grid_col_ + x) == false) {
            cnt++;
            AddKeypointForDepthFiler(kp);
//            cv::circle(display, kp.pt, 2, cv::Scalar(0, 0, 250), 2);
        }
    }

//    std::cout << "cnt new feature: " << cnt << std::endl;
//    cv::imshow("add new feature", display);
//    cv::waitKey(0);
}
