
#include "common.h"
#include "config.h"
#include "viso.h"
#include <fstream>
#include <iostream>
#include <iostream>
#include <pangolin/pangolin.h>
#include <thread>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "frame_sequence.h"
#include "keyframe.h"

using namespace std;
using namespace cv;
class VideoHandler {
    public:
        virtual void OnNewFrame(Keyframe::Ptr keyframe) = 0;
};
struct PangoState {
    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;
};

static bool running = true;

void DrawMap(Map* map);

double fx;
double fy;
double cx;
double cy;

int main(int argc, char const* argv[])
{

    //
    //Process data set
    //
    Config::setParameterFile(argv[1]);
    string dataset_dir = Config::get<string>("dataset_dir");
    VideoCapture cap(0); // open the video camera no. 0

    if (!cap.isOpened())  // if not success, exit program
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }

    //
    // Run main loop.
    //
    fx = Config::get<double>("camera.fx");
    fy = Config::get<double>("camera.fy");
    cx = Config::get<double>("camera.cx");
    cy = Config::get<double>("camera.cy");
    Viso viso(fx, fy, cx, cy);
    VideoHandler* handler_;
    double qx, qy, qz, qw, x, y, z;
    ofstream out(dataset_dir + "/estimation.txt");

    std::thread ui_thread(&DrawMap, viso.GetMap());

    while (running) {
        Mat frame;
        bool bSuccess = cap.read(frame); // read a new frame from video
        if (!bSuccess) //if not success, break loop
        {
             cout << "Cannot read a frame from video stream" << endl;
             break;
        }
        double tp = cap.get(CV_CAP_PROP_POS_MSEC);
        std::string str = std::to_string(tp); 
        handler_->OnNewFrame(std::make_shared<Keyframe>(frame, str ));
        //FrameSequence::FrameHandler* handler_->OnNewFrame(std::make_shared<Keyframe>(frame, (string) cap.get( CV_CAP_PROP_POS_MEC) ));
        Eigen::Quaternion<double> q(viso.last_frame->GetR());
        V3d t(viso.last_frame->GetT());
        q.normalize();
        qx = q.x();
        qy = q.y();
        qz = q.z();
        qw = q.w();
        //cout << viso.last_frame->times_<<" "<<t[0]<<" "<<t[1]<<" "<<t[2]<<" "<<qx<<" "<<qy<<" "<<qz<<" "<<qw<<endl;
        out.open(dataset_dir + "/estimation.txt", std::ofstream::out | std::ofstream::app);
        out << viso.last_frame->GetTime() << " " << t[0] << " " << t[1] << " " << t[2] << " " << qx << " " << qy << " " << qz << " " << qw << endl;
        out.close();
    }

    ui_thread.join();

    return 0;
}

void DrawMap(Map* map)
{
    //
    // Initialize pangolin.
    //
    pangolin::CreateWindowAndBind("Map", 1024, 768);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    PangoState pango_state;
    pango_state.s_cam = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 10000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    pango_state.d_cam = pangolin::CreateDisplay()
                            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                            .SetHandler(new pangolin::Handler3D(pango_state.s_cam));

    PangoState* pango = &pango_state;

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        pango->d_cam.Activate(pango->s_cam);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        auto points = map->GetPoints3d();
        auto poses = map->GetPoses();

        auto draw_points = [](const std::vector<V3d>& points) {
            glPointSize(2);
            glBegin(GL_POINTS);
            for (size_t i = 0; i < points.size(); i++) {
                glColor3f(0.0, 1.0, 0.0);
                glVertex3d(points[i].x(), points[i].y(), points[i].z());
            }
            glEnd();
        };

        draw_points(points);
        //draw_points(points_opt);

        // draw poses
        const float sz = 0.1;
        const int width = 640, height = 480;

        auto draw_poses = [&](const std::vector<Sophus::SE3d>& poses, double r, double g, double b) {
            for (auto& Tcw : poses) {
                glPushMatrix();
                Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
                glMultMatrixf((GLfloat*)m.data());
                glColor3f(r, g, b);
                glLineWidth(2);
                glBegin(GL_LINES);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glEnd();
                glPopMatrix();
            }
        };

        draw_poses(map->GetLastPoses(), 0, 1, 1);
        draw_poses(poses, 0, 0, 0.7);

        if (auto current = map->GetCurrent()) {
            draw_poses({ current->GetPose() }, 1, 0, 0);
        }
        //draw_poses(poses_opt);

        pangolin::FinishFrame();
        usleep(16000);
    }

    running = false;
}
