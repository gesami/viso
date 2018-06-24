
#include "common.h"
#include "config.h"
#include "viso.h"
#include <fstream>
#include <iostream>
#include <iostream>
#include <pangolin/pangolin.h>
#include <thread>
#include <vector>

using namespace std;
struct PangoState {
    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;
};

static bool running = true;

void DrawMap(viso::Map* map);

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
    ifstream fin(dataset_dir + "/rgb.txt");
    if (!fin) {
        cout << "no file found!" << endl;
        return 1;
    }

    vector<string> rgb_files;
    vector<string> rgb_times;

    {
        string dummy;
        getline(fin, dummy);
        getline(fin, dummy);
        getline(fin, dummy);
    }

    while (!fin.eof()) {
        string rgb_time, rgb_file;
        fin >> rgb_time >> rgb_file;
        rgb_times.push_back(rgb_time);
        rgb_files.push_back(dataset_dir + "/" + rgb_file);
        if (fin.good() == false)
            break;
    }

    //
    // Run main loop.
    //
    fx = Config::get<double>("camera.fx");
    fy = Config::get<double>("camera.fy");
    cx = Config::get<double>("camera.cx");
    cy = Config::get<double>("camera.cy");
    Viso viso(fx, fy, cx, cy);
    FrameSequence sequence(&viso, rgb_files, rgb_times);
    double qx, qy, qz, qw, x, y, z;
    ofstream out(dataset_dir + "/estimation.txt");

    std::thread ui_thread(&DrawMap, viso.GetMap());

    for ( int i=0; i<rgb_files.size(); i++ )
    {
        sequence.RunOnce();
        //Eigen::Quaternion<double> q(viso.last_frame->GetR());
        //V3d t(viso.last_frame->GetT());
        //q.normalize();
        //qx = q.x();
        //qy = q.y();
        //qz = q.z();
        //qw = q.w();
        //cout << viso.last_frame->times_<<" "<<t[0]<<" "<<t[1]<<" "<<t[2]<<" "<<qx<<" "<<qy<<" "<<qz<<" "<<qw<<endl;
        //out.open(dataset_dir + "/estimation.txt", std::ofstream::out | std::ofstream::app);
        //out << viso.last_frame->GetTime() << " " << t[0] << " " << t[1] << " " << t[2] << " " << qx << " " << qy << " " << qz << " " << qw << endl;
        //out.close();
    }

    cout << "print out trajectory to estimation.txt" << endl;
    /*for (int i=0; i<viso.GetMap()->Keyframes().size(); i++)
    {
        Eigen::Quaternion<double> q(viso.GetMap()->Keyframes()[i]->GetR());
        V3d t(viso.GetMap()->Keyframes()[i]->GetT());
        q.normalize();
        qx = q.x();
        qy = q.y();
        qz = q.z();
        qw = q.w();
        out.open(dataset_dir + "/estimation.txt", std::ofstream::out | std::ofstream::app);
        out << viso.GetMap()->Keyframes()[i]->GetTime() << " " << t[0] << " " << t[1] << " " << t[2] << " " << qx << " " << qy << " " << qz << " " << qw << endl;
        out.close();
    }*/

    for (int i=0; i<viso.frame_time.size(); i++)
    {
        cout << i << endl;
        int ref = viso.ref_key[i]; //get reference keyframe id
        cout << ref << endl;
        Sophus::SE3d key2frame = {viso.GetMap()->Keyframes()[ref]->GetR(),viso.GetMap()->Keyframes()[ref]->GetT()};
        Sophus::SE3d world2frame = key2frame*(viso.ref_pose[i]);
        cout << world2frame.matrix() << endl;
        Eigen::Quaternion<double> q(world2frame.rotationMatrix());
        V3d t(world2frame.translation());
        q.normalize();
        qx = q.x();
        qy = q.y();
        qz = q.z();
        qw = q.w();
        out.open(dataset_dir + "/estimation.txt", std::ofstream::out | std::ofstream::app);
        out << viso.frame_time[i] << " " << t[0] << " " << t[1] << " " << t[2] << " " << qx << " " << qy << " " << qz << " " << qw << endl;
        out.close();
    }
    viso.running = false;
    ui_thread.join();

    return 0;
}

void DrawMap(viso::Map* map)
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
