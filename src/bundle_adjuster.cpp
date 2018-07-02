#include "bundle_adjuster.h"

Opt::Opt()
{
}
void Opt::BA_LOCAL(viso::Map* map, M3d K)
{
    std::cout << "start BA for only keyframes" << std::endl;
    // build optimization problem
    // setup g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block; // pose is 6x1, landmark is 3x1
    std::unique_ptr<Block::LinearSolverType> linearSolver(
        new g2o::LinearSolverDense<Block::PoseMatrixType>()); // linear solver

    // use levernberg-marquardt here (or you can choose gauss-newton)
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<Block>(std::move(linearSolver)));
    g2o::SparseOptimizer optimizer; // graph optimizer
    optimizer.setAlgorithm(solver); // solver
    optimizer.setVerbose(false);    // open the output


    std::vector<VertexSBAPointXYZ *> points_v;
    std::vector<VertexSophus *> cameras_v;
    int id = 0;

    std::vector<MapPoint::Ptr> map_points = map->GetPoints();

    for (size_t i = 0; i < map_points.size(); i++, id++)
    { //for each mappoint
        VertexSBAPointXYZ *p = new VertexSBAPointXYZ();
        p->setId(id);
        p->setMarginalized(true);
        p->setEstimate(map_points[i]->GetWorldPos());
        optimizer.addVertex(p);
        points_v.push_back(p);
    }

    std::vector<Keyframe::Ptr> keyframes = map->Keyframes();
    std::map<int, int> keyframe_indices;

    for (size_t i = 0; i < keyframes.size(); i++, id++)
    {
        VertexSophus *cam = new VertexSophus();
        cam->setId(id);
        if(keyframes.size()>=(window+1)){
            if(i < keyframes.size()-window) cam->setFixed(true);
            //else cam->setFixed(false);
        }
        else{
            if(i < 2) cam->setFixed(true);
            //else cam->setFixed(false);
        }
        cam->setEstimate(keyframes[i]->GetPose());
        optimizer.addVertex(cam);
        cameras_v.push_back(cam);
        keyframe_indices.insert({keyframes[i]->GetId(), i});
    }
    const float deltaMono = sqrt(5.991);
    std::vector<EdgeObservation *> edge_v;
    std::vector<std::pair<int, int> > obs_idx; //first: mappoint index, second: observation index
    for (size_t i = 0; i < map_points.size(); i++)
    {
        //std::cout << i << "th mappint" << std::endl;
        MapPoint::Ptr mp = map_points[i];
        for (size_t j = 0; j < mp->GetObservations().size(); j++, id++)
        { //for each observation
            std::pair<Keyframe::Ptr, int> obs = mp->GetObservations()[j];
            EdgeObservation *e = new EdgeObservation(K);
            e->setVertex(0, points_v[i]);
            e->setVertex(1, cameras_v[keyframe_indices.find(obs.first->GetId())->second]);
            e->setInformation(Eigen::Matrix2d::Identity()); //intensity is a scale?
            int idx = obs.second;
            V2d xy(obs.first->Keypoints()[idx].pt.x, obs.first->Keypoints()[idx].pt.y);
            e->setMeasurement(xy);
            e->setId(id);
            if(add_huber_kernal){
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);
            }
            optimizer.addEdge(e);
            edge_v.push_back(e);
            obs_idx.push_back({i,j});
        }
    }

    // perform optimization
    //set terminateAction
    g2o::SparseOptimizerTerminateAction* terminateAction = 0;
    terminateAction = new SparseOptimizerTerminateAction;
    terminateAction->setGainThreshold(gain);
    terminateAction->setMaxIterations(30);
    optimizer.addPostIterationAction(terminateAction);

    std::cout << "optimize!" << std::endl;
    optimizer.initializeOptimization(0);
    optimizer.optimize(5);
    std::cout << "end!" << std::endl;
    
    double error_avg=0;
    std::vector<int> out_cnt(map_points.size(), 0);
    // Check inlier observations
    for(size_t i=0; i < edge_v.size(); i++)
    {
        EdgeObservation* e = edge_v[i];
        //cout << e->chi2() << endl;
        error_avg += e->chi2();
        if(e->chi2()>thresh || !e->isDepthPositive())
        {
            e->setLevel(1);
            if(remove){
                map_points[obs_idx[i].first]->EraseObservation(obs_idx[i].second);
                out_cnt[obs_idx[i].first]++;
                cout << "erase observation: " << obs_idx[i].second << "of mappoint " <<  obs_idx[i].first << endl;
            }
        }

        e->setRobustKernel(0);
    }
    error_avg = error_avg/edge_v.size();
    cout << "error_avg" << error_avg << endl;
    std::cout << "optimize again!" << std::endl;
    // Optimize again without the outliers
    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
    std::cout << "end!" << std::endl;
    
    {
        //std::lock_guard<std::mutex> lock(update_map_);

        for (int i = 0; i < keyframes.size(); i++)
        {
            VertexSophus *pose = cameras_v[i];
            map->Keyframes()[i]->SetPose(pose->estimate());
        }

        for (int i = 0; i < map_points.size(); i++)
        {
            VertexSBAPointXYZ *point = points_v[i];
            map->GetPoints()[i]->SetWorldPos(point->estimate());
        }

        //remove map points
        int k = 0;
        for (int i = 0; i < map_points.size();)
        {
            if (out_cnt[k] > out_thresh){
                map->ErasePoint(i);
                //cout << "remove mappoint" << endl;
            }
            else
                ++i;
            k++;
        } 

        /*int k = 0;
        for (auto it = map->GetPoints().begin(); it != map->GetPoints().end(); )
        {
            if (out_cnt[k] > out_thresh){
                it = map->GetPoints().erase(it);
                //cout << "remove mappoint" << endl;
            }
            else
                ++it;
            k++;
        }*/ 
    }
 
}


void Opt::BA_LOCAL2(viso::Map* map, M3d K)
{
    std::cout << "start BA for only keyframes" << std::endl;
    // build optimization problem
    // setup g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block; // pose is 6x1, landmark is 3x1
    std::unique_ptr<Block::LinearSolverType> linearSolver(
        new g2o::LinearSolverDense<Block::PoseMatrixType>()); // linear solver

    // use levernberg-marquardt here (or you can choose gauss-newton)
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<Block>(std::move(linearSolver)));
    g2o::SparseOptimizer optimizer; // graph optimizer
    optimizer.setAlgorithm(solver); // solver
    optimizer.setVerbose(false);    // open the output

    std::vector<VertexSBAPointXYZ *> points_v;
    std::vector<VertexSophus *> cameras_v;
    int id = 0;

    std::vector<MapPoint::Ptr> map_points = map->GetPoints();

    for (size_t i = 0; i < map_points.size(); i++, id++)
    { //for each mappoint
        VertexSBAPointXYZ *p = new VertexSBAPointXYZ();
        p->setId(id);
        p->setMarginalized(true);
        p->setEstimate(map_points[i]->GetWorldPos());
        optimizer.addVertex(p);
        points_v.push_back(p);
    }

    std::vector<Keyframe::Ptr> keyframes = map->Keyframes();
    std::map<int, int> keyframe_indices;

    for (size_t i = 0; i < keyframes.size(); i++, id++)
    {
        VertexSophus *cam = new VertexSophus();
        cam->setId(id);
        if(keyframes.size()>= (window+1)){
            if(i < keyframes.size()-window) cam->setFixed(true);
            //else cam->setFixed(false);
        }
        else{
            if(i < 2) cam->setFixed(true);
            //else cam->setFixed(false);
        }
        cam->setEstimate(keyframes[i]->GetPose());
        optimizer.addVertex(cam);
        cameras_v.push_back(cam);
        keyframe_indices.insert({keyframes[i]->GetId(), i});
    }
    const float deltaMono = sqrt(5.991);
    std::vector<EdgeObservation *> edge_v;
    std::vector<bool> NotIncludedP;
    for (size_t i = 0; i < map_points.size(); i++)
    {
        //std::cout << i << "th mappint" << std::endl;
        MapPoint::Ptr mp = map_points[i];
        bool seen = false;
        for (size_t j = 0; j < mp->GetObservations().size(); j++, id++)
        { //for each observation
            std::pair<Keyframe::Ptr, int> obs = mp->GetObservations()[j];
            EdgeObservation *e = new EdgeObservation(K);
            if(keyframes.size() < (window+1)){
                seen = true;
                e->setVertex(0, points_v[i]);
                e->setVertex(1, cameras_v[keyframe_indices.find(obs.first->GetId())->second]);
                e->setInformation(Eigen::Matrix2d::Identity()); //intensity is a scale?
                int idx = obs.second;
                V2d xy(obs.first->Keypoints()[idx].pt.x, obs.first->Keypoints()[idx].pt.y);
                e->setMeasurement(xy);
                e->setId(id);
                if(add_huber_kernal){
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);
                }
                optimizer.addEdge(e);
            }
            else{
                if(keyframe_indices.find(obs.first->GetId())->second >= keyframes.size()-window){
                    cout << keyframe_indices.find(obs.first->GetId())->second << endl;
                    seen = true;
                    e->setVertex(0, points_v[i]);
                    e->setVertex(1, cameras_v[keyframe_indices.find(obs.first->GetId())->second]);
                    e->setInformation(Eigen::Matrix2d::Identity()); //intensity is a scale?
                    int idx = obs.second;
                    V2d xy(obs.first->Keypoints()[idx].pt.x, obs.first->Keypoints()[idx].pt.y);
                    e->setMeasurement(xy);
                    e->setId(id);
                    if(add_huber_kernal){
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);
                    }
                    optimizer.addEdge(e);
                }
            }
        }
        if(!seen){
            NotIncludedP.push_back(true);
            optimizer.removeVertex(points_v[i]);
        }
        else NotIncludedP.push_back(false);
    }

    // perform optimization
    std::cout << "optimize!" << std::endl;
    optimizer.initializeOptimization(0);
    optimizer.optimize(30);
    std::cout << "end!" << std::endl;
    
    
    // Check inlier observations
    for(size_t i=0; i < edge_v.size(); i++)
    {
        EdgeObservation* e = edge_v[i];

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }
    std::cout << "optimize again!" << std::endl;
    // Optimize again without the outliers
    optimizer.initializeOptimization(0);
    optimizer.optimize(70);
    std::cout << "end!" << std::endl;
    
    {
        //std::lock_guard<std::mutex> lock(update_map_);

        for (int i = 0; i < keyframes.size(); i++)
        {
            VertexSophus *pose = cameras_v[i];
            map->Keyframes()[i]->SetPose(pose->estimate());
        }

        for (int i = 0; i < map_points.size(); i++)
        {
            if(NotIncludedP[i]) continue;
            VertexSBAPointXYZ *point = points_v[i];
            map->GetPoints()[i]->SetWorldPos(point->estimate());
        }
    }
 
}

void Opt::BA(viso::Map* map, bool map_only, int fix_cnt, M3d K)
{
    using KernelType = g2o::RobustKernelHuber;

    std::cout << "start BA" << std::endl;
    // build optimization problem
    // setup g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block; // pose is 6x1, landmark is 3x1
    std::unique_ptr<Block::LinearSolverType> linearSolver(
        new g2o::LinearSolverDense<Block::PoseMatrixType>()); // linear solver

    // use levernberg-marquardt here (or you can choose gauss-newton)
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<Block>(std::move(linearSolver)));
    g2o::SparseOptimizer optimizer; // graph optimizer
    optimizer.setAlgorithm(solver); // solver
    optimizer.setVerbose(false);    // open the output

    std::vector<VertexSBAPointXYZ *> points_v;
    std::vector<VertexSophus *> cameras_v;
    int id = 0;

    std::vector<MapPoint::Ptr> map_points = map->GetPoints();

    for (size_t i = 0; i < map_points.size(); i++, id++)
    { //for each mappoint
        VertexSBAPointXYZ *p = new VertexSBAPointXYZ();
        p->setId(id);
        p->setMarginalized(true);
        p->setEstimate(map_points[i]->GetWorldPos());
        optimizer.addVertex(p);
        points_v.push_back(p);
    }

    std::vector<Keyframe::Ptr> keyframes = map->Keyframes();
    std::map<int, int> keyframe_indices;

    for (size_t i = 0; i < keyframes.size(); i++, id++)
    {
        VertexSophus *cam = new VertexSophus();
        cam->setId(id);
        if (i < fix_cnt)
        {
            cam->setFixed(true); //fix the pose of the first frame
        }
        cam->setEstimate(keyframes[i]->GetPose());
        optimizer.addVertex(cam);
        cameras_v.push_back(cam);
        keyframe_indices.insert({keyframes[i]->GetId(), i});
    }
    const float deltaMono = sqrt(5.991);
    for (size_t i = 0; i < map_points.size(); i++)
    {
        //std::cout << i << "th mappint" << std::endl;
        MapPoint::Ptr mp = map_points[i];
        for (size_t j = 0; j < mp->GetObservations().size(); j++, id++)
        { //for each observation
            std::pair<Keyframe::Ptr, int> obs = mp->GetObservations()[j];
            EdgeObservation *e = new EdgeObservation(K);
            e->setVertex(0, points_v[i]);
            e->setVertex(1, cameras_v[keyframe_indices.find(obs.first->GetId())->second]);
            e->setInformation(Eigen::Matrix2d::Identity()); //intensity is a scale?
            int idx = obs.second;
            V2d xy(obs.first->Keypoints()[idx].pt.x, obs.first->Keypoints()[idx].pt.y);
            e->setMeasurement(xy);
            e->setId(id);
            //KernelType *robustKernel = new KernelType();
            //robustKernel->setDelta(ba_outlier_thresh);
            //e->setRobustKernel(robustKernel);
            if(add_huber_kernal){
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);
            }
            optimizer.addEdge(e);
        }
    }

    // perform optimization
    std::cout << "optimize!" << std::endl;
    optimizer.initializeOptimization(0);
    optimizer.optimize(BA_iteration);
    std::cout << "end!" << std::endl;

    {
        //std::lock_guard<std::mutex> lock(update_map_);

        for (int i = 0; i < keyframes.size(); i++)
        {
            VertexSophus *pose = cameras_v[i];
            Sophus::SE3d p_opt = pose->estimate();
            map->Keyframes()[i]->SetPose(p_opt);
        }

        for (int i = 0; i < map_points.size(); i++)
        {
            VertexSBAPointXYZ *point = points_v[i];
            V3d point_opt = point->estimate();
            map->GetPoints()[i]->SetWorldPos(point_opt);
        }
    }
}