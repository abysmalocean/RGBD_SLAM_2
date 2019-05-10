#include <iostream>
#include "utility.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <dirent.h>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <opencv2/xfeatures2d/nonfree.hpp>

#include <omp.h>

//g2o的头文件
#include <g2o/types/slam3d/types_slam3d.h> //顶点类型
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"

#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>


using std::cout; 
using std::endl;

int main( int argc, char** argv )
{
    //Read data
    ParameterReader pd;
    int scaleOfGoodMatch = atoi( pd.getData( "scaleOfGoodMatch" ).c_str() );
    int display = atoi( pd.getData( "display" ).c_str() );
    int imgDisplay = atoi( pd.getData( "imageDisplay" ).c_str() );
    int width  =   atoi( pd.getData( "width" ).c_str() );
    int height    =   atoi( pd.getData( "height"   ).c_str() );

    std::string folder = pd.getData("DataFolder"); 
    cout << "Working on the folder --> [" << folder << "]<---" << endl; 
    

    int nFrames = 0;

	std::string imgsFolder = folder;
	int count = 0; 
	std::vector<std::string> paths; 
	if (auto dir = opendir(imgsFolder.c_str()))
	{
		while(auto f= readdir(dir))
		{
			if (!f->d_name || f->d_name[0] == '.')
                continue; // Skip everything that starts with a dot
			std::string filePath = imgsFolder + "/" + f->d_name; 
			//printf("File: %s | num [%zu]\n", f->d_name, count++);
			paths.push_back(filePath); 
			++count;
		}
	}
    

    std::vector<FRAME> source(count); 
    count = 0;
    std::sort(paths.begin(), paths.end()); 
    
    cout << "Loading the Source Image to memory" << endl; 

    #pragma omp parallel for
    for (int i = 0; i < paths.size(); ++i)
    {
        FRAME frame = readImage(paths[i], &pd);
        //frame.frameID = count++; 
        source[i] = frame; 

    }
    printf("Source finish \n");
    int nOctaveLayers =  atoi( pd.getData( "nOctaveLayers" ).c_str() );
    double contrastThreshold = atoi(pd.getData( "contrastThreshold" ).c_str())/1000.0; 
    cv::Ptr<cv::Feature2D> detector = 
            cv::xfeatures2d::SIFT::create(0,nOctaveLayers, contrastThreshold);

    #pragma omp parallel for
    for (int i = 0; i < source.size(); ++i)
    {
        detector->detectAndCompute(source[i].rgb, cv::Mat(), 
                                    source[i].kp, source[i].desp);
    }
    cout << "Finish Computing the key points" << endl; 

    typedef g2o::BlockSolver_6_3 SlamBlockSolver; 
    typedef g2o::LinearSolverCholmod< SlamBlockSolver::PoseMatrixType > SlamLinearSolver;
    // initialze the solver
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering( false );
    SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( blockSolver );
    g2o::SparseOptimizer globalOptimizer; 
    globalOptimizer.setAlgorithm( solver ); 
    // on output information
    globalOptimizer.setVerbose( true );

    int currIndex = 0; 

    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId( currIndex );
    v->setEstimate( Eigen::Isometry3d::Identity() );
    v->setFixed( true );
    globalOptimizer.addVertex( v );
    cout << "Finish initialze the optimizer " << endl; 

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    // flann mather
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>();

    for (int i = 1; i < source.size(); ++i)
    {
        // find matches
        std::vector<cv::DMatch> matches;
        FRAME f1 = source[currIndex]; 
        FRAME f2 = source[i]; 

        if (imgDisplay)
        {
            cv::imshow("Frame1", f1.rgb); 
            cv::imshow("Frame2", f2.rgb); 
            cv::waitKey(0); 

        }

        matcher->match( f1.desp, f2.desp , matches );
        double max_dist = 0; double min_dist = 100;
        for( int i = 0; i < f1.desp.rows; i++ )
        { double dist = matches[i].distance;
          if( dist < min_dist ) min_dist = dist;
          if( dist > max_dist ) max_dist = dist;
        }
        std::vector< cv::DMatch > goodMatches;
        for( int i = 0; i < f1.desp.rows; i++ )
        { if( matches[i].distance <= std::max(scaleOfGoodMatch*min_dist, 0.02) )
          { goodMatches.push_back( matches[i]); }
        }

        cout << "Image " << i << " good Matches " << goodMatches.size() << endl;
        if (goodMatches.size() < 5) continue; 
        // 3D poitns
        std::vector<cv::Point3d> src; 
        std::vector<cv::Point3d> dst; 


        for (size_t i = 0; i<goodMatches.size(); ++i)
        {
            cv::Point2d p1 = f1.kp[goodMatches[i].queryIdx].pt;
            cv::Point2d p2 = f2.kp[goodMatches[i].trainIdx].pt;
            cv::Point3d point1; 
            cv::Point3d point2;
            point1.x = f1.depth_x.at<double>(int(p1.y), int(p1.x)); 
            point1.y = f1.depth_y.at<double>(int(p1.y), int(p1.x)); 
            point1.z = f1.depth_z.at<double>(int(p1.y), int(p1.x));
            point2.x = f2.depth_x.at<double>(int(p2.y), int(p2.x)); 
            point2.y = f2.depth_y.at<double>(int(p2.y), int(p2.x)); 
            point2.z = f2.depth_z.at<double>(int(p2.y), int(p2.x));
            src.push_back(point1); 
            dst.push_back(point2);
        }

        int half = src.size() * 0.6;
        double threshold = 5.0; 
        cv::Mat rvec, translationVec, inliers, ratationVector;
        cv::Mat affine = cv::Mat::zeros(3,4,CV_64F);
    
        count = 0; 
        while (count < half)
        {
            threshold += 0.4;
            cv::estimateAffine3D(src, dst, affine,inliers, threshold ,0.98);
            count = 0; 
            for (int i = 0; i < src.size(); ++i)
            {
                if(inliers.at<bool>(0,i) == true)
                {
                    ++count; 
                }
            }
        }
        cout << "Current threshold " << threshold <<endl;
        //if (threshold > 50.0) continue; 
        std::vector<cv::Point3d> srcSVD; 
        std::vector<cv::Point3d> dstSVD; 

        for (int i = 0; i < src.size(); ++i)
        {
            if(inliers.at<bool>(0,i) == true)
            {
                srcSVD.push_back(src[i]); 
                dstSVD.push_back(dst[i]); 
            }
        }
        std::vector<double> t(3); 
        std::vector<double> Rot(3); 
        ResultOfSVD Rt = poseEstimation3D3DReturn(srcSVD, dstSVD, Rot, t);
        T(0,3) = Rt.t_(0,0); 
        T(1,3) = Rt.t_(1,0); 
        T(2,3) = Rt.t_(2,0); 
        T(0,0) = Rt.R_(0,0);
        T(0,1) = Rt.R_(0,1);
        T(0,2) = Rt.R_(0,2);
        T(1,0) = Rt.R_(1,0);
        T(1,1) = Rt.R_(1,1);
        T(1,2) = Rt.R_(1,2);
        T(2,0) = Rt.R_(2,0);
        T(2,1) = Rt.R_(2,1);
        T(2,2) = Rt.R_(2,2);
        
        // add vertex(Nodes ) 
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId( i );
        v->setEstimate( Eigen::Isometry3d::Identity() );
        globalOptimizer.addVertex(v);

        // add edge
        g2o::EdgeSE3* edge = new g2o::EdgeSE3();
        edge->vertices() [0] = globalOptimizer.vertex( currIndex );
        edge->vertices() [1] = globalOptimizer.vertex( i );

        // Inforamtion Matrix
        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();

        information(0,0) = information(1,1) = information(2,2) = 20;
        information(3,3) = information(4,4) = information(5,5) = 20;
        edge->setInformation( information );
        edge->setMeasurement( T );
        globalOptimizer.addEdge(edge);


        currIndex = i; 

    }
    cout<<"optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<endl;
    globalOptimizer.save("./result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize( 1000 );
    globalOptimizer.save( "./result_after.g2o" );
    cout<<"Optimization done."<<endl;

    globalOptimizer.clear();
    

   
    return 0; 
}

