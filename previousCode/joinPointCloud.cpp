/*
CMSC 591 Slam Project
First Part, VO 
Author: Liang Xu
Data: 04/2019
Contact: liangxuav@gmail.com
*/

#include <iostream>
#include <../include/utility.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <dirent.h>
#include <string>

#include <opencv2/xfeatures2d/nonfree.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <pcl/common/transforms.h>

using std::cout; 
using std::endl;




int main( int argc, char** argv )
{
    //Read data
    ParameterReader pd;
    int display = atoi( pd.getData( "display" ).c_str() );
    int imgDisplay = atoi( pd.getData( "imageDisplay" ).c_str() );
    int width  =   atoi( pd.getData( "width" ).c_str() );
    int height    =   atoi( pd.getData( "height"   ).c_str() );

    std::string folder = pd.getData("DataFolder"); 
    std::vector<FRAME> source; 

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
			
		}
	}
    count = 0;
    std::sort(paths.begin(), paths.end()); 
    int frame1 = atoi( pd.getData( "frame1" ).c_str() );
    int frame2 = atoi( pd.getData( "frame2" ).c_str() );

    FRAME f1 = readImage(paths[frame1], &pd);
    FRAME f2 = readImage(paths[frame2], &pd);
    std::cout << "working on : \n" << paths[frame1] << "\n" << paths[frame2] << endl;  
    
    if (imgDisplay)
    {
        cv::imshow("Frame1", f1.rgb); 
        cv::imshow("Frame2", f2.rgb); 
        cv::waitKey(0); 
    } 

    int nfeatures = atoi( pd.getData( "nfeatures" ).c_str() );
    int nOctaveLayers =  atoi( pd.getData( "nOctaveLayers" ).c_str() );
    double contrastThreshold = atoi(pd.getData( "contrastThreshold" ).c_str())/1000.0; 
    double edgeThreshold  = atoi(pd.getData( "edgeThreshold" ).c_str()) * 1.0; 
    double sigma         = atoi(pd.getData( "sigma" ).c_str())/10.0; 

    float scaleFactor = atoi(pd.getData( "scaleFactor" ).c_str())/100.0; 

    if (display)
    {
        cout << "Image height " << height << " width is " << width << endl; 
        cout << "nFeature: " << nfeatures << endl << "nOctaveLayers : " << nOctaveLayers << endl;
        cout << "scaleFactor: " << scaleFactor << endl; 
        cout << "nOctaveLayers: " << nOctaveLayers << endl; 
    }
    cv::Ptr<cv::Feature2D> detector = 
            cv::xfeatures2d::SIFT::create(0,nOctaveLayers, contrastThreshold);
            //cv::xfeatures2d::SIFT::create(0,5,.002);
    //auto detector = cv::ORB::create(nfeatures, scaleFactor, nOctaveLayers, 31);
    
    detector->detectAndCompute(f1.rgb, cv::Mat(), f1.kp, f1.desp);
    detector->detectAndCompute(f2.rgb, cv::Mat(), f2.kp, f2.desp); 
    //cout << "Detect the keypoint " << endl; 
    
    // find matches
    std::vector<cv::DMatch> matches;

    // flann mather
    cv::Ptr<cv::DescriptorMatcher> matcher = 
      cv::makePtr<cv::FlannBasedMatcher>();
    //std::vector< cv::DMatch > matches;
    matcher->match( f1.desp, f2.desp , matches );

     //-- Quick calculation of max and min distances between keypoints
    double max_dist = 0; double min_dist = 100;
    for( int i = 0; i < f1.desp.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector< cv::DMatch > goodMatches;

      for( int i = 0; i < f1.desp.rows; i++ )
      { if( matches[i].distance <= std::max(10*min_dist, 0.02) )
        { goodMatches.push_back( matches[i]); }
      }
    

    // display good matches
    if (display)
    {
        cout<<"good matches="<<goodMatches.size()<<endl;
    }

    if (imgDisplay)
    {
        cv::Mat imgMatches;
        cv::drawMatches( f1.rgb, f1.kp, f2.rgb, f2.kp, goodMatches, imgMatches );
        cv::imwrite( "good_matches.png", imgMatches );
        cv::imshow( "good matches", imgMatches );
        cv::waitKey(0); 
    }
    

    // 3D poitns
    std::vector<cv::Point3d> src ; 
    std::vector<cv::Point3d> dst; 
    // 2D location
    std::vector<cv::Point2d> imagCor; 

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
    
    cv::Mat rvec, translationVec, inliers, ratationVector;
    cv::Mat affine = cv::Mat::zeros(3,4,CV_64F);
    if (display)
    {
        cout<<"src.size "<<src.size()<<endl;
        cout<<"dst.size "<<dst.size()<<endl;
    }
    
    int half = src.size() * 0.6;
    double threshold = 100.0; 
    
    count = 0; 
    while (count < half)
    {
        threshold += 10;
        //cout << threshold << "  " <<count << endl; 
        cv::estimateAffine3D(src, dst,affine,inliers, threshold ,0.99);
        count = 0; 
        for (int i = 0; i < src.size(); ++i)
        {
            if(inliers.at<bool>(0,i) == true)
            {
                ++count; 
            }
        }
    }
    
    std::cout << "Inliners : " << count << " Total : " << src.size() << std::endl;
    std::cout << "thres hold " << threshold << std::endl; 

    int writeImg = atoi( pd.getData( "writeImg" ).c_str() );
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

    if (writeImg)
    {
        cv::Mat imgMatches;
        std::vector<cv::DMatch> goodMatches2;
        for (int i = 0; i < src.size(); ++i)
        {
            if(inliers.at<bool>(0,i) == true)
            {
                goodMatches2.push_back(goodMatches[i]); 
            }
        }
        cv::drawMatches( f1.rgb, f1.kp, f2.rgb, f2.kp, goodMatches2, imgMatches );
        cv::imwrite( "good_matches.png", imgMatches );
    }

    std::vector<double> t(3); 
    std::vector<double> Rot(3); 

    ResultOfSVD Rt = poseEstimation3D3DReturn(srcSVD, dstSVD, Rot, t);
    Eigen::Isometry3d transform_2 = Eigen::Isometry3d::Identity();
    //Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();

    Eigen::AngleAxisd angle(Rt.R_);
    //cout << angle << endl; 
    //T = angle; 
    
    Eigen::Translation<double,3> trans(Rt.t_(0,0), Rt.t_(1,0), Rt.t_(2,0));
    transform_2(0,3) = Rt.t_(0,0); 
    transform_2(1,3) = Rt.t_(1,0); 
    transform_2(2,3) = Rt.t_(2,0); 
    transform_2(0,0) = Rt.R_(0,0);
    transform_2(0,1) = Rt.R_(0,1);
    transform_2(0,2) = Rt.R_(0,2);
    transform_2(1,0) = Rt.R_(1,0);
    transform_2(1,1) = Rt.R_(1,1);
    transform_2(1,2) = Rt.R_(1,2);
    transform_2(2,0) = Rt.R_(2,0);
    transform_2(2,1) = Rt.R_(2,1);
    transform_2(2,2) = Rt.R_(2,2);

    cout << "Rows " <<transform_2.rows() << " Cols " << transform_2.cols() << endl; 
    //cout << "T " << endl; 
    //cout << T << endl; 

    cout<<"converting image to clouds"<<endl;

    pointCloud::Ptr cloud1 = image2PointCloud(f1,height, width);
    pointCloud::Ptr cloud2 = image2PointCloud(f2,height, width);
    cout << "Saving the sub clould " << endl; 
    pcl::io::savePCDFile("clould1.pcd", *cloud1);
    pcl::io::savePCDFile("clould2.pcd", *cloud2);

    // combine cloulds
    pointCloud::Ptr output (new pointCloud());
    cout<<"combining clouds"<<endl;
    cout << transform_2.matrix() << endl; 

    pcl::transformPointCloud( *cloud1, *output, transform_2.matrix());
    
    cout << "Combine result " << endl; 
    *output += *cloud2;

    pcl::io::savePCDFile("result.pcd", *output);


    return 0; 

}

