#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>


// Reusable objects
ros::Publisher detect_bush_pub;
ros::Subscriber depth_sub;
pcl::VoxelGrid<pcl::PCLPointCloud2> vox;
pcl::PassThrough<pcl::PointXYZ> pass;
pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
pcl::SACSegmentation<pcl::PointXYZ> seg;
pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;







void depthPointsCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {

    // Instantiate various clouds
    pcl::PCLPointCloud2* cloud_intermediate = new pcl::PCLPointCloud2;
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud_intermediate);
    pcl::PointCloud<pcl::PointXYZ> cloud;

    // Convert to PCL data type
    pcl_conversions::toPCL(*cloud_msg, *cloud_intermediate);

    // Apply Voxel Filter on PCLPointCloud2
    vox.setInputCloud (cloudPtr);
    vox.setInputCloud (cloudPtr);
    vox.filter (*cloud_intermediate);

   

    // Convert PCL::PointCloud2 to PCL::PointCloud<PointXYZ>
    pcl::fromPCLPointCloud2(*cloud_intermediate, cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p = cloud.makeShared();


    // Apply Passthrough Filter
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (0.2, 3);
    pass.setInputCloud (cloud_p);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.0, 3.0);
    //pass.setFilterLimitsNegative (true);
    pass.filter (*cloud_p);

    // Apply Statistical Noise Filter
    sor.setInputCloud (cloud_p);
    sor.filter (*cloud_p);

    // Planar segmentation: Remove large planes? Or extract floor?
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

    int nr_points = (int) cloud_p->points.size ();


    // Set the axis for which to search for planes, in this case every axis
    Eigen::Vector3f axis (1, 1, 1);
    // Set a allowed deviation angle from vector axis
    seg.setEpsAngle(  30.0f * (3.14f/180.0f) );
    seg.setAxis(axis);
    //while(cloud_p->points.size () > 0.2 * nr_points) {
    sor.setInputCloud (cloud_p);
    sor.filter (*cloud_p);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);
    // Segment the planes and insert the coefficients and inliers in vectors
    seg.setInputCloud (cloud_p);
    seg.segment (*inliers, *coefficients);

    // check that we found a plane from cloud
    if (inliers->indices.size () == 0)
    {
        std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
        //break;
    }
    else {
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_p);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud_p);
    }
    //}
    // Set the axis for which to search for cluster
    Eigen::Vector3f axis_p (0.5f, 0, 0.5f);
    seg.setAxis(axis_p);
    while(cloud_p->points.size () > 0.1 * nr_points) {

        seg.setInputCloud (cloud_p);
        seg.segment (*inliers, *coefficients);

        if (inliers->indices.size () == 0)
        {
          std::cerr << "Could not estimate a bush model for the given dataset." << std::endl;
            //break;
        }
        else {
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(cloud_p);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*cloud_p);
        }
    }

    // Apply Statistical Noise Filter
    sor.setInputCloud (cloud_p);
    sor.filter (*cloud_p);

    //EuclideanClusterExtraction functions best with octree data structure fetches 
    if(cloud_p->points.size() > 40) {
        std::vector<pcl::PointIndices> cluster_indices;
        tree->setInputCloud (cloud_p);
        ec.setInputCloud (cloud_p);
        ec.extract (cluster_indices);

        std::cout << "Bush-like object found: " << cluster_indices.size() << "\n";
        // Convert to ROS data type
    }
    // Convert to ROS data type
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud_p, output);


    // Publish the data
    detect_bush_pub.publish(output);

}

int main(int argc, char** argv) {
    ros::init(argc, argv, "detect_bush");
    ros::NodeHandle n;
    detect_bush_pub = n.advertise<sensor_msgs::PointCloud2>("bush_detected", 1);
    depth_sub = n.subscribe("/camera/depth/color/points", 1, depthPointsCallback);
    ros::Rate loop_rate(10);

    // Set variables for filters

    vox.setLeafSize (0.01f, 0.01f, 0.01f);

    sor.setMeanK (10);
    sor.setStddevMulThresh (0.1);

    ec.setClusterTolerance (0.01); // 1cm
    ec.setMinClusterSize (100); // min number of points cluser needs to be valid
    ec.setMaxClusterSize (nr_points + 1); // max number of points cluser needs to be valid
    ec.setSearchMethod (tree);

    //seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.02);

    ros::Rate r(25);

    while (ros::ok())
    {
      ros::spinOnce();
      r.sleep();
    }
}
