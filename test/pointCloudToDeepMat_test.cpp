#include <iostream>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/common.h>
#include <pcl/common/time.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/sample_consensus/model_types.h> // 必须加这个！否则 SACMODEL_PLANE 报错
#include <pcl/filters/voxel_grid.h>
#include <execution> 
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "../test.h"
using namespace std;



/**
 * 极简版生成器
 * num_points: 总点数 (如 1000000)
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateRansacPlaneData2(size_t num_points) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->resize(num_points);
    cloud->width = (uint32_t)num_points;
    cloud->height = 1;
    cloud->is_dense = true;

    // 1. 设置随机数引擎
    std::mt19937 gen(12345); // 固定种子，保证每次生成的数据一样，方便调试
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f); // 坐标范围
    std::normal_distribution<float> noise(0.0f, 0.10f);       // 1.0的微小噪声

    // 2. 确定内点数量
    size_t num_inliers = (size_t)(num_points * 0.7f);

    // 3. 一个循环搞定所有
    for (size_t i = 0; i < num_points; ++i) {
        float x = dist(gen);
        float y = dist(gen);

        if (i < num_inliers) {
            // 生成平面点: 假设平面方程是 z = 0.5x + 0.8y + 10
            float z = x + y - 10.0f + noise(gen);
            cloud->points[i].x = x;
            cloud->points[i].y = y;
            cloud->points[i].z = z;
        }
        else {
            cloud->points[i] = { dist(gen), dist(gen), dist(gen) };
        }
    }

    std::cout << "Successfully generated " << num_points << " points." << std::endl;
    return cloud;
}

void pclPointCloud2Deep(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int width, int height,
    cv::Mat &depth_image, float resolution, float min_x, float min_y)
{
    // 创建深度图像
    depth_image = cv::Mat::zeros(height, width, CV_32FC1);

    // 填充深度图像
    for (const auto& point : cloud->points) {
        int u = std::floor((point.x - min_x) / resolution);
        int v = std::floor((point.y - min_y) / resolution);
        float depth_value = point.z;
        float current_z = depth_image.at<float>(v, u);
            
        if (depth_value > current_z) {
            depth_image.at<float>(v, u) = depth_value;
        }

        
    }
}



void pointCloudToDeepMat_test(size_t numPoints)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud = generateRansacPlaneData2(numPoints);
    int NUM_POINTS_SOURCE = cloud->points.size();
    std::cout << "NUM_POINTS SOURCE: " << NUM_POINTS_SOURCE << "\n\n";
    std::vector<float> h_source_x(NUM_POINTS_SOURCE);
    std::vector<float> h_source_y(NUM_POINTS_SOURCE);
    std::vector<float> h_source_z(NUM_POINTS_SOURCE);
    for (int i = 0; i < NUM_POINTS_SOURCE; ++i) {

        // 填充 SOA
        h_source_x[i] = cloud->points[i].x;
        h_source_y[i] = cloud->points[i].y;
        h_source_z[i] = cloud->points[i].z;
    }
    cv::Mat depth_image;
    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*cloud, minPt, maxPt);
    // 设置分辨率
    float resolution = 0.1f; 

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
    auto t1 = std::chrono::high_resolution_clock::now();


    // 计算图像大小
    int width = static_cast<int>((maxPt.x - minPt.x) / resolution) + 1;
    int height = static_cast<int>((maxPt.y - minPt.y) / resolution) + 1;
    pclPointCloud2Deep(cloud, width, height,depth_image, resolution,minPt.x,minPt.y);
    std::cout << "[PCL] end" << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    double pcl_time = std::chrono::duration<double, std::milli>(t2 - t1).count();


    pcl::cuda::GpuPointCloud cloud_source;
    cloud_source.upload(h_source_x, h_source_y, h_source_z);
  
    int test_num = 50;
    double gpu_all_time = 0.0;
    float* mat_out = new float[width * height];
    //cv::Mat depth_image_gpu = cv::Mat::zeros(height, width, CV_32FC1);
    for (int i = 0; i < test_num; i++)
    {
        auto t3 = std::chrono::high_resolution_clock::now();
        pcl::cuda::pointCloudToDeepMat(cloud_source, width, height, mat_out, resolution, minPt.x, minPt.y);
        //depth_image_gpu
        auto t4 = std::chrono::high_resolution_clock::now();
        double gpu_time = std::chrono::duration<double, std::milli>(t4 - t3).count();
        if (i != 0)
        {
            gpu_all_time += gpu_time;
        }

    }
    cv::Mat depth_image_gpu = cv::Mat(height, width, CV_32FC1, mat_out);


    std::cout << "[PCL] pointCloudToDeepMat Time: " << pcl_time << " ms " << "\n";
    std::cout << "[GPU] pointCloudToDeepMat Time: " << gpu_all_time / (test_num - 1) << " ms " << "\n";




    // =============================================================
    // 结果验证
    // =============================================================
    std::cout << "--- Comparison ---\n";
    std::cout << "Speedup: " << pcl_time / gpu_all_time * (test_num - 1) << "x\n";
}

