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
#include "../test.h"
using namespace std;



/**
 * 极简版生成器
 * num_points: 总点数 (如 1000000)
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateRansacSphereData(size_t num_points) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->resize(num_points);
    cloud->width = (uint32_t)num_points;
    cloud->height = 1;
    cloud->is_dense = true;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::uniform_real_distribution<float> phi_dist(0, 2.0f * M_PI);
    std::uniform_real_distribution<float> theta_dist(0, M_PI);
    std::normal_distribution<float> noise(0.0f, 0.10f);

    float center_x = 0, center_y = 0, center_z = 0, radius = 20.0f;
    size_t num_inliers = (size_t)(num_points * 0.7f);

    for (size_t i = 0; i < num_points; ++i) {
        if (i < num_inliers) {
            float phi = phi_dist(gen);
            float theta = theta_dist(gen);
            float r = radius + noise(gen); // 噪声加在半径上
            cloud->points[i].x = center_x + r * sin(theta) * cos(phi);
            cloud->points[i].y = center_y + r * sin(theta) * sin(phi);
            cloud->points[i].z = center_z + r * cos(theta);
        }
        else {
            cloud->points[i] = { dist(gen), dist(gen), dist(gen) };
        }
    }
    return cloud;
}
void ransacSphere_test(size_t numPoints)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud = generateRansacSphereData(numPoints);
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
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    coefficients->values.resize(4);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);  //inliers内点的索引
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_SPHERE);   //设置拟合模型参数
    seg.setMethodType(pcl::SAC_RANSAC);     //拟合方法：随机采样法
    seg.setDistanceThreshold(0.05);         //设置点到直线距离阈值
    seg.setMaxIterations(1024);              //最大迭代次数
    seg.setInputCloud(cloud);               //输入点云



    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
    auto t1 = std::chrono::high_resolution_clock::now();

    seg.segment(*inliers, *coefficients);   //拟合点云
    pcl::copyPointCloud(*cloud, inliers->indices, *cloud_out);
    std::cout << "[PCL] end" << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    double pcl_time = std::chrono::duration<double, std::milli>(t2 - t1).count();


    pcl::cuda::GpuPointCloud cloud_source;
    cloud_source.upload(h_source_x, h_source_y, h_source_z);
    pcl::cuda::Ransac ransac;
    ransac.setInput(&cloud_source);
    ransac.setDistanceThreshold(0.05);
    ransac.setMaxIterations(1024);
    ransac.setModelType(pcl::cuda::SACMODEL_SPHERE);

    pcl::cuda::ModelCoefficients model_coeffs;
    pcl::cuda::GpuPointCloud cloud_out_gpu;

    int test_num = 50;
    double gpu_all_time = 0.0;
    for (int i = 0; i < test_num; i++)
    {
        auto t3 = std::chrono::high_resolution_clock::now();
        ransac.segment(cloud_out_gpu, model_coeffs);
        auto t4 = std::chrono::high_resolution_clock::now();
        double gpu_time = std::chrono::duration<double, std::milli>(t4 - t3).count();
        if (i != 0)
        {
            gpu_all_time += gpu_time;
        }

    }



    std::cout << "[PCL] ransacSphere Time: " << pcl_time << " ms " << "\n";
    std::cout << "[GPU] ransacSphere Time: " << gpu_all_time / (test_num - 1) << " ms " << "\n";

    //------------------模型系数---------------------
    cout << "PCL 拟合球面的模型系数为：" << endl;
    cout << "a：" << coefficients->values[0] << endl;
    cout << "b：" << coefficients->values[1] << endl;
    cout << "c：" << coefficients->values[2] << endl;
    cout << "r：" << coefficients->values[3] << endl;

    cout << "PCL Point Nums :" << cloud_out->points.size() << endl;


    cout << "cuPCL 拟合球面的模型系数为：" << endl;
    cout << "a：" << model_coeffs.values[0] << endl;
    cout << "b：" << model_coeffs.values[1] << endl;
    cout << "c：" << model_coeffs.values[2] << endl;
    cout << "r：" << model_coeffs.values[3] << endl;
    cout << "cuPCL Point Nums :" << cloud_out_gpu.size() << endl;





    // =============================================================
    // 结果验证
    // =============================================================
    std::cout << "--- Comparison ---\n";
    std::cout << "Speedup: " << pcl_time / gpu_all_time * (test_num - 1) << "x\n";
}

