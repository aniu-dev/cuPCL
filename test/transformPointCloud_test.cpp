
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
#include <cuda_runtime.h>
#include "../include/pcl_cuda/test.h"
using namespace std;



/**
 * 随机生成指定数量的点云
 * @param num_points 点数
 * @param range 坐标范围 [-range, range]
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateRandomCloud6(size_t num_points, float range = 100.0f) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 1. 预分配内存
    cloud->width = num_points;
    cloud->height = 1;
    cloud->is_dense = true;
    cloud->points.resize(num_points);

    // 2. 使用现代 C++ 随机数生成器 (Mersenne Twister)
    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dist(-range, range);

    std::cout << "Generating " << num_points << " points..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // 3. 填充点
    for (size_t i = 0; i < num_points; ++i) {
        cloud->points[i].x = dist(gen);
        cloud->points[i].y = dist(gen);
        cloud->points[i].z = dist(gen);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    std::cout << "Done! Time: " << diff.count() << " ms" << std::endl;

    return cloud;
}

void transformPointCloud_test(size_t numPoints, float range)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud = generateRandomCloud6(numPoints, range);
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

    // 定义旋转矩阵（例如，绕Z轴旋转45度）
    float theta = M_PI / 4; // 45度
    Eigen::Matrix3f rotation;
    rotation = Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ());

    // 定义平移向量
    Eigen::Vector3f translation(0.1, 0.2, 0.3);

    // 创建4x4变换矩阵
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = rotation;
    transform.block<3, 1>(0, 3) = translation;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
    auto t1 = std::chrono::high_resolution_clock::now();

    pcl::transformPointCloud(*cloud, *cloud_out, transform);

    std::cout << "[PCL] end" << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    double pcl_time = std::chrono::duration<double, std::milli>(t2 - t1).count();


    pcl::cuda::GpuPointCloud cloud_source;
    cloud_source.upload(h_source_x, h_source_y, h_source_z);

    pcl::cuda::GpuPointCloud cloud_out_gpu;

    int test_num = 50;
    double gpu_all_time = 0.0;
    for (int i = 0; i < test_num; i++)
    {
        auto t3 = std::chrono::high_resolution_clock::now();
        pcl::cuda::transformPointCloud(cloud_source, cloud_out_gpu, transform);
        auto t4 = std::chrono::high_resolution_clock::now();
        double gpu_time = std::chrono::duration<double, std::milli>(t4 - t3).count();
        if (i != 0)
        {
            gpu_all_time += gpu_time;
        }

    }



    std::cout << "[PCL] transformPointCloud Time: " << pcl_time << " ms " << "\n";
    std::cout << "[GPU] transformPointCloud Time: " << gpu_all_time / (test_num - 1) << " ms " << "\n";

   
   
    std::vector<float> gpu_out_x(NUM_POINTS_SOURCE);
    std::vector<float> gpu_out_y(NUM_POINTS_SOURCE);
    std::vector<float> gpu_out_z(NUM_POINTS_SOURCE);
    cloud_out_gpu.download(gpu_out_x, gpu_out_y, gpu_out_z);

    std::cout << "\n" << std::string(85, '=') << "\n";
    printf("%-5s | %-25s | %-25s | %-10s\n", "IDX", "PCL (x, y, z)", "cuPCL (x, y, z)", "Status");
    std::cout << std::string(85, '-') << "\n";

    int print_count = std::min(NUM_POINTS_SOURCE, 10);
    for (int i = 0; i < print_count; ++i) {
        // PCL 坐标
        float px = cloud_out->points[i].x;
        float py = cloud_out->points[i].y;
        float pz = cloud_out->points[i].z;

        // GPU 坐标
        float gx = gpu_out_x[i];
        float gy = gpu_out_y[i];
        float gz = gpu_out_z[i];

        // 计算简单误差判断
        float diff = std::abs(px - gx) + std::abs(py - gy) + std::abs(pz - gz);
        string status = (diff < 1e-3) ? "OK" : "DIFF";

        printf("%-5d | %7.3f, %7.3f, %7.3f | %7.3f, %7.3f, %7.3f | %-10s\n",
            i, px, py, pz, gx, gy, gz, status.c_str());
    }
    std::cout << std::string(85, '=') << "\n";


    // =============================================================
    // 结果验证
    // =============================================================
    std::cout << "--- Comparison ---\n";
    std::cout << "Speedup: " << pcl_time / gpu_all_time * (test_num - 1) << "x\n";
}

