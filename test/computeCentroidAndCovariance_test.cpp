
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
pcl::PointCloud<pcl::PointXYZ>::Ptr generateRandomCloud3(size_t num_points, float range = 100.0f) {
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

void computeCentroidAndCovariance_test(size_t numPoints, float range)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud = generateRandomCloud3(numPoints, range);
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



    auto t1 = std::chrono::high_resolution_clock::now();
    // 计算点云的质心
    Eigen::Vector4f centroid;
    Eigen::Matrix3f covariance_matrix;
    pcl::computeMeanAndCovarianceMatrix(*cloud, covariance_matrix, centroid);
    std::cout << "[PCL] end" << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    double pcl_time = std::chrono::duration<double, std::milli>(t2 - t1).count();


    pcl::cuda::GpuPointCloud cloud_source;
    cloud_source.upload(h_source_x, h_source_y, h_source_z);

    Eigen::Matrix3f covariance_matrix_gpu;

    pcl::cuda::PointXYZ centroid_gpu;
    int test_num = 50;
    double gpu_all_time = 0.0;
    for (int i = 0; i < test_num; i++)
    {
        auto t3 = std::chrono::high_resolution_clock::now();
        pcl::cuda::computeMeanAndCovarianceMatrix(cloud_source, covariance_matrix_gpu, centroid_gpu);
        auto t4 = std::chrono::high_resolution_clock::now();
        double gpu_time = std::chrono::duration<double, std::milli>(t4 - t3).count();
        if (i != 0)
        {
            gpu_all_time += gpu_time;
        }

    }



    std::cout << "[PCL] computeMeanAndCovarianceMatrix Time: " << pcl_time << " ms " << "\n";
    std::cout << "[GPU] computeMeanAndCovarianceMatrix Time: " << gpu_all_time / (test_num - 1) << " ms " << "\n";

    // 输出质心
    std::cout << "PCL-covariance_matrix:"
        << covariance_matrix << "\n "
        << "PCL-centroid:"
        << centroid[0] << ", "
        << centroid[1] << ", "
        << centroid[2] << std::endl;
    std::cout << "cuPCL-covariance_matrix:"
        << covariance_matrix_gpu << "\n "
        << "cuPCL-centroid: "
        << centroid_gpu.x << ", "
        << centroid_gpu.y << ", "
        << centroid_gpu.z << std::endl;


    // =============================================================
    // 结果验证
    // =============================================================
    std::cout << "--- Comparison ---\n";
    std::cout << "Speedup: " << pcl_time / gpu_all_time * (test_num - 1) << "x\n";
}

