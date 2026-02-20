
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
#include <pcl/features/moment_of_inertia_estimation.h>
#include <cuda_runtime.h>
#include "../include/pcl_cuda/test.h"
using namespace std;



/**
 * 随机生成指定数量的点云
 * @param num_points 点数
 * @param range 坐标范围 [-range, range]
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateRandomCloud4(size_t num_points, float range = 100.0f) {
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

void computeOBB_test(size_t numPoints, float range)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud = generateRandomCloud4(numPoints, range);
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


    pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud(cloud);
    auto t1 = std::chrono::high_resolution_clock::now();
    // 创建MomentOfInertiaEstimation对象
    feature_extractor.compute();
    std::cout << "[PCL] end" << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    double pcl_time = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // 变量声明
    pcl::PointXYZ min_point_OBB, max_point_OBB, position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    // 获取OBB
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);


    pcl::cuda::GpuPointCloud cloud_source;
    cloud_source.upload(h_source_x, h_source_y, h_source_z);
    Eigen::Matrix3f covariance_matrix_gpu;

    pcl::cuda::PointXYZ centroid_gpu;
    int test_num = 50;
    double gpu_all_time = 0.0;
    pcl::cuda::PointXYZ min_point_OBB_gpu;
    pcl::cuda::PointXYZ max_point_OBB_gpu;
    pcl::cuda::PointXYZ position_OBB_gpu;
    Eigen::Matrix3f rotational_matrix_OBB_gpu;
    for (int i = 0; i < test_num; i++)
    {
        auto t3 = std::chrono::high_resolution_clock::now();
        pcl::cuda::computeMeanAndCovarianceMatrix(cloud_source, covariance_matrix_gpu, centroid_gpu);
        pcl::cuda::computeOBB(cloud_source, covariance_matrix_gpu, centroid_gpu,
            min_point_OBB_gpu, max_point_OBB_gpu, position_OBB_gpu, rotational_matrix_OBB_gpu);          
        auto t4 = std::chrono::high_resolution_clock::now();
        double gpu_time = std::chrono::duration<double, std::milli>(t4 - t3).count();
        if (i != 0)
        {
            gpu_all_time += gpu_time;
        }

    }






    std::cout << "[PCL] computeOBB Time: " << pcl_time << " ms " << "\n";
    std::cout << "[GPU] computeOBB Time: " << gpu_all_time / (test_num - 1) << " ms " << "\n";

    std::cout << "\n------------------- OBB Result Comparison -------------------" << std::endl;

    // 1. 位置 (Position / Centroid)
    printf("PCL Position:    (%f, %f, %f)\n", position_OBB.x, position_OBB.y, position_OBB.z);
    printf("GPU Position:    (%f, %f, %f)\n", position_OBB_gpu.x, position_OBB_gpu.y, position_OBB_gpu.z);
    std::cout << std::endl;

    // 2. 最小/最大点 (Min/Max Point)
    printf("PCL Min Point:   (%f, %f, %f)\n", min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
    printf("GPU Min Point:   (%f, %f, %f)\n", min_point_OBB_gpu.x, min_point_OBB_gpu.y, min_point_OBB_gpu.z);
    std::cout << std::endl;

    printf("PCL Max Point:   (%f, %f, %f)\n", max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
    printf("GPU Max Point:   (%f, %f, %f)\n", max_point_OBB_gpu.x, max_point_OBB_gpu.y, max_point_OBB_gpu.z);
    std::cout << std::endl;

    // 3. 旋转矩阵 (Rotation Matrix)
    std::cout << "PCL Rotation Matrix:" << std::endl << rotational_matrix_OBB << std::endl;
    std::cout << "GPU Rotation Matrix:" << std::endl << rotational_matrix_OBB_gpu << std::endl;

    // =============================================================
    // 结果验证
    // =============================================================
    std::cout << "--- Comparison ---\n";
    std::cout << "Speedup: " << pcl_time / gpu_all_time * (test_num - 1) << "x\n";
}

