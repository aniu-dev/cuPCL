
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
#include "../test.h"
using namespace std;



/**
 * 随机生成指定数量的点云
 * @param num_points 点数
 * @param range 坐标范围 [-range, range]
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateRandomCpy(size_t num_points, float range = 100.0f) {
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
// 生成 80% 的随机索引
std::vector<int> generateRandomIndices(size_t source_size, float ratio = 0.8f) {
    size_t num_indices = static_cast<size_t>(source_size * ratio);
    std::vector<int> indices(num_indices);

    // 生成顺序索引
    std::vector<int> all_indices(source_size);
    for (size_t i = 0; i < source_size; ++i) all_indices[i] = i;

    // 洗牌并取前 80%
    std::mt19937 gen(456);
    std::uniform_int_distribution<int> dist(0, source_size - 1);
    for (size_t i = 0; i < num_indices; ++i) {
        indices[i] = dist(gen);
    }
    return indices;
}
void copyPointCloud_test(size_t numPoints, float range)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud = generateRandomCpy(numPoints, range);
    auto h_indices = generateRandomIndices(numPoints, 0.8f);
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


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);

    auto t1 = std::chrono::high_resolution_clock::now();
    pcl::copyPointCloud(*cloud, h_indices, *cloud_out);
    auto t2 = std::chrono::high_resolution_clock::now();
    double pcl_time = std::chrono::duration<double, std::milli>(t2 - t1).count();

 

    pcl::cuda::GpuPointCloud cloud_source, cloud_out_gpu;
    cloud_source.upload(h_source_x, h_source_y, h_source_z);
    Eigen::Matrix3f covariance_matrix_gpu;

    int test_num = 50;
    double gpu_all_time = 0.0;

    for (int i = 0; i < test_num; i++)
    {
        auto t3 = std::chrono::high_resolution_clock::now();
        pcl::cuda::copyPointCloud(cloud_source, cloud_out_gpu, h_indices,false);
        auto t4 = std::chrono::high_resolution_clock::now();
        double gpu_time = std::chrono::duration<double, std::milli>(t4 - t3).count();
        if (i != 0)
        {
            gpu_all_time += gpu_time;
        }

    }






    std::cout << "[PCL] copyPointCloud Time: " << pcl_time << " ms " << "\n";
    std::cout << "[GPU] copyPointCloud Time: " << gpu_all_time / (test_num - 1) << " ms " << "\n";
    std::cout << "--- Comparison ---\n";
    std::cout << "Speedup: " << pcl_time / gpu_all_time * (test_num - 1) << "x\n";
}

