
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
pcl::PointCloud<pcl::PointXYZ>::Ptr generateRandomCloudEC(size_t num_points) {
    const int num_clusters = 5;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->reserve(num_points);

    // 1. 随机分配每个簇的点数，确保总和为 1,0000,000
    std::vector<int> cluster_sizes;
    int remaining_points = num_points;
    std::default_random_engine generator;

    for (int i = 0; i < num_clusters - 1; ++i) {
        // 确保后面剩下的簇至少能分到 100,000 点
        int min_val = num_points / 10;
        int max_val = std::min(min_val * 3, remaining_points - (num_clusters - 1 - i) * min_val);
        std::uniform_int_distribution<int> distribution(min_val, max_val);
        int size = distribution(generator);
        cluster_sizes.push_back(size);
        remaining_points -= size;
    }
    cluster_sizes.push_back(remaining_points); // 最后一个簇拿走剩余所有

    // 2. 定义 5 个簇的中心点，确保它们相距足够远（欧式聚类好区分）
    std::vector<pcl::PointXYZ> centers = {
        {0.0f,   0.0f,   0.0f},
        {50.0f,  0.0f,   0.0f},
        {0.0f,   50.0f,  0.0f},
        {50.0f,  50.0f,  0.0f},
        {25.0f,   25.0f, 50.0f}
    };

    // 3. 为每个簇生成高斯分布的点
    std::normal_distribution<float> dist_noise(0.0f, 2.0f); // 标准差为 2.0

    for (int i = 0; i < num_clusters; ++i) {
        std::cout << "Generating Cluster " << i << " with " << cluster_sizes[i] << " points..." << std::endl;

        for (int j = 0; j < cluster_sizes[i]; ++j) {
            pcl::PointXYZ p;
            p.x = centers[i].x + dist_noise(generator);
            p.y = centers[i].y + dist_noise(generator);
            p.z = centers[i].z + dist_noise(generator);
            cloud->push_back(p);
        }
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}


void EuclideanClusterExtraction_test(size_t numPoints)
{



    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud = generateRandomCloudEC(numPoints);

    pcl::io::savePCDFileBinary("cloud.pcd",*cloud);

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

    // 创建Kd树用于邻域搜索
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);


    int min_num = numPoints / 10;
    int max_num = min_num * 3;
    // 设置欧式聚类参数
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.5);  // 设置邻域范围的距离阈值
    ec.setMinClusterSize(min_num);    // 设置最小簇的大小
    ec.setMaxClusterSize(max_num);  // 设置最大簇的大小
    ec.setSearchMethod(tree);     // 设置搜索方法
    ec.setInputCloud(cloud);   // 设置输入点云

    auto t1 = std::chrono::high_resolution_clock::now();
    ec.extract(cluster_indices);  // 执行聚类提取  
    std::cout << "[PCL] end" << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    double pcl_time = std::chrono::duration<double, std::milli>(t2 - t1).count();


    pcl::cuda::GpuPointCloud cloud_source;
    cloud_source.upload(h_source_x, h_source_y, h_source_z);


    pcl::cuda::EuclideanClusterExtraction nes;
    nes.setInputCloud(cloud_source);
    nes.setClusterTolerance(0.5);  // 设置邻域范围的距离阈值
    nes.setMinClusterSize(min_num);    // 设置最小簇的大小
    nes.setMaxClusterSize(max_num);  // 设置最大簇的大小

    std::vector<std::vector<int>> clusters;
    int test_num = 50;
    double gpu_all_time = 0.0;
    for (int i = 0; i < test_num; i++)
    {
        auto t3 = std::chrono::high_resolution_clock::now();
        clusters.clear();
        nes.extract(clusters);
        auto t4 = std::chrono::high_resolution_clock::now();
        double gpu_time = std::chrono::duration<double, std::milli>(t4 - t3).count();
        if (i != 0)
        {
            gpu_all_time += gpu_time;
        }

    }
    // =============================================================
    // 结果验证
    // =============================================================

    std::sort(cluster_indices.begin(), cluster_indices.end(), [](const pcl::PointIndices& a, const pcl::PointIndices& b) {
        if (a.indices.size() != b.indices.size()) return a.indices.size() < b.indices.size();
        return a.indices[0] < b.indices[0];
        });
    for (int i = 0; i < cluster_indices.size(); i++)
    {
        std::cout << "CPU_cluster_indices_" + to_string(i) + ": " << cluster_indices[i].indices.size() << std::endl;
    }
    std::sort(clusters.begin(), clusters.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
        if (a.size() != b.size()) return a.size() < b.size(); // 先按大小排
        return a[0] < b[0];                                  // 大小相同按第一个索引排
        });
    for (int i = 0; i < clusters.size(); i++)
    {
        std::cout << "GPU_cluster_indices_" + to_string(i) + ": " << clusters[i].size() << std::endl;
    }

    std::cout << "[PCL] EC Time: " << pcl_time << " ms " << "\n";
    std::cout << "[GPU] EC Time: " << gpu_all_time / (test_num - 1) << " ms " << "\n";
    std::cout << "--- Comparison ---\n";
    std::cout << "Speedup: " << pcl_time / gpu_all_time * (test_num - 1) << "x\n";
}