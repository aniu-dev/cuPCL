
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
pcl::PointCloud<pcl::PointXYZ>::Ptr generateRandomCloud5(size_t num_points, float range = 100.0f) {
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

void PassThrough_test(size_t numPoints, float range)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud = generateRandomCloud5(numPoints, range);
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
    // 创建滤波器对象
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z"); //设置为Z轴方向滤波
    pass.setFilterLimits(25, 75); //设置滤波方向上的过滤范围
    pass.setNegative(false); //标志为false时保留范围内的点

    auto t1 = std::chrono::high_resolution_clock::now();
    pass.filter(*cloud_out);  
    std::cout << "[PCL] end" << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    double pcl_time = std::chrono::duration<double, std::milli>(t2 - t1).count();

    pcl::cuda::GpuPointCloud cloud_source;
    cloud_source.upload(h_source_x, h_source_y, h_source_z);
    pcl::cuda::GpuPointCloud cloud_out_gpu;

    pcl::cuda::PassThrough pass_gpu;
    pass_gpu.setInputCloud(&cloud_source);
    pass_gpu.setFilterFieldName("z"); //设置为Z轴方向滤波
    pass_gpu.setFilterLimits(25, 75); //设置滤波方向上的过滤范围
    pass_gpu.setNegative(false); //标志为false时保留范围内的点

    int test_num = 50;
    double gpu_all_time = 0.0;
    
    for (int i = 0; i < test_num; i++)
    {
        auto t3 = std::chrono::high_resolution_clock::now();
        pass_gpu.filter(cloud_out_gpu);
        auto t4 = std::chrono::high_resolution_clock::now();
        double gpu_time = std::chrono::duration<double, std::milli>(t4 - t3).count();
        if (i != 0)
        {
            gpu_all_time += gpu_time;
        }

    }
    std::cout << "[PCL] PassThrough Time: " << pcl_time << " ms " << "\n";
    std::cout << "[GPU] PassThrough Time: " << gpu_all_time / (test_num - 1) << " ms " << "\n";
    // 3. 旋转矩阵 (Rotation Matrix)
    std::cout << "PCL Points Nums:" << cloud_out->points.size() << std::endl;
    std::cout << "GPU Points Nums:" << cloud_out_gpu.size() << std::endl;

    // =============================================================
    // 结果验证
    // =============================================================
    std::cout << "--- Comparison ---\n";
    std::cout << "Speedup: " << pcl_time / gpu_all_time * (test_num - 1) << "x\n";
}

