
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
pcl::PointCloud<pcl::PointXYZ>::Ptr generateRandomCloudICP(size_t num_points, float range = 100.0f) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->resize(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        if (i < num_points / 3) { // X轴上的线
            cloud->points[i] = { (float)i / 100.0f, 0, 0 };
        }
        else if (i < 2 * num_points / 3) { // Y轴上的线
            cloud->points[i] = { 0, (float)(i - num_points / 3) / 100.0f, 0 };
        }
        else { // Z轴上的线
            cloud->points[i] = { 0, 0, (float)(i - 2 * num_points / 3) / 100.0f };
        }
    }
    return cloud;
}

void ICP_test(size_t numPoints, float range)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud = generateRandomCloudICP(numPoints, range);

    // 2. 构造一个已知的旋转平移矩阵 (Ground Truth)
    Eigen::Matrix4f T_gt = Eigen::Matrix4f::Identity();
    float angle = 4.0f * M_PI / 180.0f; // 旋转4度
    T_gt.block<3, 3>(0, 0) = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    T_gt(0, 3) = 0.5f; // X偏移0.5米
    T_gt(1, 3) = -0.2f; // Y偏移-0.2米
    T_gt(2, 3) = 0.1f;  // Z偏移0.1米

    // 3. 生成目标点云 (Target = T_gt * Source + Noise)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *cloud_target, T_gt);
    // 给目标点云加一点微小的噪声，模拟真实场景
    std::mt19937 gen(123);
    std::normal_distribution<float> noise(0.0f, 0.001f);
    for (auto& p : cloud_target->points) {
        p.x += noise(gen); p.y += noise(gen); p.z += noise(gen);
    }

    int NUM_POINTS_SOURCE = cloud->points.size();
    std::cout << "NUM_POINTS SOURCE: " << NUM_POINTS_SOURCE << "\n\n";
    std::vector<float> h_source_x(NUM_POINTS_SOURCE);
    std::vector<float> h_source_y(NUM_POINTS_SOURCE);
    std::vector<float> h_source_z(NUM_POINTS_SOURCE);
    std::vector<float> h_target_x(NUM_POINTS_SOURCE);
    std::vector<float> h_target_y(NUM_POINTS_SOURCE);
    std::vector<float> h_target_z(NUM_POINTS_SOURCE);
    for (int i = 0; i < NUM_POINTS_SOURCE; ++i) {

        // 填充 SOA
        h_source_x[i] = cloud->points[i].x;
        h_source_y[i] = cloud->points[i].y;
        h_source_z[i] = cloud->points[i].z;

        h_target_x[i] = cloud_target->points[i].x;
        h_target_y[i] = cloud_target->points[i].y;
        h_target_z[i] = cloud_target->points[i].z;
    }




    // 创建ICP对象
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud);
    icp.setInputTarget(cloud_target);

    // 设置ICP参数
    icp.setMaximumIterations(50);//最大迭代次数
    icp.setTransformationEpsilon(1e-10);//设置最小转换差异
    icp.setEuclideanFitnessEpsilon(1e-6);//设置收敛条件的均方差误差阈值
    icp.setMaxCorrespondenceDistance(5.0);//设置对应点的最大距离


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
    auto t1 = std::chrono::high_resolution_clock::now();
    icp.align(*cloud_out);
    std::cout << "[PCL] end" << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    double pcl_time = std::chrono::duration<double, std::milli>(t2 - t1).count();


    pcl::cuda::GpuPointCloud cloud_source_gpu,cloud_target_gpu;
    cloud_source_gpu.upload(h_source_x, h_source_y, h_source_z);
    cloud_target_gpu.upload(h_target_x,h_target_y,h_target_z);
    pcl::cuda::IterativeClosestPoint icp_gpu;

    icp_gpu.setInputSource(cloud_source_gpu);
    icp_gpu.setInputTarget(cloud_target_gpu);

    // 设置ICP参数
    icp_gpu.setMaximumIterations(50);//最大迭代次数
    icp_gpu.setTransformationEpsilon(1e-10);//设置最小转换差异
    icp_gpu.setEuclideanFitnessEpsilon(1e-6);//设置收敛条件的均方差误差阈值
    icp_gpu.setMaxCorrespondenceDistance(5.0);//设置对应点的最大距离

    pcl::cuda::GpuPointCloud cloud_out_gpu;
    pcl::cuda::PointXYZ centroid_gpu;
    int test_num = 50;
    double gpu_all_time = 0.0;
    for (int i = 0; i < test_num; i++)
    {
        auto t3 = std::chrono::high_resolution_clock::now();
        icp_gpu.align(cloud_out_gpu);
        auto t4 = std::chrono::high_resolution_clock::now();
        double gpu_time = std::chrono::duration<double, std::milli>(t4 - t3).count();
        if (i != 0)
        {
            gpu_all_time += gpu_time;
        }

    }

    // 1. 将 GPU 结果下载到 Host
    cout << "真实 变换矩阵：" << T_gt << endl;
    cout << "PCL 变换矩阵：" << icp.getFinalTransformation() << endl;
    cout << "cuPCL 变换矩阵：" << icp_gpu.getFinalTransformation() << endl;


    std::cout << "--------------------------------------------------------------------------------------------------\n";
    std::cout << "[PCL] IterativeClosestPoint Time: " << pcl_time << " ms " << "\n";
    std::cout << "[GPU] IterativeClosestPoint Time: " << gpu_all_time / (test_num - 1) << " ms " << "\n";
    std::cout << "Speedup: " << pcl_time / (gpu_all_time / (test_num - 1)) << "x\n";
}

