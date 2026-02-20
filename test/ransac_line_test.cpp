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
pcl::PointCloud<pcl::PointXYZ>::Ptr generateRansacLineData(size_t num_points) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->resize(num_points);
    cloud->width = (uint32_t)num_points;
    cloud->height = 1;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::uniform_real_distribution<float> t_dist(-30.0f, 30.0f); // 直线长度范围
    std::normal_distribution<float> noise(0.0f, 0.10f);

    // 直线：过 (0,0,0) 方向 (1,1,1)
    float dir_x = 1.0f, dir_y = 1.0f, dir_z = 1.0f;
    float len = sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
    dir_x /= len; dir_y /= len; dir_z /= len;

    size_t num_inliers = (size_t)(num_points * 0.7f);

    for (size_t i = 0; i < num_points; ++i) {
        if (i < num_inliers) {
            float t = t_dist(gen);
            cloud->points[i].x = t * dir_x + noise(gen);
            cloud->points[i].y = t * dir_y + noise(gen);
            cloud->points[i].z = t * dir_z + noise(gen);
        }
        else {
            cloud->points[i] = { dist(gen), dist(gen), dist(gen) };
        }
    }
    return cloud;
}
void ransacLine_test(size_t numPoints)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud = generateRansacLineData(numPoints);
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
    coefficients->values.resize(6);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);  //inliers内点的索引
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);   //设置拟合模型参数
    seg.setMethodType(pcl::SAC_RANSAC);     //拟合方法：随机采样法
    seg.setDistanceThreshold(0.1);         //设置点到直线距离阈值
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
    ransac.setDistanceThreshold(0.1);
    ransac.setMaxIterations(1024);
    ransac.setModelType(pcl::cuda::SACMODEL_LINE);

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



    std::cout << "[PCL] ransacLine Time: " << pcl_time << " ms " << "\n";
    std::cout << "[GPU] ransacLine Time: " << gpu_all_time / (test_num - 1) << " ms " << "\n";

    //------------------模型系数---------------------
    cout << "PCL 拟合直线的模型系数为：" << endl;
    cout << "a：" << coefficients->values[0] << endl;
    cout << "b：" << coefficients->values[1] << endl;
    cout << "c：" << coefficients->values[2] << endl;
    cout << "d：" << coefficients->values[3] << endl;
    cout << "e：" << coefficients->values[4] << endl;
    cout << "f：" << coefficients->values[5] << endl;

    cout << "PCL Point Nums :" << cloud_out->points.size() << endl;


    cout << "cuPCL 拟合直线的模型系数为：" << endl;
    cout << "a：" << model_coeffs.values[0] << endl;
    cout << "b：" << model_coeffs.values[1] << endl;
    cout << "c：" << model_coeffs.values[2] << endl;
    cout << "d：" << model_coeffs.values[3] << endl;
    cout << "e：" << model_coeffs.values[4] << endl;
    cout << "f：" << model_coeffs.values[5] << endl;
    cout << "cuPCL Point Nums :" << cloud_out_gpu.size() << endl;





    // =============================================================
    // 结果验证
    // =============================================================
    std::cout << "--- Comparison ---\n";
    std::cout << "Speedup: " << pcl_time / gpu_all_time * (test_num - 1) << "x\n";
}

