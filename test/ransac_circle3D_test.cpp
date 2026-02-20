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
pcl::PointCloud<pcl::PointXYZ>::Ptr generateRansacCircle3DData(size_t num_points) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->resize(num_points);
    cloud->width = (uint32_t)num_points;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::uniform_real_distribution<float> angle_dist(0, 2.0f * M_PI);
    std::normal_distribution<float> noise(0.0f, 0.10f);

    // 圆心 (0,0,0)，半径 20，法向量 (1,1,0)
    // 需要构建两个相互垂直的基向量 u, v
    Eigen::Vector3f normal(1.0f, 1.0f, 0.0f);
    normal.normalize();
    Eigen::Vector3f u = normal.unitOrthogonal();
    Eigen::Vector3f v = normal.cross(u);

    float radius = 20.0f;
    size_t num_inliers = (size_t)(num_points * 0.7f);

    for (size_t i = 0; i < num_points; ++i) {
        if (i < num_inliers) {
            float angle = angle_dist(gen);
            float r = radius + noise(gen);
            // P = C + r*cos(a)*u + r*sin(a)*v
            Eigen::Vector3f p = r * cos(angle) * u + r * sin(angle) * v;
            cloud->points[i].x = p.x();
            cloud->points[i].y = p.y();
            cloud->points[i].z = p.z();
        }
        else {
            cloud->points[i] = { dist(gen), dist(gen), dist(gen) };
        }
    }
    return cloud;
}
void ransacCircle3D_test(size_t numPoints)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud = generateRansacCircle3DData(numPoints);
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
    coefficients->values.resize(7);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);  //inliers内点的索引
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CIRCLE3D);   //设置拟合模型参数
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
    ransac.setModelType(pcl::cuda::SACMODEL_CIRCLE3D);

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



    std::cout << "[PCL] ransacCircle3D Time: " << pcl_time << " ms " << "\n";
    std::cout << "[GPU] ransacCircle3D Time: " << gpu_all_time / (test_num - 1) << " ms " << "\n";

    //------------------模型系数---------------------
    cout << "PCL 拟合3d圆的模型系数为：" << endl;
    cout << "X：" << coefficients->values[0] << endl;
    cout << "Y：" << coefficients->values[1] << endl;
    cout << "Z：" << coefficients->values[2] << endl;
    cout << "R：" << coefficients->values[3] << endl;
    cout << "nx：" << coefficients->values[4] << endl;
    cout << "ny：" << coefficients->values[5] << endl;
    cout << "nz：" << coefficients->values[6] << endl;


    cout << "PCL Point Nums :" << cloud_out->points.size() << endl;


    cout << "cuPCL 拟合3d圆的模型系数为：" << endl;

    cout << "X：" << model_coeffs.values[0] << endl;
    cout << "Y：" << model_coeffs.values[1] << endl;
    cout << "Z：" << model_coeffs.values[2] << endl;
    cout << "R：" << model_coeffs.values[3] << endl;
    cout << "nx：" << model_coeffs.values[4] << endl;
    cout << "ny：" << model_coeffs.values[5] << endl;
    cout << "nz：" << model_coeffs.values[6] << endl;
    cout << "cuPCL Point Nums :" << cloud_out_gpu.size() << endl;





    // =============================================================
    // 结果验证
    // =============================================================
    std::cout << "--- Comparison ---\n";
    std::cout << "Speedup: " << pcl_time / gpu_all_time * (test_num - 1) << "x\n";
}

