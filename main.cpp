
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
#include "pcl_cuda/cuPCL.hpp"
using namespace std;
// 辅助函数：不同架构下 SM 里的 CUDA Core 数量不同
int convertSMtoCores(int major, int minor, int smCount) {
    int coresPerSM = 0;
    if (major == 8) { // Ampere (RTX 30xx)
        if (minor == 0) coresPerSM = 64; // A100
        else if (minor == 6) coresPerSM = 128; // RTX 3090/3080
    }
    else if (major == 7) { // Volta / Turing
        if (minor == 0) coresPerSM = 64; // V100
        else if (minor == 5) coresPerSM = 64; // RTX 20xx
    }
    else if (major == 6) { // Pascal
        if (minor == 1) coresPerSM = 128; // GTX 1080
        else if (minor == 0) coresPerSM = 64; // GP100
    }
    else if (major == 9) { // Hopper / Ada (RTX 40xx)
        coresPerSM = 128;
    }
    // 默认为未知架构，返回近似值
    return (coresPerSM == 0 ? 64 : coresPerSM) * smCount;
}
void printGpuProperties() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "!! No CUDA compatible GPU found !!" << std::endl;
        return;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        std::cout << "\n==================================================" << std::endl;
        std::cout << " Device " << dev << ": " << prop.name << std::endl;
        std::cout << "==================================================" << std::endl;

        // --- 1. 核心计算能力 (Compute Power) ---
        std::cout << "[Compute Architecture]" << std::endl;
        std::cout << "  Compute Capability:       " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Multiprocessors (SMs):    " << prop.multiProcessorCount << " (核心的'车道数')" << std::endl;
        std::cout << "  CUDA Cores (Approx):      " << convertSMtoCores(prop.major, prop.minor, prop.multiProcessorCount)
            << " (Estimated)" << std::endl;
        std::cout << "  Clock Rate:               " << prop.clockRate / 1000.0 << " MHz" << std::endl;
        std::cout << "  Warp Size:                " << prop.warpSize << " (Usually 32)" << std::endl;

        // --- 2. 显存规格 (Global Memory) ---
        std::cout << "\n[Global Memory]" << std::endl;
        std::cout << "  Total Global Memory:      " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Memory Bus Width:         " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Memory Clock Rate:        " << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
        std::cout << "  L2 Cache Size:            " << prop.l2CacheSize / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Constant Mem Size:        " << prop.totalConstMem / (1024.0) << " KB" << std::endl;
        // --- 3. 极速缓存限制 (On-Chip Memory) - 决定 Occupancy 的关键 ---
        std::cout << "\n[On-Chip Resources (Per Block/SM)]" << std::endl;
        std::cout << "  Shared Mem per Block:     " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "  Shared Mem per SM:        " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
        std::cout << "  Registers per Block:      " << prop.regsPerBlock << " (32-bit)" << std::endl;
        std::cout << "  Registers per SM:         " << prop.regsPerMultiprocessor << " (32-bit)" << std::endl;

        // --- 4. 线程布局限制 (Threading Constraints) ---
        std::cout << "\n[Threading Limits]" << std::endl;
        std::cout << "  Max Threads per Block:    " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads per SM:       " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Max Blocks per SM:        " << prop.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "  Max Grid Size:            [" << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
        std::cout << "  Max Block Dim:            [" << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;

        std::cout << "==================================================\n" << std::endl;
    }
}
int main() {
    printGpuProperties();
    const int total_points = 1000000;
    const int num_clusters = 5;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->reserve(total_points);

    // 1. 随机分配每个簇的点数，确保总和为 1,0000,000
    std::vector<int> cluster_sizes;
    int remaining_points = total_points;
    std::default_random_engine generator;

    for (int i = 0; i < num_clusters - 1; ++i) {
        // 确保后面剩下的簇至少能分到 100,000 点
        int min_val = 100000;
        int max_val = std::min(300000, remaining_points - (num_clusters - 1 - i) * 100000);
        std::uniform_int_distribution<int> distribution(min_val, max_val);
        int size = distribution(generator);
        cluster_sizes.push_back(size);
        remaining_points -= size;
    }
    cluster_sizes.push_back(remaining_points); // 最后一个簇拿走剩余所有

    // 2. 定义 5 个簇的中心点，确保它们相距足够远（欧式聚类好区分）
    std::vector<pcl::PointXYZ> centers = {
        {0.0f,   0.0f,   0.0f},
        {15.0f,  0.0f,   0.0f},
        {0.0f,   15.0f,  0.0f},
        {15.0f,  15.0f,  0.0f},
        {7.5f,   7.5f,   15.0f}
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

    // 设置欧式聚类参数
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.5);  // 设置邻域范围的距离阈值
    ec.setMinClusterSize(100000);    // 设置最小簇的大小
    ec.setMaxClusterSize(300000);  // 设置最大簇的大小
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
    nes.setMinClusterSize(100000);    // 设置最小簇的大小
    nes.setMaxClusterSize(300000);  // 设置最大簇的大小

    auto t3 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> clusters;
    for (int i = 0; i < 10; i++)
    {
        clusters.clear();
        nes.extract(clusters);
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(t4 - t3).count();


    std::cout << "[PCL] EC Time: " << pcl_time << " ms " << "\n";
    std::cout << "[GPU] EC Time: " << gpu_time / 10.0 << " ms " << "\n";
    // =============================================================
    // 结果验证
    // =============================================================
    std::cout << "--- Comparison ---\n";
    std::cout << "Speedup: " << pcl_time / gpu_time * 10 << "x\n";
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
    std::vector<float> h_gpu_nx, h_gpu_ny, h_gpu_nz, h_gpu_cuvature;
    
    // std::vector<float> h_gpu_c; 

    // 从显存下载到内存
    // 注意：请确保你的 GpuPointCloudNormal::download 函数实现了 resize 逻辑
    // h_gpu_nx.resize(NUM_POINTS_SOURCE); h_gpu_ny.resize(...); h_gpu_nz.resize(...);
    //cloud_normal.download(h_gpu_nx, h_gpu_ny, h_gpu_nz, h_gpu_cuvature);

    // 打印前 10 个点的对比
   /* std::cout << "\n==========================================================================" << std::endl;
    std::cout << "                     前 10 个法向量结果对比 (Top 10 Normals)                " << std::endl;
    std::cout << "==========================================================================" << std::endl;
    printf("| %-3s | %-24s | %-24s | %-6s |\n", "IDX", "PCL (nx, ny, nz)", "GPU (nx, ny, nz)", "Status");
    std::cout << "|-----|--------------------------|--------------------------|--------|" << std::endl;*/

    //int print_count = std::min(10, (int)source_cloud.size());
    //int match_count = 0;

    //for (int i = 0; i < print_count; ++i) {
    //    // PCL 结果
    //    float pcl_nx = normals->points[i].normal_x;
    //    float pcl_ny = normals->points[i].normal_y;
    //    float pcl_nz = normals->points[i].normal_z;
    //    float pcl_cuvature = normals->points[i].curvature;
    //    // GPU 结果
    //    float gpu_nx = h_gpu_nx[i];
    //    float gpu_ny = h_gpu_ny[i];
    //    float gpu_nz = h_gpu_nz[i];
    //    float gpu_curvature = h_gpu_cuvature[i];
    //    // 计算误差 (欧氏距离)
    //    float diff = sqrt(pow(pcl_nx - gpu_nx, 2) +
    //        pow(pcl_ny - gpu_ny, 2) +
    //        pow(pcl_nz - gpu_nz, 2));

    //    // 判断是否方向相反 (法线方向相反也是合法的，取决于视点翻转逻辑是否完全一致)
    //    // 如果 diff 很大，但 sum 很小，说明是反向了
    //    float sum_diff = sqrt(pow(pcl_nx + gpu_nx, 2) +
    //        pow(pcl_ny + gpu_ny, 2) +
    //        pow(pcl_nz + gpu_nz, 2));

    //    std::string status = "OK";
    //    if (diff > 1e-3) {
    //        if (sum_diff < 1e-3) status = "Flip"; // 方向反了，数值是对的
    //        else status = "Diff"; // 数值有较大差异
    //    }

    //    printf("| %3d | %6.3f, %6.3f, %6.3f,%6.3f | %6.3f, %6.3f, %6.3f,%6.3f | %-6s |\n",
    //        i,
    //        pcl_nx, pcl_ny, pcl_nz, pcl_cuvature,
    //        gpu_nx, gpu_ny, gpu_nz, gpu_curvature,
    //        status.c_str()
    //    );
    //}
    //std::cout << "==========================================================================" << std::endl;
    //std::cout << "* Status 'Flip' 表示法线方向相反 (180度)，但在数学上通常也是正确的平面法线。" << std::endl;
    //std::cout << "* Status 'Diff' 表示计算结果有数值偏差。" << std::endl;







    return 0;
}