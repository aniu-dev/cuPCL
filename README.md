# cuPCL: 超大规模 CUDA 加速点云算法库


**cuPCL** 是一款专为实时感知和亿级点云处理设计的高性能 CUDA 算子库。它不仅是 PCL (Point Cloud Library) 的 GPU 移植版，更是针对并行架构深度重构的加速引擎。

### 🌟 核心突破
- **亿级点云支撑**: 在 **12GB** 显存限制下，完美支持 **1 亿（100M）** 级别点云的全流程处理（过滤、特征计算、配准）。
- **精度鲁棒性**: 修正了原生 PCL 在处理千万级以上数据时因浮点数累加导致的 **数值溢出与结果出错** 问题。
- **极致加速比**: ICP 算法加速 **750x+**，OBB 包围盒计算加速 **42,000x+**。

---
## 💎 与 PCL 严格一致的接口声明 (API Consistency)

cuPCL 采用了与 PCL 官方完全一致的类封装模式。开发者只需更改命名空间，即可将现有的 PCL 流程迁移至 GPU 加速版本，实现“零成本”替换。

### 代码对比示例 (Euclidean Clustering)

**原生 PCL (CPU):**
- pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
- ec.setInputCloud(cloud);
- ec.setClusterTolerance(0.02);
- ec.setMinClusterSize(100);
- ec.extract(cluster_indices); // 耗时：数分钟 (针对10M点)

**GPU 加速:**
// 接口、方法名、参数完全对齐
- pcl::cuda::EuclideanClusterExtraction ec; 
- ec.setInputCloud(cloud);
- ec.setClusterTolerance(0.02);
- ec.setMinClusterSize(100);
- ec.extract(cluster_indices); // 耗时：约 6 秒 (针对10M点)

## 💻 硬件测试环境 (Hardware Specs)

项目针对最新的移动端高性能架构进行了深度优化：

- **GPU**: NVIDIA GeForce **RTX 5070 Ti Laptop** (12GB GDDR6 VRAM / Blackwell Architecture)
- **CPU**: Intel Core **i9-14900HX** (24 Cores / 32 Threads, up to 5.8 GHz)
- **内存**: 32GB DDR5 5600MHz
- **环境**: CUDA 12.6 / C++ 17 / CMake 3.18+

---

## 📊 性能对标 (Benchmark)

以下数据基于 **RTX 5070 Ti** 与 **PCL 1.14 (CPU单核)** 的对比测试。

| 算法名称 (Function) | 数据规模 (million) | PCL (CPU) / ms | cuPCL (GPU) / ms | 加速比 (Speedup) | 结果对比 | 备注 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **getMinMax3D** | 1 | 1.804 | 0.096 | 18.79 | 完全一致 | |
| | 10 | 20.078 | 0.296 | 67.83 | 完全一致 | |
| | 100 | 184.14 | 2.181 | 84.43 | 完全一致 | |
| **passFilter** | 1 | 11.166 | 3.64 | 3.07 | 完全一致 | |
| | 10 | 77.22 | 4.835 | 15.97 | 完全一致 | |
| | 100 | 818.939 | 35.0625 | 23.36 | 完全一致 | 对比copyPointCloud时间短的主要原因是，1亿滤波后只有几千万，而copy函数是1亿的索引，所以申请时间变少，并且没有索引的拷贝时间 |
| **Centroid3D** | 1 | 3.806 | 0.077 | 49.43 | e-4 | |
| | 10 | 22.2977 | 0.252 | 88.48 | e-4 | |
| | 100 | 181.269 | 1.942 | 93.34 | e-4 | PCL结果出错 |
| **Covariance** | 1 | 3.986 | 0.064 | 62.28 | e-2 | |
| | 10 | 44.7842 | 0.215 | 208.3 | e-2 | |
| | 100 | 372.181 | 1.996 | 186.46 | e-2 | PCL结果出错 |
| **OBB包围盒** | 1 | 2926.95 | 0.13 | 22515 | e-4 | |
| | 10 | 28643.8 | 0.773 | 37055.37 | e-4 | |
| | 100 | 314000 | 7.305 | 42984.26 | e-4 | PCL结果出错 |
| **transformPointCloud** | 1 | 6.185 | 0.539 | 11.47 | 完全一致 | |
| | 10 | 45.364 | 0.814 | 55.73 | 完全一致 | |
| | 100 | 574.423 | 6.1952 | 92.72 | 完全一致 | |
| **copyPointCloud** | 1 | 3.31 | 1.019 | 3.25 | 完全一致 | |
| | 10 | 35.459 | 5.982 | 5.93 | 完全一致 | |
| | 100 | 409.651 | 83.3887 | 4.91 | 完全一致 | 主要原因在于索引拷贝时间和输出点云申请空间时间过长，占整体的89%时间 |
| **pointCloud2Mat** | 1 | 69.3323 | 0.8766 | 79.09 | 完全一致 | |
| | 10 | 151.857 | 2.296 | 66.14 | 完全一致 | |
| | 100 | 877.365 | 19.879 | 44.14 | 完全一致 | 多个点映射到同一个2D坐标时，导致产生原子冲突 |
| **projectPlane** | 1 | 7.712 | 0.173 | 44.58 | 完全一致 | |
| | 10 | 85.9887 | 0.627718 | 136.99 | 完全一致 | |
| | 100 | 998.513 | 5.39297 | 185.15 | 完全一致 | |
| **projectLine** | 1 | 10.557 | 0.159 | 66.4 | 完全一致 | |
| | 10 | 88.5049 | 0.704 | 125.72 | 完全一致 | |
| | 100 | 991.478 | 5.142 | 192.82 | 完全一致 | |
| **projectSphere** | 1 | 9.811 | 0.302 | 32.49 | 完全一致 | |
| | 10 | 99.8724 | 0.566171 | 176.4 | 完全一致 | |
| | 100 | 1150.57 | 4.98104 | 230.99 | 完全一致 | |
| **projectCylinder** | 1 | 10.0123 | 0.272 | 36.81 | 完全一致 | |
| | 10 | 97.569 | 1.235 | 79 | 完全一致 | |
| | 100 | 1096.59 | 6.021 | 182.13 | 完全一致 | |
| **ransacPlane_V1** | 1 | 22.51 | 4.21267 | 5.34 | e-4 | 基于1个block计算多个参数模型 |
| | 10 | 199.514 | 31.7079 | 6.29 | e-4 | |
| | 100 | 1845.66 | 338.471 | 5.45 | PCL结果出错 | PCL结果出错 |
| **ransacLine_V1** | 1 | 29.0283 | 5.5017 | 5.28 | e-4 | |
| | 10 | 256.451 | 57.3227 | 4.47 | e-4 | |
| | 100 | 2531.31 | 571.603 | 4.43 | e-4 | |
| **ransacCircle2D_V1**| 1 | 66.939 | 4.31867 | 15.5 | e-4 | |
| | 10 | 710.886 | 45.6793 | 15.56 | e-4 | |
| | 100 | 6906.27 | 459.423 | 15.03 | e-4 | |
| **ransacCircle3D_V1**| 1 | 74.624 | 7.22686 | 10.33 | e-4 | |
| | 10 | 724.583 | 72.8217 | 9.95 | e-4 | |
| | 100 | 7102.31 | 728.771 | 9.75 | e-4 | |
| **ransacSphere_V1** | 1 | 70.2069 | 4.49417 | 15.62 | e-4 | |
| | 10 | 738.505 | 43.3817 | 17.02 | e-4 | |
| | 100 | 8167 | 455 | 17.95 | e-4 | |
| **ransacPlane_V2** | 1 | 22.51 | 1.216 | 18.51 | e-4 | 基于1个threads计算所有参数模型，最多支持1024个参数模型 |
| | 10 | 199.514 | 10.442 | 19.11 | e-4 | |
| | 100 | 2045.66 | 100.045 | 20.45 | e-4 | |
| **ransacLine_V2** | 1 | 29.0283 | 1.95714 | 14.83 | e-4 | |
| | 10 | 256.451 | 17.9796 | 14.26 | e-4 | |
| | 100 | 2531.31 | 177.307 | 14.28 | e-4 | |
| **ransacCircle2D_V2**| 1 | 66.939 | 1.71641 | 39 | e-4 | |
| | 10 | 710.886 | 15.2838 | 46.51 | e-4 | |
| | 100 | 6906.27 | 151.664 | 45.54 | e-4 | |
| **ransacCircle3D_V2**| 1 | 74.624 | 3.252 | 22.95 | e-4 | |
| | 10 | 724.583 | 30.4022 | 23.83 | e-4 | |
| | 100 | 7102.31 | 302.221 | 23.5 | e-4 | |
| **ransacSphere_V2** | 1 | 70.2069 | 1.917 | 36.62 | e-4 | |
| | 10 | 738.505 | 17.253 | 42.8 | e-4 | |
| | 100 | 8167 | 172.637 | 47.31 | e-4 | |
| **EuclideanClusterExtraction** | 1 | 15308.6 | 75.32 | 203.25 | 完全一致 | |
| | 5 | 448897 | 1549.48 | 289.71 | 完全一致 | gpu时间慢的主要原因是，并查集的原子操作冲突，但是暂时没有好的办法解决 |
| | 10 | 1800000 | 6628.74 | 271.72 | 完全一致 | |
| **NormalEstimation** | 1 | 1418.62 | 37.6249 | 37.7 | 完全一致 | |
| | 10 | 14721.8 | 271.491 | 54.23 | 完全一致 | |
| | 50 | 76092.2 | 1029.44 | 73.92 | 完全一致 | 电脑内存为12GB，LBVH树的节点是64字节，一般为2N个节点，超过5000万时，内存占用显著增加，加上各种buffer，可能导致内存溢出，可能触发驱动的虚拟内存（系统内存交换），导致时间增加，因此使用LBVH树时，尽量不超过5000万为好 |
| **radius_outlier_removal** | 1 | 1712.66 | 7 | 244.67 | 完全一致 | SEARCH_RADIUS = 1.0f; MIN_NEIGHBORS = 5; |
| | 10 | 179172 | 91.02 | 1968.49 | 完全一致 | SEARCH_RADIUS = 1.0f; MIN_NEIGHBORS = 50; |
| | 50 | 89586000 | 351.17 | 255107.21 | PCL未测试时间过长 | SEARCH_RADIUS = 1.0f; (时间为预估) |
| **voxel_grid** | 1 | 42.5 | 4.58 | 9.28 | 完全一致 | lx ly lz = 1.5f; const float; WORLD_SIZE = 100.0f; |
| | 10 | 442.56 | 16.77 | 26.39 | 完全一致 | lx ly lz = 1.5f; const float; WORLD_SIZE = 150.0f; |
| | 100 | 4470.41 | 145.48 | 30.73 | 完全一致 | lx ly lz = 1.5f; const float; WORLD_SIZE = 250.0f; |
| **statistical_outlier_removal** | 1 | 1315.13 | 17.44 | 75.41 | 完全一致 | int nr_k = 20; float stddev_mult = 1.2; |
| | 10 | 13363.3 | 144.54 | 92.45 | 完全一致 | int nr_k = 20; float stddev_mult = 1.2; |
| | 50 | 71175.6 | 756.68 | 94.06 | 完全一致 | int nr_k = 20; float stddev_mult = 1.2; |
| **icp** | 1 | 27026.7 | 66.6 | 405.81 | e-6 | |
| | 5 | 227680 | 340.813 | 668.05 | e-6 | |
| | 10 | 539284 | 709.97 | 759.59 | e-6 | |

---

## 🛠 关键技术栈 (Technical Implementation)

### 1. 线性索引结构 (Linear BVH)
放弃了传统的递归 Kd-Tree，采用自研的 **LBVH**。利用 Morton 码进行基数排序，将空间邻域查询转换为高效的线性扫描，构建速度比 PCL 快 100 倍以上。

### 2. RANSAC V2 架构
实现了线程级并行的 RANSAC 拟合，每个线程独立评估 1024 个模型候选参数，极大提升了在噪声环境下的平面、球面拟合精度。

### 3. 并查集 (Union-Find) 异步合并
在欧式聚类中，针对全局原子操作冲突问题，采用了 **Shared Memory 局部预合并** 与 **原子操作重排** 技术，将千万级点的聚类时间从分钟级压缩至秒级。

### 4. 内存布局优化 (SoA)
全库采用 **SoA (Structure of Arrays)** 内存布局，确保了 GPU 显存访问的合并（Memory Coalescing），最大化利用了 RTX 5070 Ti 的显存带宽。

---

## 🛠 算子路线图 (Algorithm Checklist)

- [x] **Core**: MinMax, PassThrough, Transform, Copy, MatConversion.
- [x] **Indexing**: LBVH Construction, KNN Search, Radius Search.
- [x] **Geometry**: Centroid, Covariance, OBB/AABB Box, Projection.
- [x] **Segmentation**: Euclidean Cluster Extraction, RANSAC (Plane/Sphere/Line).
- [x] **Registration**: ICP (Iterative Closest Point).
- [x] **Filters**: RadiusOutlierRemoval, StatisticalOutlierRemoval, VoxelGrid.
- [x] **Features**: Normal Estimation, Curvature Calculation.
- [ ] **Planned**: Marching Cubes, Surface Reconstruction, Deep Learning Kernels.

---

## 📦 构建说明 (Build)

### 环境要求
- **CMake**: 3.18+
- **CUDA**: 12.x
- **PCL**: 1.14+ (仅用于验证结果与 IO)

### 编译步骤
```bash
# 克隆仓库
git clone https://github.com/aniu-dev/cuPCL.git
cd cuPCL

# 编译
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
---


## 📝 免责声明
cuPCL 是个人独立开发的开源项目，不包含任何商业公司保密代码。所有加速比数据均有完整的 Benchmark 代码可复现。


