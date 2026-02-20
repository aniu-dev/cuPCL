# cuPCL: 超大规模 CUDA 加速点云算法库


**cuPCL** 是一款专为实时感知和亿级点云处理设计的高性能 CUDA 算子库。它不仅是 PCL (Point Cloud Library) 的 GPU 移植版，更是针对并行架构深度重构的加速引擎。

### 🌟 核心突破
- **亿级点云支撑**: 在 **12GB** 显存限制下，完美支持 **1 亿（100M）** 级别点云的全流程处理（过滤、特征计算、配准）。
- **精度鲁棒性**: 修正了原生 PCL 在处理千万级以上数据时因浮点数累加导致的 **数值溢出与结果出错** 问题。
- **极致加速比**: 200万点云，ICP 算法加速 **9400x+**，OBB 包围盒计算加速 **19800x+**。

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
| **getMinMax3D** | 1 | 2.237 | 0.298 | 7.51 | 完全一致 | |
| | 10 | 19.626 | 0.485 | 40.47 | 完全一致 | |
| | 100 | 179.26 | 2.8 | 64.02 | 完全一致 | |
| **passFilter** | 1 | 14.876 | 0.868 | 17.14 | 完全一致 | |
| | 10 | 145.049 | 4.15 | 34.95 | 完全一致 | |
| | 100 | 1401.64 | 28.89 | 48.52 | 完全一致 | 1亿滤波后计算量变少，且无索引拷贝时间 |
| **Centroid3D** | 1 | 1.958 | 0.355 | 5.52 | 完全一致 | |
| | 10 | 17.007 | 0.738 | 23.04 | 完全一致 | 小数点后7位才不同，认为一致 |
| | 100 | 162.172 | 4.4637 | 36.33 | 完全一致 | |
| **Covariance** | 1 | 1.899 | 0.243 | 7.82 | e-4 | PCL协方差和质心计算误差较大 |
| | 10 | 17.556 | 1.261 | 13.92 | e-4 | |
| | 100 | 152.035 | 11.595 | 13.11 | e-4 | |
| **OBB包围盒** | 1 | 2776.05 | 0.293 | 9474.57 | e-4 | |
| | 10 | 28892.2 | 1.611 | 17934.33 | e-4 | |
| | 100 | 283460 | 14.302 | **19819.61** | PCL出错 | PCL在大规模点云下出错，cuPCL极致加速 |
| **transformPointCloud** | 1 | 6.185 | 0.539 | 11.47 | 完全一致 | |
| | 10 | 45.364 | 0.814 | 55.73 | 完全一致 | |
| | 100 | 574.423 | 6.195 | 92.72 | 完全一致 | |
| **copyPointCloud** | 1 | 4.972 | 1.22 | 4.08 | 完全一致 | 耗时主因在显存申请与拷贝开销 |
| | 10 | 83.81 | 13.208 | 6.35 | 完全一致 | |
| | 100 | 943.92 | 144.7 | 6.52 | 完全一致 | |
| **ransacPlane_V3** | 1 | 56.43 | 1.802 | 31.32 | e-4 | |
| | 10 | 787.23 | 6.223 | 126.49 | e-4 | |
| | 100 | 7163.49 | 48.397 | 148.02 | PCL出错 | PCL内点数量错误，cuPCL采用两阶段拟合 |
| **ransacLine_V3** | 1 | 156.97 | 2.109 | 74.43 | e-4 | |
| | 10 | 1450.77 | 5.715 | 253.85 | e-4 | |
| | 100 | 15101.4 | 48.924 | 308.67 | e-4 | cuPCL耗时不随参数变动，表现稳定 |
| **ransacCircle3D_V3** | 1 | 1946.71 | 1.811 | 1074.94 | e-4 | |
| | 10 | 20261 | 6.569 | 3084.34 | e-4 | 两阶段RANSAC，大幅缓解计算压力 |
| | 100 | 178049 | 49.168 | **3621.23** | e-4 | |
| **voxel_grid** | 1 | 48.39 | 3.8 | 12.73 | 完全一致 | |
| | 10 | 456.61 | 23.21 | 19.67 | 完全一致 | |
| | 100 | 4588.64 | 227.719 | 20.15 | 完全一致 | 采用64-bit Indexing支持百米级高分辨率 |
| **EuclideanCluster** | 1 | 15308 | 46.31 | 330.57 | 完全一致 | 针对Warp级原子操作优化，减少冲突 |
| | 5 | 448897 | 853.96 | 525.66 | 完全一致 | |
| | 10 | 1.80E+06 | 3433.07 | 524.31 | 完全一致 | |
| **NormalEstimation** | 1 | 3317.16 | 29.78 | 111.37 | 完全一致 | k=20 |
| | 10 | 45154.2 | 290.77 | 155.29 | 完全一致 | |
| | 50 | 257584 | 1472.43 | 174.94 | 完全一致 | 基于LBVH树，支持大规模点云搜索 |
| **radius_outlier** | 1 | 2752.54 | 12.55 | 219.33 | 完全一致 | |
| | 10 | 42505.1 | 88.08 | 482.54 | 完全一致 | |
| | 50 | 286584 | 407.98 | 702.45 | 完全一致 | 基于LBVH剪枝策略，快速定位邻域点 |
| **icp** | 0.5 | 20494.7 | 10.245 | 2000.44 | e-6 | |
| | 1 | 66547 | 15.892 | 4187.45 | e-6 | cuPCL精度比PCL更接近真实值 |
| | 2 | 264609 | 29.252 | **9045.84** | e-6 | 200万点配准仅需29ms |

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


