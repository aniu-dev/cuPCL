# cuPCL: Industrial-Grade High-Performance CUDA Point Cloud Library
# cuPCL: 超大规模 CUDA 加速点云算法库


**cuPCL** is a high-performance CUDA operator library designed for real-time perception and massive point cloud processing (up to 100M+ points). It is not just a GPU port of PCL, but a re-engineered acceleration engine optimized for parallel architectures.
**cuPCL** 是一款专为实时感知和亿级点云处理设计的高性能 CUDA 算子库。它不仅是 PCL (Point Cloud Library) 的 GPU 移植版，更是针对并行架构深度重构的加速引擎。


## 🌟 Key Breakthroughs / 核心突破

*   **Massive Data Support (100M+)**: Successfully processes up to **100 Million** points within a **12GB** VRAM limit, covering the full pipeline (filtering, feature estimation, registration).
    *   **亿级点云支撑**: 在 **12GB** 显存限制下，完美支持 **1 亿（100M）** 级别点云的全流程处理（过滤、特征计算、配准）。
*   **Numerical Robustness**: Fixes **numerical overflow and precision errors** found in native PCL when processing datasets exceeding 10M points.
    *   **精度鲁棒性**: 修正了原生 PCL 在处理千万级以上数据时因浮点数累加导致的 **数值溢出与结果出错** 问题。
*   **Extreme Speedup**: Achieve up to **9400x+** for ICP and **19800x+** for OBB calculation (2M points).
    *   **极致加速比**: 200万点云，ICP 算法加速 **9400x+**，OBB 包围盒计算加速 **19800x+**。

---

## 💎 API Consistency / 与 PCL 严格一致的接口声明

cuPCL follows PCL's class encapsulation logic. Developers can migrate existing PCL pipelines to GPU with "zero-cost" by simply changing the namespace.
cuPCL 采用了与 PCL 官方完全一致的类封装模式。开发者只需更改命名空间，即可将现有的 PCL 流程迁移至 GPU 加速版本，实现“零成本”替换。

### Code Comparison (Euclidean Clustering)

** PCL (CPU):**
- pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
- ec.setInputCloud(cloud);
- ec.setClusterTolerance(0.02);
- ec.setMinClusterSize(100);
- ec.extract(cluster_indices); 

** cuPCL(GPU) :**
- pcl::cuda::EuclideanClusterExtraction ec; 
- ec.setInputCloud(cloud);
- ec.setClusterTolerance(0.02);
- ec.setMinClusterSize(100);
- ec.extract(cluster_indices); 

## 💻 硬件测试环境 (Hardware Specs)
项目针对最新的移动端高性能架构进行了深度优化：
- **GPU**: NVIDIA GeForce **RTX 5070 Ti Laptop** (12GB GDDR6 VRAM / Blackwell Architecture)
- **CPU**: Intel Core **i9-14900HX** (24 Cores / 32 Threads, up to 5.8 GHz)
- **内存**: 32GB DDR5 5600MHz
- **环境**: CUDA 12.6 / C++ 17 / CMake 3.18+

---

## 📊 性能对标 (Benchmark)
**Test Environment:** NVIDIA RTX 5070 Ti Laptop (12GB) | Intel i9-14900HX | CUDA 12.6
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

### 1. Linear BVH (LBVH) / 线性索引结构
Replaced recursive Kd-Trees with a custom **LBVH**. Utilizing Morton encoding and Radix Sort to convert spatial queries into linear scans, achieving 100x faster construction than PCL.
放弃传统递归 Kd-Tree，采用自研 **LBVH**。利用 Morton 码进行基数排序，将空间查询转换为高效线性扫描，构建速度比 PCL 快 100 倍以上。

### 2. RANSAC V3 Engine / 两阶段拟合引擎
A novel two-stage pipeline that reduces computation by 90% while maintaining industrial precision.
创新的“两阶段评估流水线”，在保证工业级精度的同时，将计算开销降低了 90% 以上。
*   **Stage 1 (Coarse)**: Evaluate 1/50 subset to prune 98% of the search space.
*   **Stage 2 (Refinement)**: Locked-in top 16 candidates for global refinement on 100M points.

### 3. Parallel Union-Find / 并查集并行合并
Implements **Warp-level & Shared Memory pre-merging** to resolve atomic contention, compressing 10M-point clustering from minutes to seconds.
采用 **Warp级与共享内存局部预合并** 技术，解决了高并发下的原子冲突，将千万级聚类耗时压缩至秒级。

### 4. Memory Layout Optimization (SoA) / 内存布局优化 (SoA)
The entire library adopts a **SoA (Structure of Arrays)** memory layout to ensure GPU **Memory Coalescing**, maximizing the memory bandwidth utilization of the RTX 5070 Ti.
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

### Environment(环境要求)
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
```

---


## 📝 免责声明 (Disclaimer)

EN: cuPCL is an independently developed open-source project. Its architecture and core logic are based on personal technical research and do not contain any confidential code from current or former employers. Benchmarks are fully reproducible on specified hardware.
CN: cuPCL 是个人独立开发的开源项目，底层架构与核心算法完全基于个人技术预研，不包含任何原单位或现任公司的商业保密代码。加速比数据在指定硬件环境下完全可复现。

