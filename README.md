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
- pcl::cuda::EuclideanClusterExtraction<pcl::PointXYZ> ec; 
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

| 算法分类 | 算子名称 (Function) | 数据规模 (Million) | PCL (CPU) / ms | cuPCL (GPU) / ms | 加速比 (Speedup) | 精度对比 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **基础算子** | **getMinMax3D** | 100 | 184.14 | 2.18 | **84.4x** | 完全一致 |
| | **passThrough** | 100 | 818.94 | 35.06 | **23.4x** | 完全一致 |
| **几何特征** | **OBB 包围盒** | 100 | 314,000 | 7.31 | **42,984x** | **PCL 出错 / cuPCL 修正** |
| | **Centroid (质心)** | 100 | 181.27 | 1.94 | **93.3x** | **PCL 出错 / cuPCL 修正** |
| **空间配准** | **ICP (配准)** | 10 | 539,284 | 709.97 | **759.6x** | 完全一致 (e-6) |
| **分割聚类** | **EuclideanCluster** | 10 | 1,800,000 | 6,628.7 | **271.7x** | 完全一致 |
| **特征提取** | **NormalEstimation** | 10 | 14,721.8 | 271.49 | **54.2x** | 完全一致 |
| **统计滤波** | **RadiusOutlier** | 50 | 89,586,000 | 351.17 | **255,107x** | 理论预估 |
| | **VoxelGrid** | 100 | 4,470.41 | 145.48 | **30.7x** | 完全一致 |
<img width="1343" height="2377" alt="1556a578-4fc2-42db-b44e-1480db30284f" src="https://github.com/user-attachments/assets/26068091-595e-4fd3-8f03-dc9c9d207216" />

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


