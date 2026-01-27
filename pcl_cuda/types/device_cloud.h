#pragma once
#include <vector>
#include<string>
#include<iostream>
namespace pcl {
    namespace cuda {
        struct PointXYZ { float x, y, z; };

        class GpuPointCloud {
        public:
            GpuPointCloud();
            ~GpuPointCloud(); // 在 .cu 中实现释放
            // =========================================================
          // 核心：资源管理 (Rule of 5)
          // =========================================================

          // 1. 禁止拷贝 (防止如果不小心写了 B=A，导致严重的深拷贝性能问题或Double Free)
            GpuPointCloud(const GpuPointCloud&) = delete;
            GpuPointCloud& operator=(const GpuPointCloud&) = delete;

            // 2. 启用移动 (Move) - 允许所有权转移
            // 这让你可以写: GpuPointCloud cloud = someFunction();
            // 实现细节在 .cu 中，头文件很干净
            GpuPointCloud(GpuPointCloud&& other) noexcept;
            GpuPointCloud& operator=(GpuPointCloud&& other) noexcept;
            // 分配显存
            void alloc(size_t size);

            // 清空并释放显存
            void clear();
            // 从主机内存上传 (Host -> Device)
            void upload(const std::vector<float>& h_x, const std::vector<float>& h_y, const std::vector<float>& h_z);

            // 下载到主机内存 (Device -> Host)
            void download(std::vector<float>& h_x, std::vector<float>& h_y, std::vector<float>& h_z) const;
            void fromDevicePointers(const float* d_x, const float* d_y, const float* d_z, size_t count);

           
            // 获取原始指针（用于传给 Kernel）
            float* x() { return x_; }
            float* y() { return y_; }
            float* z() { return z_; }
            const float* x() const { return x_; }
            const float* y() const { return y_; }
            const float* z() const { return z_; }

            size_t size() const { return size_; }
            bool empty() const { return size_ == 0; }


        private:
            // 允许与其他 GpuPointCloud 交换指针（用于高效的 Filter 输出）
            void swap(GpuPointCloud& other);
            float* x_, * y_, * z_;
            size_t size_;
            size_t capacity_;
        };



        class GpuPointCloudNormal
        {
        public:
            GpuPointCloudNormal();
            ~GpuPointCloudNormal(); // 在 .cu 中实现释放
            // =========================================================
          // 核心：资源管理 (Rule of 5)
          // =========================================================

          // 1. 禁止拷贝 (防止如果不小心写了 B=A，导致严重的深拷贝性能问题或Double Free)
            GpuPointCloudNormal(const GpuPointCloudNormal&) = delete;
            GpuPointCloudNormal& operator=(const GpuPointCloudNormal&) = delete;

            // 2. 启用移动 (Move) - 允许所有权转移
            // 这让你可以写: GpuPointCloud cloud = someFunction();
            // 实现细节在 .cu 中，头文件很干净
            GpuPointCloudNormal(GpuPointCloudNormal&& other) noexcept;
            GpuPointCloudNormal& operator=(GpuPointCloudNormal&& other) noexcept;
            // 分配显存
            void alloc(size_t size);

            // 清空并释放显存
            void clear();
            // 从主机内存上传 (Host -> Device)
            void upload(const std::vector<float>& h_nx, const std::vector<float>& h_ny, const std::vector<float>& h_nz, const std::vector<float>& h_curvature);

            // 下载到主机内存 (Device -> Host)
            void download(std::vector<float>& h_nx, std::vector<float>& h_ny, std::vector<float>& h_nz, std::vector<float>& h_curvature) const;
            void fromDevicePointers(const float* d_nx, const float* d_ny, const float* d_nz, const float* d_curvature, size_t count);


            // 获取原始指针（用于传给 Kernel）
            float* nx() { return nx_; }
            float* ny() { return ny_; }
            float* nz() { return nz_; }
            float* curvature() { return curvature_; }
            const float* nx() const { return nx_; }
            const float* ny() const { return ny_; }
            const float* nz() const { return nz_; }
            const float* curvature() const { return curvature_; }
            size_t size() const { return size_; }
            bool empty() const { return size_ == 0; }


        private:
            // 允许与其他 GpuPointCloud 交换指针（用于高效的 Filter 输出）
            void swap(GpuPointCloudNormal& other);
            float* nx_, * ny_, * nz_,*curvature_;
            size_t size_;
            size_t capacity_;
        };



    } // namespace cuda
} // namespace pcl
