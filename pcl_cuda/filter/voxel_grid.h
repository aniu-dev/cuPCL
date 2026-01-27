#pragma once
#include "filters.h"
namespace pcl {
    namespace cuda {

        // 体素滤波
        class VoxelGrid : public Filter {
        public:
            VoxelGrid();
            ~VoxelGrid();
            // 禁止拷贝 ：独占资源通常禁止拷贝，或者需要实现深拷贝
            VoxelGrid(const VoxelGrid&) = delete;
            VoxelGrid& operator=(const VoxelGrid&) = delete;
            VoxelGrid(VoxelGrid&&) noexcept;
            VoxelGrid& operator=(VoxelGrid&&) noexcept;
            void setLeafSize(const float& lx,const float& ly, const float& lz);
            void filter(GpuPointCloud& output) override;
        private:
            struct Impl;
            std::unique_ptr<Impl> pimpl_;

        };


    } // namespace cuda
} // namespace pcl