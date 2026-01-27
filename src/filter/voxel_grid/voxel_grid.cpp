#include "pcl_cuda/filter/voxel_grid.h"
#include "voxel_grid.cuh"
namespace pcl
{
    namespace cuda
    {

        struct VoxelGrid::Impl
        {
            std::string axis_ = "z";
            float lx_ = 1.0f;
            float ly_ = 1.0f;
            float lz_ = 1.0f;
        };


        VoxelGrid::VoxelGrid() : pimpl_(std::make_unique<Impl>()) {}
        VoxelGrid::~VoxelGrid() = default;
        // 移动构造函数
        VoxelGrid::VoxelGrid(VoxelGrid&&) noexcept = default;
        VoxelGrid& VoxelGrid::operator=(VoxelGrid&&) noexcept = default;
        void VoxelGrid::setLeafSize(const float& lx, const float& ly, const float& lz)
        {
            pimpl_->lx_ = lx; pimpl_->ly_ = ly; pimpl_->lz_ = lz;
        };
        void VoxelGrid::filter(GpuPointCloud& output)
        {
           
            device::launchVoxelGridFilter(
                this->input_,
                pimpl_->lx_,
                pimpl_->ly_,
                pimpl_->lz_,
                output
            );


        }
    }
}