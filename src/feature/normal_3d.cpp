#include "pcl_cuda/feature/normal_3d.h"
#include "normal_3d.cuh"
namespace pcl
{
    namespace cuda
    {

        struct NormalEstimation::Impl
        {
            int nr_k_ = 20;
            const GpuPointCloud* input_ = nullptr;
        };


        NormalEstimation::NormalEstimation() : pimpl_(std::make_unique<Impl>()) {}
        NormalEstimation::~NormalEstimation() = default;
        // 移动构造函数
        NormalEstimation::NormalEstimation(NormalEstimation&&) noexcept = default;
        NormalEstimation& NormalEstimation::operator=(NormalEstimation&&) noexcept = default;

        void NormalEstimation::setInputCloud(const GpuPointCloud& input)
        {
            pimpl_->input_ = &input;
        }

        void NormalEstimation::setKSearch(const int nr_k)
        {
            pimpl_->nr_k_ = nr_k;
        }
        void NormalEstimation::compute(GpuPointCloudNormal& output)
        {
            device::launchNormalEstimation(
                pimpl_->input_,
                pimpl_->nr_k_,
                output
            );
        }
    }
}