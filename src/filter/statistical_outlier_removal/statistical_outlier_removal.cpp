
#include "pcl_cuda/filter/statistical_outlier_removal.h"
#include "statistical_outlier_removal.cuh"
namespace pcl
{
    namespace cuda
    {

        struct StatisticalOutlierRemoval::Impl
        {
            int nr_k_ = 20;
            float stddev_mult_ = 1.0f;
            
        };


        StatisticalOutlierRemoval::StatisticalOutlierRemoval() : pimpl_(std::make_unique<Impl>()) {}
        StatisticalOutlierRemoval::~StatisticalOutlierRemoval() = default;
        // 移动构造函数
        StatisticalOutlierRemoval::StatisticalOutlierRemoval(StatisticalOutlierRemoval&&) noexcept = default;
        StatisticalOutlierRemoval& StatisticalOutlierRemoval::operator=(StatisticalOutlierRemoval&&) noexcept = default;

        void StatisticalOutlierRemoval::setMeanK(const int& nr_k) { pimpl_->nr_k_ = nr_k; };
        void StatisticalOutlierRemoval::setStddevMulThresh(const float& stddev_mult) { pimpl_->stddev_mult_ = stddev_mult; };
        void StatisticalOutlierRemoval::filter(GpuPointCloud& output)
        {
            device::launchStatisticalOutlierRemovalFilter(
                this->input_,
                pimpl_->nr_k_,
                pimpl_->stddev_mult_,
                output);
        }
    }
}
