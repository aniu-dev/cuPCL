#include "pcl_cuda/filter/passthrough.h"
#include "passthrough.cuh"
namespace pcl
{
    namespace cuda
    {

        struct PassThrough::Impl
        {
            std::string axis_ = "z";
            float min_limit_ = -1000.0f;
            float max_limit_ = 1000.f;
            bool negative_ = false;
        };


        PassThrough::PassThrough() : pimpl_(std::make_unique<Impl>()) {}
        PassThrough::~PassThrough() = default;
        // 移动构造函数
        PassThrough::PassThrough(PassThrough&&) noexcept = default;
        PassThrough& PassThrough::operator=(PassThrough&&) noexcept = default;

        void PassThrough::setFilterFieldName(const std::string& axis)
        { pimpl_->axis_ = axis; }
        void PassThrough::setFilterLimits(const float& min_limit, const float& max_limit)
        { pimpl_->min_limit_ = min_limit; pimpl_->max_limit_ = max_limit; }
        void PassThrough::setNegative(const bool& negative) { pimpl_->negative_ = negative; }
        void PassThrough::filter(GpuPointCloud& output)
        {
            device::launchPassThroughFilter(
                this->input_,
                pimpl_->axis_,
                pimpl_->min_limit_,
                pimpl_->max_limit_,
                pimpl_->negative_,
                output);
        }
    }
}