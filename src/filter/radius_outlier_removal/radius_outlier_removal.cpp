
#include "pcl_cuda/filter/radius_outlier_removal.h"
#include "radius_outlier_removal.cuh"
namespace pcl
{
    namespace cuda
    {

        struct RadiusOutlierRemoval::Impl
        {
            float radius_ = 0.1f;
            int min_pts_ = 10;
        };


        RadiusOutlierRemoval::RadiusOutlierRemoval() : pimpl_(std::make_unique<Impl>()) {}
        RadiusOutlierRemoval::~RadiusOutlierRemoval() = default;
        // 移动构造函数
        RadiusOutlierRemoval::RadiusOutlierRemoval(RadiusOutlierRemoval&&) noexcept = default;
        RadiusOutlierRemoval& RadiusOutlierRemoval::operator=(RadiusOutlierRemoval&&) noexcept = default;

        void RadiusOutlierRemoval::setRadiusSearch(const float &radius) { pimpl_->radius_ = radius; };
        void RadiusOutlierRemoval::setMinNeighborsInRadius(const int &min_pts) { pimpl_->min_pts_ = min_pts; };
        void RadiusOutlierRemoval::filter(GpuPointCloud& output)
        {
            device::launchRadiusOutlierRemovalFilter(
                this->input_,
                pimpl_->radius_,
                pimpl_->min_pts_,
                output);
        }





    }


}



