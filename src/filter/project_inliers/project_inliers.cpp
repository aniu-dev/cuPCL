#include "pcl_cuda/filter/project_inliers.h"
#include "project_inliers.cuh"
namespace pcl
{
    namespace cuda
    {

        struct ProjectInliersFilter::Impl
        {
            ModelType model_type_ = SACMODEL_PLANE;
            ModelCoefficients model_cofficients_ = {{0.0f, 0.0f, 1.0f, 0.0f}};
        };
        ProjectInliersFilter::ProjectInliersFilter() : pimpl_(std::make_unique<Impl>()) {}
        ProjectInliersFilter::~ProjectInliersFilter() = default;
        // 移动构造函数
        ProjectInliersFilter::ProjectInliersFilter(ProjectInliersFilter&&) noexcept = default;
        ProjectInliersFilter& ProjectInliersFilter::operator=(ProjectInliersFilter&&) noexcept = default;

        void ProjectInliersFilter::setModelType(const ModelType &model_type) { pimpl_->model_type_ = model_type; };
        void ProjectInliersFilter::setModelCoefficients(const ModelCoefficients &model_cofficients) 
        {pimpl_->model_cofficients_ = model_cofficients; };
        void ProjectInliersFilter::filter(GpuPointCloud& output)
        {
            device::launchProjectInliersFilter(
                this->input_,
                pimpl_->model_type_,
                pimpl_->model_cofficients_,
                output);
        }





    }


}