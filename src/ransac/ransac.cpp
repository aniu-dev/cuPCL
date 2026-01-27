#include "pcl_cuda/ransac/ransac.h"
#include "ransac.cuh"
namespace pcl
{
    namespace cuda
    {

        struct Ransac::Impl
        {
            ModelType model_type_ = SACMODEL_PLANE;
            const GpuPointCloud* input_ = nullptr;
            float threshold_ = 0.01f;
            int max_iterations_ = 1000;
        };


        Ransac::Ransac() : pimpl_(std::make_unique<Impl>()) {}
        Ransac::~Ransac() = default;
        // 移动构造函数
        Ransac::Ransac(Ransac&&) noexcept = default;
        Ransac& Ransac::operator=(Ransac&&) noexcept = default;

        void Ransac::setInput(const GpuPointCloud* cloud) { pimpl_->input_ = cloud; };
        void Ransac::setDistanceThreshold(float threshold) { pimpl_->threshold_ = threshold; };
        void Ransac::setMaxIterations(int max_iterations) { pimpl_->max_iterations_ = max_iterations; };
        void Ransac::setModelType(ModelType model_type) { pimpl_->model_type_ = model_type; };
        void Ransac::segment(GpuPointCloud& cloud_out, ModelCoefficients& model_coeffs)
        {
            device::launchRansacSegment(
                pimpl_->input_,
                pimpl_->model_type_,
                pimpl_->max_iterations_,
                pimpl_->threshold_,
                cloud_out,
                model_coeffs);
        }
    }


}