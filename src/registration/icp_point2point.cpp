#include "pcl_cuda/registration/icp_point2point.h"
#include "icp_point2point.cuh"
namespace pcl
{
    namespace cuda
    {

        struct IterativeClosestPoint::Impl
        {
            int max_iterations_ = 50;
            float transformation_epsilon_ = 1e-10;
            float euclidean_fitness_epsilon_ = 1e-3;
            float max_correspondence_distance_ = 1.0;
            Eigen::Matrix4f matrix_ = Eigen::Matrix4f::Identity();
            const GpuPointCloud* source_input_ = nullptr;
            const GpuPointCloud* target_input_ = nullptr;
        };


        IterativeClosestPoint::IterativeClosestPoint() : pimpl_(std::make_unique<Impl>()) {}
        IterativeClosestPoint::~IterativeClosestPoint() = default;
        // 移动构造函数
        IterativeClosestPoint::IterativeClosestPoint(IterativeClosestPoint&&) noexcept = default;
        IterativeClosestPoint& IterativeClosestPoint::operator=(IterativeClosestPoint&&) noexcept = default;

        void IterativeClosestPoint::setInputSource(GpuPointCloud& source_input)
        {
            pimpl_->source_input_ = &source_input;
        }

        void IterativeClosestPoint::setInputTarget(GpuPointCloud& target_input)
        {
            pimpl_->target_input_ = &target_input;
        }

        void IterativeClosestPoint::setMaximumIterations(const int& max_iterations)
        {
            pimpl_->max_iterations_ = max_iterations;
        }
        void IterativeClosestPoint::setTransformationEpsilon(const float& transformation_epsilon)
        {
            pimpl_->transformation_epsilon_ = transformation_epsilon;
        }
        void IterativeClosestPoint::setEuclideanFitnessEpsilon(const float& euclidean_fitness_epsilon)
        {
            pimpl_->euclidean_fitness_epsilon_ = euclidean_fitness_epsilon;
        }
        void IterativeClosestPoint::setMaxCorrespondenceDistance(const float& max_correspondence_distance)
        {
            pimpl_->max_correspondence_distance_ = max_correspondence_distance;
        }

        Eigen::Matrix4f IterativeClosestPoint::getFinalTransformation()
        {
            Eigen::Matrix4f matrix = pimpl_->matrix_;
            return matrix;
        }
        void IterativeClosestPoint::align(GpuPointCloud& output)
        {
            device::launchIterativeClosestPoint(
                pimpl_->source_input_,
                pimpl_->target_input_,
                pimpl_->max_iterations_,
                pimpl_->transformation_epsilon_,
                pimpl_->euclidean_fitness_epsilon_,
                pimpl_->max_correspondence_distance_,
                pimpl_->matrix_,
                output
            );
        }
    }
}