#pragma once
#include "../types/device_cloud.h"
#include "../types/model_types.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>
namespace pcl {
    namespace cuda {
    
        class IterativeClosestPoint {
        public:
            IterativeClosestPoint();
            ~IterativeClosestPoint();
            IterativeClosestPoint(const IterativeClosestPoint&) = delete;
            IterativeClosestPoint& operator=(const IterativeClosestPoint&) = delete;
            IterativeClosestPoint(IterativeClosestPoint&&) noexcept;
            IterativeClosestPoint& operator=(IterativeClosestPoint&&) noexcept;
            void setInputSource(GpuPointCloud& source_input); // 'x', 'y', 'z'
            void setInputTarget(GpuPointCloud& target_input);

            void setMaximumIterations(const int & max_iterations);
            void setTransformationEpsilon(const float& transformation_epsilon);
            void setEuclideanFitnessEpsilon(const float& euclidean_fitness_epsilon);
            void setMaxCorrespondenceDistance(const float& max_correspondence_distance);
            void align(GpuPointCloud& output);
            Eigen::Matrix4f getFinalTransformation();
        private:

            struct Impl;
            std::unique_ptr<Impl> pimpl_;
        };
       
    } // namespace cuda
} // namespace pcl