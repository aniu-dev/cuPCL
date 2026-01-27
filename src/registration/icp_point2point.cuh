#pragma once
#include "pcl_cuda/types/device_cloud.h"
#include <Eigen/Core>
#include <Eigen/Dense>
namespace pcl {
    namespace cuda {

        namespace device
        {

            void launchIterativeClosestPoint(
                const GpuPointCloud* source_input,
                const GpuPointCloud* target_input,
                const int& max_iterations,
                const float& transformation_epsilon,
                const float& euclidean_fitness_epsilon,
                const float& max_correspondence_distance,
                Eigen::Matrix4f& matrix,
                GpuPointCloud& output
            );
        }
    }
}