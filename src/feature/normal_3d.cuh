#pragma once
#include "pcl_cuda/types/device_cloud.h"
#include <Eigen/Core>
#include <Eigen/Dense>
namespace pcl {
    namespace cuda {

        namespace device
        {

            void launchNormalEstimation(
                const GpuPointCloud* input,
                const int nr_k,
                GpuPointCloudNormal& output
            );
        }
    }
}