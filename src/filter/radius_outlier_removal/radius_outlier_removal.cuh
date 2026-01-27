#pragma once
#include "pcl_cuda/types/device_cloud.h"
namespace pcl {
    namespace cuda {

        namespace device
        {

            void launchRadiusOutlierRemovalFilter(
                const GpuPointCloud* input,
                const float& radius,
                const int& min_pts,
                GpuPointCloud& output
            );
        }
    }
}