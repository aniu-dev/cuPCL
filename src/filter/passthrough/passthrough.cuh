#pragma once
#include "pcl_cuda/types/device_cloud.h"
namespace pcl {
    namespace cuda {

        namespace device 
        {

            void launchPassThroughFilter(
                const GpuPointCloud* input,
                const std::string &axis,
                const float& min_limit,
                const float& max_limit,
                const bool& negative,
                GpuPointCloud& output
            );
        }
    }
}