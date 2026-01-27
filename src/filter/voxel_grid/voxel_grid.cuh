#pragma once
#include "pcl_cuda/types/device_cloud.h"
namespace pcl {
    namespace cuda {

        namespace device
        {

            void launchVoxelGridFilter(
                const GpuPointCloud* input,
                const float& lx,
                const float& ly,
                const float& lz,
                GpuPointCloud& output
            );
        }
    }
}