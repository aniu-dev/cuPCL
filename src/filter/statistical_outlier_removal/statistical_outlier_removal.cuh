#pragma once
#include "pcl_cuda/types/device_cloud.h"
namespace pcl {
    namespace cuda {

        namespace device
        {
 
            void launchStatisticalOutlierRemovalFilter(
                const GpuPointCloud* input,
                const int& nr_k,
                const float& stddev_mult,
                GpuPointCloud& output
            );
        }
    }
}