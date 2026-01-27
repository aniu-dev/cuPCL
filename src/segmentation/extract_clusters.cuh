#pragma once
#include "pcl_cuda/types/device_cloud.h"
namespace pcl {
    namespace cuda {

        namespace device
        {

            void launchEuclideanClusterExtraction(
                const GpuPointCloud* input,
                const float& radius,
                const int& min_cluster_size,
                const int& max_cluster_size,
                std::vector<std::vector<int>>& clusters
            );
        }
    }
}