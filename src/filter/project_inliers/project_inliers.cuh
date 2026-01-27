#pragma once
#include "pcl_cuda/types/device_cloud.h"
#include "pcl_cuda/types/model_types.h"
namespace pcl {
    namespace cuda {

        namespace device
        {

            void launchProjectInliersFilter(
                const GpuPointCloud* input,
                const ModelType& model_type,
                const ModelCoefficients& model_cofficients,
                GpuPointCloud& output
            );
        }
    }
}