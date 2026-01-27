#pragma once
#include "pcl_cuda/types/device_cloud.h"
#include "pcl_cuda/types/model_types.h"
namespace pcl {
    namespace cuda {

        namespace device
        {
            
            void launchRansacSegment(
                const GpuPointCloud* input,
                const ModelType &model_type,
                const int &max_iterations, 
                const float& threshold,
                GpuPointCloud& output,
                ModelCoefficients& coefficients
            );
        }
    }
}