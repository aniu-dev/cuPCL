#include "passthrough.cuh"
#include "internal/error.cuh"
#include "pcl_cuda/common/common.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
namespace pcl 
{
    namespace cuda 
    {
        namespace device 
        {
            __global__ void passThroughKernel(const float* __restrict__ d_in, size_t size, int* d_mask, float min_v, float max_v, bool negative)
            {
                int tid = threadIdx.x + blockIdx.x * blockDim.x;
                if (tid >= size) return;
                float dataVal = d_in[tid];
                bool inside = (dataVal > min_v && dataVal < max_v);
                d_mask[tid] = (inside != negative) ? 1 : 0;
            }
            void launchPassThroughFilter(const GpuPointCloud* input,              
                const std::string& axis,
                const float& min_limit,
                const float& max_limit,
                const bool& negative,
                GpuPointCloud& output) 
            {
                int size = input->size();
                if (size == 0) {

                    std::cerr << "[Warning] PassThroughFilter received empty cloud!" << std::endl;
                    return;
                }
                thrust::device_vector<int> d_mask(size);
                int blockSize = 512;
                int gridSize = (size + blockSize - 1) / blockSize;

                if (axis == "x")
                {
                    passThroughKernel << <gridSize, blockSize >> > (input->x(), size, thrust::raw_pointer_cast(d_mask.data()), min_limit, max_limit, negative);
                }
                else if (axis == "y")
                {
                    passThroughKernel << < gridSize, blockSize >> > (input->y(), size, thrust::raw_pointer_cast(d_mask.data()), min_limit, max_limit, negative);
                }
                else if (axis == "z")
                {
                    passThroughKernel << <gridSize, blockSize >> > (input->z(), size, thrust::raw_pointer_cast(d_mask.data()), min_limit, max_limit, negative);
                }
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	
                copyPointCloud(*input, output, thrust::raw_pointer_cast(d_mask.data()));
            }
        }
    }
}