#include "pcl_cuda/common/common.h"
#include"internal/device_primitives.cuh"

#include "internal/error.cuh"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include <thrust/count.h>
namespace pcl {
    namespace cuda {
        __global__ void pointCloud2DeepMatKernel(const float* __restrict__ d_in_x,
            const float* __restrict__ d_in_y,
            const float* __restrict__ d_in_z,
            float* __restrict__ mat_out,
            float x_min, float y_min, float inv_resolution, int width,
            int numPoints)
        {
            // 计算全局线程ID（每个线程处理1个点）
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numPoints) return;  // 边界检查

            const float x = d_in_x[idx];
            const float y = d_in_y[idx];
            const float z = d_in_z[idx];

            int u = __float2int_rn((x - x_min) * inv_resolution);
            int v = __float2int_rn((y - y_min) * inv_resolution);
            atomicMaxFloat(&mat_out[v * width + u], z);
            //mat_out[v * width + u] = z;
        }
        void pointCloudToDeepMat(const GpuPointCloud& cloud_in, const int width, const int height,
            float* mat_out, float resolution, float min_x, float min_y)
        {
            size_t size = cloud_in.size();
            if (size == 0) {

                std::cerr << "[Warning] pointCloudToDeepMat received empty cloud!" << std::endl;
                return;
            }
            const int blockSize = 512;
            const int gridSize = (size + blockSize - 1) / blockSize;
            float* d_matout;
           
            CHECK_CUDA(cudaMalloc(&d_matout, sizeof(float) * width * height));
            float resolution_inv = 1.0 / resolution;
            pointCloud2DeepMatKernel << <gridSize, blockSize >> > (cloud_in.x(), cloud_in.y(), cloud_in.z(), d_matout,
                min_x, min_y, resolution_inv, width, size);
            CHECK_CUDA_KERNEL_ERROR();
            CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	
          
            CHECK_CUDA(cudaMemcpy(mat_out, d_matout, sizeof(float) * width * height, cudaMemcpyDeviceToHost));

            CHECK_CUDA(cudaFree(d_matout));


        }
    } // namespace cuda
} // namespace pcl