#include "pcl_cuda/common/common.h"
#include "internal/error.cuh"
#include <device_launch_parameters.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include <thrust/count.h>
namespace pcl {
    namespace cuda {
        __constant__ float d_T[16];
        __global__ void transformPointCloudKernel(
            const float* __restrict__ d_in_x,
            const float* __restrict__ d_in_y,
            const float* __restrict__ d_in_z,
            float* __restrict__ d_out_x,
            float* __restrict__ d_out_y,
            float* __restrict__ d_out_z,
            int numPoints
        ) {
            // 计算全局线程ID（每个线程处理1个点）
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numPoints) return;  // 边界检查

            // 读取输入点（寄存器暂存，减少全局内存访问）
            const float x = d_in_x[idx];
            const float y = d_in_y[idx];
            const float z = d_in_z[idx];

            // 应用变换矩阵（直接展开计算，避免分支和循环）
            const float x_out = d_T[0] * x + d_T[4] * y + d_T[8] * z + d_T[12];
            const float y_out = d_T[1] * x + d_T[5] * y + d_T[9] * z + d_T[13];
            const float z_out = d_T[2] * x + d_T[6] * y + d_T[10] * z + d_T[14];

            // 写入输出点（合并访问全局内存）
            d_out_x[idx] = x_out;
            d_out_y[idx] = y_out;
            d_out_z[idx] = z_out;
        }
        void transformPointCloud(const GpuPointCloud& cloud_in, GpuPointCloud& cloud_out, Eigen::Matrix4f matrix_)
        {
            size_t size = cloud_in.size();
            if (size == 0) {

                std::cerr << "[Warning] transformPointCloud received empty cloud!" << std::endl;
                return;
            }
            CHECK_CUDA(cudaMemcpyToSymbol(d_T, matrix_.data(), 16 * sizeof(float)));
            const int blockSize = 512;
            const int gridSize = (size + blockSize - 1) / blockSize;
            cloud_out.alloc(size);
            transformPointCloudKernel << <gridSize, blockSize >> > (cloud_in.x(), cloud_in.y(), cloud_in.z(), 
                cloud_out.x(), cloud_out.y(), cloud_out.z(), size);
            CHECK_CUDA_KERNEL_ERROR();
            CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	
        }
    } // namespace cuda
} // namespace pcl