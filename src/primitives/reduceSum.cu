#include "pcl_cuda/common/primitives.h"
#include"internal/device_primitives.cuh"
#include "internal/error.cuh"
#include <cuda_runtime.h>

namespace pcl {
	namespace cuda {


		__global__ void reduceSumWarpKernel(const float* __restrict__ d_in, const int numPoints, float* d_out)
		{
			const int tx = threadIdx.x;
			const int tid = tx + 8 * blockDim.x * blockIdx.x;
			float sum = 0.0f;

#pragma unroll
			for (int i = 0; i < 8; i++)
			{
				int idx = tid + i * blockDim.x;
				if (idx < numPoints)  // 仅累加有效元素，无效则加0（不break，避免分支分歧）
					sum += d_in[idx];
			}
			sum = warpSum(sum);
			const int warpId = tx / 32;
			const int laneId = tx % 32;
			const int warpNums = (blockDim.x + 31) / 32;
			__shared__ float sm_sum[32];
			if (laneId == 0)
			{
				sm_sum[warpId] = sum;
			}
			__syncthreads();

			if (warpId == 0)
			{
				sum = (laneId < warpNums ? sm_sum[laneId] : 0.0);
				sum = warpSum(sum);
			}
			if (tx == 0)
			{
				atomicAdd(d_out, sum);
			}

		}

		float reduceWarpSum(const float* d_data, size_t size)
		{
			if (size == 0) {

				std::cerr << "[Warning] reduceWarpSum received empty cloud!" << std::endl;
				return 0;
			}
			float* d_sum;
			CHECK_CUDA(cudaMalloc(&d_sum, sizeof(float)));
			CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(float)));
			const int blockSize = 512;
			const int elementsPerBlock = 8 * blockSize;
			const int gridSize = (size + elementsPerBlock - 1) / elementsPerBlock;
			reduceSumWarpKernel << <gridSize, blockSize >> > (d_data, size, d_sum);
			CHECK_CUDA_KERNEL_ERROR();
			CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	
			float h_sum;
			CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
			CHECK_CUDA(cudaFree(d_sum));
			return h_sum;

		}
		__global__ void reduceSumBlockKernel(const float* __restrict__  d_in, int numPoints, float* d_out)
		{
			__shared__ float s_m[512];
			const int tx = threadIdx.x;
			const int tid = tx + 8 * blockIdx.x * blockDim.x;
			float sum = 0.0;
#pragma unroll
			for (int i = 0; i < 8; i++)
			{
				int idx = tid + i * blockDim.x;
				if (idx < numPoints)
					sum += d_in[idx];
			}
			s_m[tx] = sum;
			__syncthreads();

			if (tx < 512 && blockDim.x >= 1024)  s_m[tx] += s_m[tx + 512];
			__syncthreads();
			if (tx < 256 && blockDim.x >= 512)  s_m[tx] += s_m[tx + 256];
			__syncthreads();
			if (tx < 128 && blockDim.x >= 256)  s_m[tx] += s_m[tx + 128];
			__syncthreads();
			if (tx < 64 && blockDim.x >= 128)  s_m[tx] += s_m[tx + 64];
			__syncthreads();
			if (tx < 32 && blockDim.x >= 64)  s_m[tx] += s_m[tx + 32];
			__syncthreads();

			if (tx < 32)
			{
				sum = s_m[tx];
				sum = warpSum(sum);
			}

			if (tx == 0)
			{
				atomicAdd(d_out, sum);
			}
		}
		// 示例：规约求和
		float reduceBlockSum(const float* d_data, size_t size)
		{
			if (size == 0) {

				std::cerr << "[Warning] reduceBlockSum received empty cloud!" << std::endl;
				return 0;
			}
			float* d_sum;
			CHECK_CUDA(cudaMalloc(&d_sum, sizeof(float)));
			CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(float)));
			const int blockSize = 512;
			const int elementsPerBlock = 8 * blockSize;
			const int gridSize = (size + elementsPerBlock - 1) / elementsPerBlock;
			reduceSumBlockKernel << <gridSize, blockSize >> > (d_data, size, d_sum);
			CHECK_CUDA_KERNEL_ERROR();
			CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	
			float h_sum;
			CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
			CHECK_CUDA(cudaFree(d_sum));
			return h_sum;
		}
	}
}
