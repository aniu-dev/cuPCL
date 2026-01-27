#include "pcl_cuda/common/primitives.h"
#include"internal/device_primitives.cuh"
#include "internal/error.cuh"
#include <cuda_runtime.h>

namespace pcl {
	namespace cuda {
		__global__ void reduceMinKernel(const float* __restrict__ d_x_in, int num_points,
			float* min_x)
		{
			const int tx = threadIdx.x;
			const int tid = tx + 8 * blockIdx.x * blockDim.x;
			float min_x_temp = INFINITY;
#pragma unroll
			for (int i = 0; i < 8; i++)
			{
				int idx = tid + i * blockDim.x;
				if (idx < num_points)
				{
					float x_in_temp = d_x_in[idx];
					min_x_temp = fmin(min_x_temp, x_in_temp);

				}

			}
			__syncthreads();
			min_x_temp = warpMin(min_x_temp);

			const int warpId = tx / 32;
			const int laneId = tx % 32;
			const int warpNum = blockDim.x / 32;
			__shared__ float sx_min[32];

			if (laneId == 0)
			{
				sx_min[warpId] = min_x_temp;
			}
			__syncthreads();
			if (warpId == 0)
			{
				min_x_temp = (laneId < warpNum) ? sx_min[laneId] : INFINITY;
				min_x_temp = warpMin(min_x_temp);
			}

			if (tx == 0)
			{
				atomicMinFloat(min_x, min_x_temp);
			}

		}
		// 示例：规约求最小值
		float reduceMin(const float* d_data, size_t size)
		{
			if (size == 0) {

				std::cerr << "[Warning] reduceMin received empty cloud!" << std::endl;
				return 0;
			}
			float* d_customMinX = nullptr;
			CHECK_CUDA(cudaMalloc(&d_customMinX, sizeof(float)));
			float h_initialMin = INFINITY;
			CHECK_CUDA(cudaMemcpy(d_customMinX, &h_initialMin, sizeof(float), cudaMemcpyHostToDevice));
			const int blockSize = 512;
			const int elementsPerBlock = 8 * blockSize;
			const int gridSize = (size + elementsPerBlock - 1) / elementsPerBlock;
			reduceMinKernel << <gridSize, blockSize >> > (d_data, size, d_customMinX);
			CHECK_CUDA_KERNEL_ERROR();
			CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成					
			float h_customMinX;
			CHECK_CUDA(cudaMemcpy(&h_customMinX, d_customMinX, sizeof(float), cudaMemcpyDeviceToHost));
			CHECK_CUDA(cudaFree(d_customMinX));
			return h_customMinX;
		}













	}
}
