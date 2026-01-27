#include "pcl_cuda/common/primitives.h"
#include"internal/device_primitives.cuh"
#include "internal/error.cuh"
#include <cuda_runtime.h>

namespace pcl {
	namespace cuda {

		__global__ void reduceMaxKernel(const float* __restrict__ d_x_in, int num_points, float* max_x)
		{
			const int tx = threadIdx.x;
			const int tid = tx + 8 * blockIdx.x * blockDim.x;
			float max_x_temp = -INFINITY;
#pragma unroll
			for (int i = 0; i < 8; i++)
			{
				int idx = tid + i * blockDim.x;
				if (idx < num_points)
				{
					float x_in_temp = d_x_in[idx];
					max_x_temp = fmax(max_x_temp, x_in_temp);
				}

			}
			__syncthreads();
			max_x_temp = warpMax(max_x_temp);
			const int warpId = tx / 32;
			const int laneId = tx % 32;
			const int warpNum = blockDim.x / 32;
			__shared__ float sx_max[32];

			if (laneId == 0)
			{
				sx_max[warpId] = max_x_temp;

			}
			__syncthreads();
			if (warpId == 0)
			{
				max_x_temp = (laneId < warpNum) ? sx_max[laneId] : -INFINITY;
				max_x_temp = warpMax(max_x_temp);
			}

			if (tx == 0)
			{
				atomicMaxFloat(max_x, max_x_temp);
			}

		}
		float reduceMax(const float* d_data, size_t size)
		{
			if (size == 0) {

				std::cerr << "[Warning] reduceMax received empty cloud!" << std::endl;
				return 0;
			}
			float* d_customMaxX = nullptr;
			CHECK_CUDA(cudaMalloc(&d_customMaxX, sizeof(float)));
			float h_initialMax = -INFINITY;
			CHECK_CUDA(cudaMemcpy(d_customMaxX, &h_initialMax, sizeof(float), cudaMemcpyHostToDevice));

			const int blockSize = 512;
			const int elementsPerBlock = 8 * blockSize;
			const int gridSize = (size + elementsPerBlock - 1) / elementsPerBlock;
			reduceMaxKernel << <gridSize, blockSize >> > (d_data, size, d_customMaxX);
			CHECK_CUDA_KERNEL_ERROR();
			CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	
			float  h_customMaxX;
			CHECK_CUDA(cudaMemcpy(&h_customMaxX, d_customMaxX, sizeof(float), cudaMemcpyDeviceToHost));
			CHECK_CUDA(cudaFree(d_customMaxX));
			return h_customMaxX;
		}
	}
}
