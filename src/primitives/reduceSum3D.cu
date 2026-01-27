#include "pcl_cuda/common/primitives.h"
#include"internal/device_primitives.cuh"
#include "internal/error.cuh"
#include <cuda_runtime.h>

namespace pcl {
	namespace cuda {


		__global__ void reduceSumWarp3DKernel(const float* __restrict__ d_in_x, const float* __restrict__ d_in_y,
			const float* __restrict__ d_in_z,
			const int numPoints, float* d_out)
		{
			const int tx = threadIdx.x;
			const int tid = tx + 8 * blockDim.x * blockIdx.x;
			float3 sum = make_float3(0.0, 0.0, 0.0);
			if (tid >= numPoints) return;
#pragma unroll
			for (int i = 0; i < 8; i++)
			{
				int idx = tid + i * blockDim.x;
				if (idx < numPoints)  // 仅累加有效元素，无效则加0（不break，避免分支分歧）
				{
					sum.x += d_in_x[idx];
					sum.y += d_in_y[idx];
					sum.z += d_in_z[idx];
				}
			}
			sum.x = warpSum(sum.x);
			sum.y = warpSum(sum.y);
			sum.z = warpSum(sum.z);
			const int warpId = tx / 32;
			const int laneId = tx % 32;
			const int warpNums = (blockDim.x + 31) / 32;
			__shared__ float3 sm_sum[32];
			if (laneId == 0)
			{
				sm_sum[warpId] = sum;
			}
			__syncthreads();

			if (warpId == 0)
			{
				sum = (laneId < warpNums ? sm_sum[laneId] : make_float3(0.0, 0.0, 0.0));
				sum.x = warpSum(sum.x);
				sum.y = warpSum(sum.y);
				sum.z = warpSum(sum.z);
			}
			if (tx == 0)
			{
				atomicAdd(&d_out[0], sum.x);
				atomicAdd(&d_out[1], sum.y);
				atomicAdd(&d_out[2], sum.z);
			}

		}
		void reduceWarpSum3D(const GpuPointCloud& cloud, size_t size, PointXYZ& Val)
		{
			if (size == 0) {

				std::cerr << "[Warning] reduceWarpSum3D received empty cloud!" << std::endl;
				return;
			}
			float* d_sum;
			CHECK_CUDA(cudaMalloc(&d_sum, 3 * sizeof(float)));
			CHECK_CUDA(cudaMemset(d_sum, 0, 3 * sizeof(float)));
			const int blockSize = 512;
			const int elementsPerBlock = 8 * blockSize;
			const int gridSize = (size + elementsPerBlock - 1) / elementsPerBlock;
			reduceSumWarp3DKernel << <gridSize, blockSize >> > (cloud.x(), cloud.y(), cloud.z(), size, d_sum);
			CHECK_CUDA_KERNEL_ERROR();
			CHECK_CUDA(cudaDeviceSynchronize());
			float h_sum[3];
			CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, 3 * sizeof(float), cudaMemcpyDeviceToHost));
			Val.x = h_sum[0]; Val.y = h_sum[1]; Val.z = h_sum[2];
			CHECK_CUDA(cudaFree(d_sum));

		}
	}
}
