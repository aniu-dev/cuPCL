#include "pcl_cuda/common/primitives.h"
#include"internal/device_primitives.cuh"
#include "internal/error.cuh"
#include <cuda_runtime.h>

namespace pcl {
	namespace cuda {
		__global__ void reduceMinMax3DKernel(const float* __restrict__ d_x_in, const float* __restrict__ d_y_in, 
			const float* __restrict__ d_z_in, int num_points,
			float* d_out_min, float* d_out_max)
		{
			const int tx = threadIdx.x;
			const int tid = tx + 8 * blockIdx.x * blockDim.x;
			float3 min_temp = make_float3(INFINITY, INFINITY, INFINITY);
			float3 max_temp = make_float3(-INFINITY, -INFINITY, -INFINITY);
#pragma unroll
			for (int i = 0; i < 8; i++)
			{
				int idx = tid + i * blockDim.x;
				if (idx < num_points)
				{
					float x_in_temp = d_x_in[idx];
					float y_in_temp = d_y_in[idx];
					float z_in_temp = d_z_in[idx];
					min_temp.x = fmin(min_temp.x, x_in_temp);
					min_temp.y = fmin(min_temp.y, y_in_temp);
					min_temp.z = fmin(min_temp.z, z_in_temp);

					max_temp.x = fmax(max_temp.x, x_in_temp);
					max_temp.y = fmax(max_temp.y, y_in_temp);
					max_temp.z = fmax(max_temp.z, z_in_temp);

				}

			}
			__syncthreads();
			min_temp.x = warpMin(min_temp.x);
			max_temp.x = warpMax(max_temp.x);

			min_temp.y = warpMin(min_temp.y);
			max_temp.y = warpMax(max_temp.y);

			min_temp.z = warpMin(min_temp.z);
			max_temp.z = warpMax(max_temp.z);


			const int warpId = tx / 32;
			const int laneId = tx % 32;
			const int warpNum = blockDim.x / 32;
			__shared__ float3 sx_min[32];
			__shared__ float3 sx_max[32];

			if (laneId == 0)
			{
				sx_min[warpId] = min_temp;
				sx_max[warpId] = max_temp;

			}
			__syncthreads();
			if (warpId == 0)
			{
				min_temp = (laneId < warpNum) ? sx_min[laneId] : make_float3(INFINITY, INFINITY, INFINITY);
				max_temp = (laneId < warpNum) ? sx_max[laneId] : make_float3(-INFINITY, -INFINITY, -INFINITY);

				min_temp.x = warpMin(min_temp.x);
				max_temp.x = warpMax(max_temp.x);

				min_temp.y = warpMin(min_temp.y);
				max_temp.y = warpMax(max_temp.y);

				min_temp.z = warpMin(min_temp.z);
				max_temp.z = warpMax(max_temp.z);
			}

			if (tx == 0)
			{
				atomicMinFloat(&d_out_min[0], min_temp.x);
				atomicMinFloat(&d_out_min[1], min_temp.y);
				atomicMinFloat(&d_out_min[2], min_temp.z);
				atomicMaxFloat(&d_out_max[0], max_temp.x);
				atomicMaxFloat(&d_out_max[1], max_temp.y);
				atomicMaxFloat(&d_out_max[2], max_temp.z);
			}

		}
		// 示例：规约求最小值最大值
		void reduceMinMax3D(const GpuPointCloud& cloud, size_t size, PointXYZ& minVal, PointXYZ& maxVal)
		{
			if (size == 0) {

				std::cerr << "[Warning] reduceMinMax3D received empty cloud!" << std::endl;
				return;
			}

			float* d_buffer;
			CHECK_CUDA(cudaMalloc(&d_buffer, 6 * sizeof(float)));

			float h_init[6] = { FLT_MAX,FLT_MAX ,FLT_MAX ,-FLT_MAX ,-FLT_MAX ,-FLT_MAX };

			CHECK_CUDA(cudaMemcpy(d_buffer, h_init, 6 * sizeof(float), cudaMemcpyHostToDevice));

			const int blockSize = 256;
			const int elementsPerBlock = 8 * blockSize;
			const int gridSize = (size + elementsPerBlock - 1) / elementsPerBlock;

			reduceMinMax3DKernel << <gridSize, blockSize >> > (cloud.x(), cloud.y(), cloud.z(), size, d_buffer, d_buffer + 3);
			CHECK_CUDA_KERNEL_ERROR();
			CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	

			float h_res[6];
			CHECK_CUDA(cudaMemcpy(h_res, d_buffer, 6 * sizeof(float), cudaMemcpyDeviceToHost));
			minVal.x = h_res[0];minVal.y = h_res[1];minVal.z = h_res[2];
			maxVal.x = h_res[3]; maxVal.y = h_res[4]; maxVal.z = h_res[5];
			CHECK_CUDA(cudaFree(d_buffer));

		}
	}
}
