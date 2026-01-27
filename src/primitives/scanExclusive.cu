#include "pcl_cuda/common/primitives.h"
#include <device_launch_parameters.h>
#include "internal/error.cuh"
#include <cuda_runtime.h>

namespace pcl {
	namespace cuda {
#define CONFLICT_FREE_OFFSET(n) ( (n) >> 5)
		__global__ void inBlcokScanKernel(const int* __restrict__ d_in, int* d_out, int* d_block_sums, int len, int shmem_sz, int max_elems_per_block)
		{
			extern __shared__ int s_out[];

			int tid = threadIdx.x;
			int ai = tid;
			int bi = tid + blockDim.x;
			s_out[tid] = 0;
			s_out[tid + blockDim.x] = 0;
			if (tid + max_elems_per_block < shmem_sz)
			{
				s_out[tid + max_elems_per_block] = 0;
			}
			__syncthreads();

			int copy_idx = max_elems_per_block * blockIdx.x + tid;
			if (copy_idx < len)
			{
				s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[copy_idx];
				if (copy_idx + blockDim.x < len)
				{
					s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[copy_idx + blockDim.x];
				}
			}
			__syncthreads();
			//向上规约
			int offset = 1;
			for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
			{
				__syncthreads();
				if (tid < d)
				{
					int left = offset * (tid * 2 + 1) - 1;
					int right = offset * (tid * 2 + 2) - 1;
					left += CONFLICT_FREE_OFFSET(left);
					right += CONFLICT_FREE_OFFSET(right);
					s_out[right] += s_out[left];
				}
				offset <<= 1;
			}

			if (tid == 0)
			{
				int root_idx = (max_elems_per_block - 1) + CONFLICT_FREE_OFFSET(max_elems_per_block - 1);
				d_block_sums[blockIdx.x] = s_out[root_idx];
				s_out[root_idx] = 0;
			}

			for (int d = 1; d < max_elems_per_block; d <<= 1)
			{
				offset >>= 1;

				__syncthreads();

				if (tid < d)
				{
					int left = offset * (tid * 2 + 1) - 1;
					int right = offset * (tid * 2 + 2) - 1;
					left += CONFLICT_FREE_OFFSET(left);
					right += CONFLICT_FREE_OFFSET(right);
					int temp = s_out[left];
					s_out[left] = s_out[right];
					s_out[right] += temp;
				}
			}

			if (copy_idx < len)
			{
				d_out[copy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
				if (copy_idx + blockDim.x < len)
				{
					d_out[copy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
				}
			}
		}
		__global__ void addBlockSumKernel(int* d_out, const int* d_in, int* d_block_sums, const int len)
		{
			unsigned int d_block_sum_val = d_block_sums[blockIdx.x];
			unsigned int copy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
			if (copy_idx < len)
			{
				// 处理第一个元素：d_in[cpy_idx]是块内前缀和，加上前面块的总和得到全局前缀和
				d_out[copy_idx] = d_in[copy_idx] + d_block_sum_val;
				// 边界检查：若第二个元素（cpy_idx+blockDim.x）也在范围内，同样更新
				if (copy_idx + blockDim.x < len)
					d_out[copy_idx + blockDim.x] = d_in[copy_idx + blockDim.x] + d_block_sum_val;
			}

		}
		// 排他前缀和
		void scanExclusive(const int* d_data, int* d_out, size_t size)
		{
			if (size == 0) {

				std::cerr << "[Warning] scanExclusive received empty cloud!" << std::endl;
				return;
			}
			CHECK_CUDA(cudaMemset(d_out, 0, size * sizeof(int)));
			unsigned int blcokSize = 512;
			unsigned int max_elems_per_block = 2 * blcokSize;
			unsigned int gridSize = (size + max_elems_per_block - 1) / max_elems_per_block;
			unsigned int shmemSize = max_elems_per_block + ((max_elems_per_block - 1) >> 5);

			int* d_block_sums;

			CHECK_CUDA(cudaMalloc(&d_block_sums, sizeof(int) * gridSize));
			CHECK_CUDA(cudaMemset(d_block_sums, 0, sizeof(int) * gridSize));

			inBlcokScanKernel << <gridSize, blcokSize, sizeof(unsigned int)* shmemSize >> > (
				d_data,           // 输入：原始数据
				d_out,          // 输出：块内排他前缀和
				d_block_sums,   // 输出：每个块的总累加和
				size,       // 总元素数
				shmemSize,       // 共享内存大小
				max_elems_per_block  // 每块最大元素数
				);
			CHECK_CUDA_KERNEL_ERROR();
			CHECK_CUDA(cudaDeviceSynchronize());


			if (gridSize <= max_elems_per_block)
			{

				int* d_dummy_blocks_sums;
				CHECK_CUDA(cudaMalloc(&d_dummy_blocks_sums, sizeof(unsigned int)));
				CHECK_CUDA(cudaMemset(d_dummy_blocks_sums, 0, sizeof(unsigned int)));
				inBlcokScanKernel << <1, blcokSize, sizeof(unsigned int)* shmemSize >> > (
					d_block_sums,       // 输出：块和的前缀和
					d_block_sums,       // 输入：原始块和数组
					d_dummy_blocks_sums,// 输出：虚拟块和（无用，仅满足函数参数）
					gridSize,            // 块和数组的长度（即块数）
					shmemSize,           // 共享内存大小
					max_elems_per_block // 每块最大元素数
					);
				CHECK_CUDA_KERNEL_ERROR();
				
				CHECK_CUDA(cudaDeviceSynchronize());
				CHECK_CUDA(cudaFree(d_dummy_blocks_sums));// 释放临时虚拟块和数组
			}
			else
			{

				int* d_in_block_sums;
				CHECK_CUDA(cudaMalloc(&d_in_block_sums, sizeof(unsigned int) * gridSize));

				CHECK_CUDA(cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(unsigned int) * gridSize, cudaMemcpyDeviceToDevice));
				scanExclusive(d_in_block_sums, d_block_sums, gridSize);
			
				CHECK_CUDA(cudaFree(d_in_block_sums));	// 释放临时数组
			}
			addBlockSumKernel << <gridSize, blcokSize >> > (
				d_out,          // 输出：全局排他前缀和
				d_out,          // 输入：块内前缀和（与输出数组相同）
				d_block_sums,   // 输入：块和的前缀和
				size        // 总元素数
				);
			CHECK_CUDA_KERNEL_ERROR();
			CHECK_CUDA(cudaDeviceSynchronize());			
			CHECK_CUDA(cudaFree(d_block_sums));// 释放块和数组全局内存
		}
	}
}
