#include "pcl_cuda/common/common.h"
#include"internal/device_primitives.cuh"
#include "internal/error.cuh"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include <thrust/count.h>
namespace pcl {
    namespace cuda {
      
        __global__ void covarianceReduceKernel(
            const float* __restrict__ d_in_x,
            const float* __restrict__ d_in_y,
            const float* __restrict__ d_in_z, // 输入点云
            int numPoints,                             // 点的数量
            float3 ref_point,
            float* d_out    // 全局累加缓冲区 (大小为 9)
        ) {
            // 9个累加器：
            // [0-2]: sum_x, sum_y, sum_z (用于计算均值)
            // [3-8]: sum_xx, sum_xy, sum_xz, sum_yy, sum_yz, sum_zz (用于计算协方差)
            float local_sums[9] = { 0.0f };

            const int tx = threadIdx.x;
            const int tid = tx + 8 * blockDim.x * blockIdx.x;
#pragma unroll
            for (int i = 0; i < 8; i++)
            {
                int idx = tid + i * blockDim.x;
                if (idx < numPoints)  // 仅累加有效元素，无效则加0（不break，避免分支分歧）
                {
                    float x = d_in_x[idx] - ref_point.x;
                    float y = d_in_y[idx] - ref_point.y;
                    float z = d_in_z[idx] - ref_point.z;
                    local_sums[0] += x;
                    local_sums[1] += y;
                    local_sums[2] += z;
                    // 累加二阶矩 (Outer Product Sum)，只算上三角
                    local_sums[3] += x * x;
                    local_sums[4] += x * y;
                    local_sums[5] += x * z;
                    local_sums[6] += y * y;
                    local_sums[7] += y * z;
                    local_sums[8] += z * z;
                }
            }
            // 对这9个值分别进行 Warp 归约
            for (int k = 0; k < 9; k++) {
                local_sums[k] = warpSum(local_sums[k]);
            }

            const int warpId = tx / 32;
            const int laneId = tx % 32;
            const int warpNums = (blockDim.x + 31) / 32;
            __shared__ float sm_sum[32][9];
            if (laneId == 0)
            {
                for (int k = 0; k < 9; k++) {
                    sm_sum[warpId][k] = local_sums[k];
                }

            }
            __syncthreads();
            if (warpId == 0)
            {
                for (int k = 0; k < 9; k++) {
                    local_sums[k] = (laneId < warpNums ? sm_sum[laneId][k] : 0.0);
                    local_sums[k] = warpSum(local_sums[k]);
                }
            }
            if (tx == 0)
            {
                for (int k = 0; k < 9; k++) {
                    atomicAdd(&d_out[k], local_sums[k]);
                }
            }
        }

        void computeCentroidAndCovariance(const GpuPointCloud& cloud_in, Eigen::Matrix3f& covariance_, PointXYZ& centoird_)
        {
            int size = cloud_in.size();
            if (size == 0 ) {
                std::cerr << "[Warning] computeCentroidAndCovariance received empty cloud!" << std::endl;
                return;
            }
            float* d_sum;
            CHECK_CUDA(cudaMalloc(&d_sum, sizeof(float) * 9));
            CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(float) * 9));

            float h_ref_x, h_ref_y, h_ref_z;
            CHECK_CUDA(cudaMemcpy(&h_ref_x, cloud_in.x(), sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(&h_ref_y, cloud_in.y(), sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(&h_ref_z, cloud_in.z(), sizeof(float), cudaMemcpyDeviceToHost));
            float3 ref_point = make_float3(h_ref_x, h_ref_y, h_ref_z);

            const int blockSize = 512;
            const int elementsPerBlock = 8 * blockSize;
            const int gridSize = (size + elementsPerBlock - 1) / elementsPerBlock;         
            covarianceReduceKernel << <gridSize, blockSize >> > (cloud_in.x(), cloud_in.y(), cloud_in.z(),
                size, ref_point, d_sum);
            CHECK_CUDA_KERNEL_ERROR();
            CHECK_CUDA(cudaDeviceSynchronize());
           
            float h_sums[9];
            CHECK_CUDA(cudaMemcpy(h_sums, d_sum, sizeof(float) * 9, cudaMemcpyDeviceToHost));

            //  Finalize: 利用数学公式计算最终结果 (CPU端计算，量极小)
            // 对应公式: Cov = (S2 / N) - mu' * mu'^T
            float inv_N = 1.0f / size;

            // 恢复相对均值 mu'
            float mu_prime_x = h_sums[0] * inv_N;
            float mu_prime_y = h_sums[1] * inv_N;
            float mu_prime_z = h_sums[2] * inv_N;

            centoird_.x = mu_prime_x + ref_point.x;
            centoird_.y = mu_prime_y + ref_point.y;
            centoird_.z = mu_prime_z + ref_point.z;
            // 构造相对二阶矩矩阵 (S2 / N) 并减去 (mu' * mu'^T)
            // 这里的 h_sums[3..8] 存储的是 sum((p-ref)*(p-ref)^T)
            // XX
            covariance_(0, 0) = (h_sums[3] * inv_N) - (mu_prime_x * mu_prime_x);
            // XY
            covariance_(0, 1) = (h_sums[4] * inv_N) - (mu_prime_x * mu_prime_y);
            covariance_(1, 0) = covariance_(0, 1); // 对称
            // XZ
            covariance_(0, 2) = (h_sums[5] * inv_N) - (mu_prime_x * mu_prime_z);
            covariance_(2, 0) = covariance_(0, 2); // 对称
            // YY
            covariance_(1, 1) = (h_sums[6] * inv_N) - (mu_prime_y * mu_prime_y);
            // YZ
            covariance_(1, 2) = (h_sums[7] * inv_N) - (mu_prime_y * mu_prime_z);
            covariance_(2, 1) = covariance_(1, 2); // 对称
            // ZZ
            covariance_(2, 2) = (h_sums[8] * inv_N) - (mu_prime_z * mu_prime_z);
            CHECK_CUDA(cudaFree(d_sum));
        }
 
    } // namespace cuda
} // namespace pcl