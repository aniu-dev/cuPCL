#include "statistical_outlier_removal.cuh"
#include "pcl_cuda/common/common.h"
#include "lbvh/lbvh.cuh"
#include "internal/cuda_math.cuh"
#include "internal/error.cuh"
#include <thrust/gather.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>
namespace pcl
{
    namespace cuda
    {
        namespace device
        {

            __host__ __device__ float distSqPointAABB2(float3 p, const AABB& aabb) {
                float dx = ::fmaxf(0.0f, ::fmaxf(aabb.min.x - p.x, p.x - aabb.max.x));
                float dy = ::fmaxf(0.0f, ::fmaxf(aabb.min.y - p.y, p.y - aabb.max.y));
                float dz = ::fmaxf(0.0f, ::fmaxf(aabb.min.z - p.z, p.z - aabb.max.z));
                return dx * dx + dy * dy + dz * dz;
            }

            __global__ void gatherPointsKernel2(
                const uint32_t* __restrict__ indices,
                const float* __restrict__ x_in,
                const float* __restrict__ y_in,
                const float* __restrict__ z_in,
                float* x_out,
                float* y_out,
                float* z_out,
                int numPoints)
            {
                int tx = blockIdx.x * blockDim.x + threadIdx.x;

                if (tx >= numPoints) return;

                int old_idx = indices[tx];
                x_out[tx] = x_in[old_idx];
                y_out[tx] = y_in[old_idx];
                z_out[tx] = z_in[old_idx];

            }
            // 只存距离的精简版优先队列
            template <int MAX_K>
            struct PriorityQueueSOR {
                float dists[MAX_K];

                __device__ __forceinline__ PriorityQueueSOR()
                {
                    #pragma unroll
                    for (int i = 0; i < MAX_K; ++i) dists[i] = 1e38f;
                }

                // 插入并返回当前第 K 大的距离（用于剪枝）
                __device__ __forceinline__ float insert(float dist, int k_limit) {
                    if (dist >= dists[k_limit - 1]) return dists[k_limit - 1];

                    int i = k_limit - 1;
#pragma unroll
                    while (i > 0) {
                        if (dists[i - 1] > dist) {
                            dists[i] = dists[i - 1];
                            i--;
                        }
                        else {
                            break;
                        }
                    }
                    dists[i] = dist;
                    return dists[k_limit - 1];
                }

                // 计算 K 个邻居的平均距离 (Standard PCL Logic)
                __device__ __forceinline__ float compute_mean_distance(int k_limit) {
                    double  sum = 0.0f;
#pragma unroll
                    for (int i = 0; i < MAX_K; ++i) {
                        // 过滤掉初始化的无穷大值（防止邻居不足 K 个的情况）
                        if (i < k_limit)
                        {
                            if (dists[i] < 1e30f)
                                sum +=  sqrtf((double)dists[i]); // PCL 使用的是开方后的欧氏距离
                        }

                    }
                    return (float)(sum / (double)k_limit);
                }
            };

            // ==========================================
            //  融合 KNN 与 平均距离计算
            // ==========================================
            template <int MAX_K>
            __global__ void computeMeanDistancesKernel(
                const LBVH::Node* __restrict__ nodes,
                const float* __restrict__ sorted_x,
                const float* __restrict__ sorted_y,
                const float* __restrict__ sorted_z,
                int num_points,
                int nr_k,
                float* __restrict__ out_mean_dists
            ) {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid >= num_points) return;

                // 读取当前点 (Sorted Order)
                float3 q = make_float3(sorted_x[tid], sorted_y[tid], sorted_z[tid]);

                PriorityQueueSOR<MAX_K> queue;
                float search_radius_sq = 1e38f;

                // 栈遍历
                int stack[32];
                int stack_top = 0;
                stack[stack_top++] = 0;

                while (stack_top > 0) {
                    int node_idx = stack[--stack_top];
                    const LBVH::Node& node = nodes[node_idx];

                    int left_idx = node.leftIdx;
                    int right_idx = node.rightIdx;
                    bool is_leaf_l = (left_idx & 0x80000000);
                    bool is_leaf_r = (right_idx & 0x80000000);
                    int idx_l = left_idx & 0x7FFFFFFF;
                    int idx_r = right_idx & 0x7FFFFFFF;

                    float dist_l = distSqPointAABB2(q, node.bounds[0]);
                    float dist_r = distSqPointAABB2(q, node.bounds[1]);

                    // 遍历左孩子
                    if (dist_l < search_radius_sq) {
                        if (is_leaf_l) {

                            //// 剔除 0 距离，（符合 PCL 定义）
                            if (dist_l > 1e-12f) {
                                search_radius_sq = queue.insert(dist_l, nr_k);
                            }
                            
                            dist_l = 1e38f; // 标记已处理
                        }
                    }
                    else { dist_l = 1e38f; } // 剪枝

                    // 遍历右孩子
                    if (dist_r < search_radius_sq) {
                        if (is_leaf_r) {
                          
                            if (dist_r > 1e-12f) { // 排除掉自己（距离接近 0 的点）
                                search_radius_sq = queue.insert(dist_r, nr_k);
                            }
                            dist_r = 1e38f;
                        }
                    }
                    else { dist_r = 1e38f; }

                    // 压栈逻辑：先压远的，后压近的 (DFS Closest-First)
                    bool tr_l = (dist_l < 1e38f);
                    bool tr_r = (dist_r < 1e38f);

                    if (tr_l && tr_r) {
                        if (dist_l < dist_r) {
                            stack[stack_top++] = idx_r;
                            stack[stack_top++] = idx_l;
                        }
                        else {
                            stack[stack_top++] = idx_l;
                            stack[stack_top++] = idx_r;
                        }
                    }
                    else if (tr_l) {
                        stack[stack_top++] = idx_l;
                    }
                    else if (tr_r) {
                        stack[stack_top++] = idx_r;
                    }
                }

                // 直接输出平均距离
                out_mean_dists[tid] = queue.compute_mean_distance(nr_k);
            }

            // ==========================================
            //  生成 Mask (严格单边判定)
            // ==========================================
            __global__ void generateSORMaskKernel(
                const float* __restrict__ mean_dists,
                int num_points,
                float threshold,
                int* __restrict__ mask
            ) {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid >= num_points) return;

                // 【关键逻辑】
                // 只有当距离 大于 阈值时，才会被剔除。
                // 距离特别小（密集区域）会被安全保留。
                mask[tid] = (mean_dists[tid] <= threshold) ? 1 : 0;
            }

            // Thrust 算子：计算方差
            struct VarianceOp {
                float mean;
                VarianceOp(float m) : mean(m) {}
                __host__ __device__ float operator()(float val) const {
                    return (val - mean) * (val - mean);
                }
            };

            void launchStatisticalOutlierRemovalFilter(
                const GpuPointCloud* input,
                const int& nr_k,
                const float& stddev_mult,
                GpuPointCloud& output
            )
            {

                int size = input->size();
                if (size == 0) {

                    std::cerr << "[Warning] StatisticalOutlierRemovalFilter received empty cloud!" << std::endl;
                    return;
                }
                pcl::cuda::LBVH bvh;
                bvh.compute(input, size);
                // 需要分别 gather x, y, z
                // 这样做虽然多了几次内存操作，但 SOA 的合并访存优势通常能弥补回来
                thrust::device_vector<float> d_sorted_x(size);
                thrust::device_vector<float> d_sorted_y(size);
                thrust::device_vector<float> d_sorted_z(size);

                const uint32_t* sorted_indices = bvh.getSortedIndices();
                const int blockSize = 256;
                const int gridSize = (size + blockSize - 1) / blockSize;
                gatherPointsKernel2 << <gridSize, blockSize >> > (
                    sorted_indices,
                    input->x(), input->y(), input->z(),
                    thrust::raw_pointer_cast(d_sorted_x.data()), // 传入排序后的 X
                    thrust::raw_pointer_cast(d_sorted_y.data()), // 传入排序后的 Y
                    thrust::raw_pointer_cast(d_sorted_z.data()), // 传入排序后的 Z
                    size
                    );
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	

                // 计算 KNN 平均距离
                thrust::device_vector<float> d_mean_dists(size);

                // 针对常用 K 值优化，避免寄存器溢出
                if (nr_k <= 32) {
                    computeMeanDistancesKernel<32> << <gridSize, 256 >> > (
                        bvh.getNodes(),
                        thrust::raw_pointer_cast(d_sorted_x.data()),
                        thrust::raw_pointer_cast(d_sorted_y.data()),
                        thrust::raw_pointer_cast(d_sorted_z.data()),
                        size,
                        nr_k,
                        thrust::raw_pointer_cast(d_mean_dists.data())
                        );
                }
                else if (nr_k > 32 && nr_k <= 64) {
                    computeMeanDistancesKernel<64> << <gridSize, 128 >> > (
                        bvh.getNodes(),
                        thrust::raw_pointer_cast(d_sorted_x.data()),
                        thrust::raw_pointer_cast(d_sorted_y.data()),
                        thrust::raw_pointer_cast(d_sorted_z.data()),
                        size,
                        nr_k,
                        thrust::raw_pointer_cast(d_mean_dists.data())
                        );
                }
                else if (nr_k > 64 && nr_k <= 128) {
                    computeMeanDistancesKernel<128> << <gridSize, 64 >> > ( // 减小 BlockSize 应对寄存器压力
                        bvh.getNodes(),
                        thrust::raw_pointer_cast(d_sorted_x.data()),
                        thrust::raw_pointer_cast(d_sorted_y.data()),
                        thrust::raw_pointer_cast(d_sorted_z.data()),
                        size,
                        nr_k,
                        thrust::raw_pointer_cast(d_mean_dists.data())
                        );
                }
                else {
                    std::cerr << "[SOR Error] K > 100 is not optimized on GPU." << std::endl;
                    return;
                }
                CHECK_CUDA_KERNEL_ERROR();

                // Thrust 全局统计 (均值 & 标准差)
            
                double  total_sum = thrust::reduce(d_mean_dists.begin(), d_mean_dists.end(), 0.0, thrust::plus<double >());
                double  global_mean = total_sum / size;

                //  Variance
                double  total_variance = thrust::transform_reduce(
                    d_mean_dists.begin(),
                    d_mean_dists.end(),
                    VarianceOp(global_mean),
                    0.0,
                    thrust::plus<double >()
                );
                double  global_stddev = sqrtf(total_variance / size);

                //计算单边阈值
                float distance_threshold = global_mean + stddev_mult * global_stddev;

                //生成 Mask 并过滤
                int* d_mask_ptr;
                cudaMalloc(&d_mask_ptr, size * sizeof(int));

                generateSORMaskKernel << <gridSize, blockSize >> > (
                    thrust::raw_pointer_cast(d_mean_dists.data()),
                    size,
                    distance_threshold,
                    d_mask_ptr
                    );
                CHECK_CUDA_KERNEL_ERROR();

                // 拷贝有效点 (Source 必须是 Sorted 的数据)
                copyPointCloud(
                    thrust::raw_pointer_cast(d_sorted_x.data()),
                    thrust::raw_pointer_cast(d_sorted_y.data()),
                    thrust::raw_pointer_cast(d_sorted_z.data()),
                    size,
                    output,
                    d_mask_ptr
                );

                cudaFree(d_mask_ptr);

            };
        }
    }
}