#include "normal_3d.cuh"
#include "pcl_cuda/common/common.h"
#include"internal/device_primitives.cuh"
#include "lbvh/lbvh.cuh"
#include "internal/cuda_math.cuh"
#include "internal/error.cuh"
#include <thrust/gather.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
namespace pcl 
{
    namespace cuda
    {

        namespace device
        {
            __host__ __device__ float distSqPointAABB4(float3 p, const AABB& aabb) {
                float dx = ::fmaxf(0.0f, ::fmaxf(aabb.min.x - p.x, p.x - aabb.max.x));
                float dy = ::fmaxf(0.0f, ::fmaxf(aabb.min.y - p.y, p.y - aabb.max.y));
                float dz = ::fmaxf(0.0f, ::fmaxf(aabb.min.z - p.z, p.z - aabb.max.z));
                return dx * dx + dy * dy + dz * dz;
            }
            __device__ __forceinline__ float rsqrtf_fast(float x) { return 1.0f / sqrtf(x); }

            // 返回 float4: x, y, z 为法向量, w 为曲率 (Curvature)
            __device__ __forceinline__ float4 computeNormalEigenvector(float cov[6]) {
                // 初始化特征向量矩阵 V 为单位阵
                float v00 = 1.0f, v01 = 0.0f, v02 = 0.0f;
                float v10 = 0.0f, v11 = 1.0f, v12 = 0.0f;
                float v20 = 0.0f, v21 = 0.0f, v22 = 1.0f;

                //  读取协方差 (A)
                float a00 = cov[0], a01 = cov[1], a02 = cov[2];
                float a11 = cov[3], a12 = cov[4], a22 = cov[5];

                // Jacobi 迭代 (6次足够收敛)
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    // 消除 (0,1)
                    if (fabsf(a01) > 1e-12f) {
                        float u = (a11 - a00) * 0.5f / a01;
                        float t = copysignf(1.0f / (fabsf(u) + sqrtf(1.0f + u * u)), u);
                        float c = 1.0f / sqrtf(1.0f + t * t);
                        float s = c * t;

                        float t00 = a00, t01 = a01, t11 = a11, t02 = a02, t12 = a12;
                        a00 = t00 - t * t01; a11 = t11 + t * t01; a01 = 0.0f;
                        a02 = c * t02 - s * t12; a12 = s * t02 + c * t12;

                        float tv00 = v00, tv10 = v10, tv20 = v20;
                        float tv01 = v01, tv11 = v11, tv21 = v21;
                        v00 = c * tv00 - s * tv01; v10 = c * tv10 - s * tv11; v20 = c * tv20 - s * tv21;
                        v01 = s * tv00 + c * tv01; v11 = s * tv10 + c * tv11; v21 = s * tv20 + c * tv21;
                    }
                    // 消除 (0,2)
                    if (fabsf(a02) > 1e-12f) {
                        float u = (a22 - a00) * 0.5f / a02;
                        float t = copysignf(1.0f / (fabsf(u) + sqrtf(1.0f + u * u)), u);
                        float c = 1.0f / sqrtf(1.0f + t * t);
                        float s = c * t;

                        float t00 = a00, t02 = a02, t22 = a22, t01 = a01, t12 = a12;
                        a00 = t00 - t * t02; a22 = t22 + t * t02; a02 = 0.0f;
                        a01 = c * t01 - s * t12; a12 = s * t01 + c * t12;

                        float tv00 = v00, tv10 = v10, tv20 = v20;
                        float tv02 = v02, tv12 = v12, tv22 = v22;
                        v00 = c * tv00 - s * tv02; v10 = c * tv10 - s * tv12; v20 = c * tv20 - s * tv22;
                        v02 = s * tv00 + c * tv02; v12 = s * tv10 + c * tv12; v22 = s * tv20 + c * tv22;
                    }
                    // 消除 (1,2)
                    if (fabsf(a12) > 1e-12f) {
                        float u = (a22 - a11) * 0.5f / a12;
                        float t = copysignf(1.0f / (fabsf(u) + sqrtf(1.0f + u * u)), u);
                        float c = 1.0f / sqrtf(1.0f + t * t);
                        float s = c * t;

                        float t11 = a11, t12 = a12, t22 = a22, t01 = a01, t02 = a02;
                        a11 = t11 - t * t12; a22 = t22 + t * t12; a12 = 0.0f;
                        a01 = c * t01 - s * t02; a02 = s * t01 + c * t02;

                        float tv01 = v01, tv11 = v11, tv21 = v21;
                        float tv02 = v02, tv12 = v12, tv22 = v22;
                        v01 = c * tv01 - s * tv02; v11 = c * tv11 - s * tv12; v21 = c * tv21 - s * tv22;
                        v02 = s * tv01 + c * tv02; v12 = s * tv11 + c * tv12; v22 = s * tv21 + c * tv22;
                    }
                }

                // 找最小特征值对应的特征向量
                float min_lambda = a00;
                float3 normal = make_float3(v00, v10, v20);

                // 比较找最小
                if (a11 < min_lambda) {
                    min_lambda = a11;
                    normal = make_float3(v01, v11, v21);
                }
                if (a22 < min_lambda) {
                    min_lambda = a22;
                    normal = make_float3(v02, v12, v22);
                }

                // 归一化并计算曲率
                // 默认返回值 (退化情况：法线向上，曲率0)
                float4 r_normal = make_float4(0.0f, 0.0f, 1.0f, 0.0f);

                float lenSq = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;

                if (lenSq > 1e-12f) {
                    float invLen = rsqrtf_fast(lenSq);
                    normal.x *= invLen;
                    normal.y *= invLen;
                    normal.z *= invLen;

                    // 计算曲率: lambda_0 / (lambda_0 + lambda_1 + lambda_2)
                    // 迭代后的 a00, a11, a22 即为特征值
                    float sum_eigenvalues = fabsf(a00) + fabsf(a11) + fabsf(a22);
                    float curvature = 0.0f;

                    if (sum_eigenvalues > 1e-12f) {
                        curvature = fabsf(min_lambda) / sum_eigenvalues;
                    }

                    r_normal = make_float4(normal.x, normal.y, normal.z, curvature);
                }

                return r_normal;
            }

            // Kernel 函数定义
            __global__ void gatherPointsKernel4(
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
            __global__ void scatterNormalsKernel(
                const uint32_t* __restrict__ sorted_indices, // LBVH 的索引映射表
                const float* __restrict__ nx_sorted,
                const float* __restrict__ ny_sorted,
                const float* __restrict__ nz_sorted,
                const float* __restrict__ c_sorted,
                float* __restrict__ nx_out,
                float* __restrict__ ny_out,
                float* __restrict__ nz_out,
                float* __restrict__ c_out,
                int num_points
            ) {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid >= num_points) return;

                // sorted_indices[tid] 保存了当前排序后的点在原始数组中的位置
                int original_idx = sorted_indices[tid];

                nx_out[original_idx] = nx_sorted[tid];
                ny_out[original_idx] = ny_sorted[tid];
                nz_out[original_idx] = nz_sorted[tid];
                c_out[original_idx] = c_sorted[tid];
            }
            template <int MAX_K>
            struct PriorityQueueNormal {
                float dists[MAX_K];
                int   indices[MAX_K];
                __device__ __forceinline__ PriorityQueueNormal()
                {
#pragma unroll
                    for (int i = 0; i < MAX_K; ++i)
                    {
                        dists[i] = 1e38f;
                        indices[i] = -1;
                    }
                }

                // 插入并返回当前第 K 大的距离（用于剪枝）
                __device__ __forceinline__ float insert(float dist, int index, int k_limit) {
                    if (dist >= dists[k_limit - 1]) return dists[k_limit - 1];

                    int i = k_limit - 1;
#pragma unroll
                    while (i > 0) {
                        if (dists[i - 1] > dist) {
                            dists[i] = dists[i - 1];
                            indices[i] = indices[i - 1]; // 同步移动索引
                            i--;
                        }
                        else {
                            break;
                        }
                    }
                    dists[i] = dist;
                    indices[i] = index; // 插入索引
                    return dists[k_limit - 1];
                }
            };
                // ==========================================
                //  融合 KNN 与 平均距离计算
                // ==========================================
                template <int MAX_K>
                __global__ void computeNormalKernel(
                    const LBVH::Node* __restrict__ nodes,
                    const float* __restrict__ sorted_x,
                    const float* __restrict__ sorted_y,
                    const float* __restrict__ sorted_z,
                    int num_points,
                    int nr_k,
                    float* __restrict__ out_normal_x,
                    float* __restrict__ out_normal_y,
                    float* __restrict__ out_normal_z,
                    float* __restrict__ out_curvature
                ) {
                    int tid = blockIdx.x * blockDim.x + threadIdx.x;
                    if (tid >= num_points) return;

                    // 读取当前点 (Sorted Order)
                    float3 q = make_float3(sorted_x[tid], sorted_y[tid], sorted_z[tid]);

                    PriorityQueueNormal<MAX_K> queue;
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

                        float dist_l = distSqPointAABB4(q, node.bounds[0]);
                        float dist_r = distSqPointAABB4(q, node.bounds[1]);

                        // 遍历左孩子
                        if (dist_l < search_radius_sq) {
                            if (is_leaf_l) {
                                search_radius_sq = queue.insert(dist_l, idx_l, nr_k);
                                dist_l = 1e38f; // 标记已处理
                            }
                        }
                        else { dist_l = 1e38f; } // 剪枝

                        // 遍历右孩子
                        if (dist_r < search_radius_sq) {
                            if (is_leaf_r) {
                                search_radius_sq = queue.insert(dist_r, idx_r, nr_k);
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

                    // 计算质心 (Centroid)
                    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
                    int count = 0;

#pragma unroll
                    for (int i = 0; i < MAX_K; ++i) {
                        int idx = queue.indices[i];
                        if (i < nr_k && idx != -1) {
                            sum_x += sorted_x[idx];
                            sum_y += sorted_y[idx];
                            sum_z += sorted_z[idx];
                            count++;
                        }
                    }

                    // 如果邻居太少，无法计算平面
                    if (count < 3) {
                        out_normal_x[tid] = 0.0f; out_normal_y[tid] = 0.0f; out_normal_z[tid] = 1.0f; // 默认值
                        return;
                    }

                    float cx = (float)(sum_x / count);
                    float cy = (float)(sum_y / count);
                    float cz = (float)(sum_z / count);

                    // 计算协方差矩阵 (Covariance Matrix)
                    // 只存储上三角: xx, xy, xz, yy, yz, zz
                    float cov[6] = { 0.0f };

#pragma unroll
                    for (int i = 0; i < MAX_K; ++i) {
                        int idx = queue.indices[i];
                        if (i < nr_k && idx != -1) {
                            float dx = sorted_x[idx] - cx;
                            float dy = sorted_y[idx] - cy;
                            float dz = sorted_z[idx] - cz;

                            cov[0] += dx * dx;
                            cov[1] += dx * dy;
                            cov[2] += dx * dz;
                            cov[3] += dy * dy;
                            cov[4] += dy * dz;
                            cov[5] += dz * dz;
                        }
                    }

                    float4 result = computeNormalEigenvector(cov);

                    float nx = result.x;
                    float ny = result.y;
                    float nz = result.z;
                    float curvature = result.w;

                    // 视点一致性翻转

                    float dot_product = nx * q.x + ny * q.y + nz * q.z;

                    if (dot_product > 0.0f) {
                        nx = -nx;
                        ny = -ny;
                        nz = -nz;
                    }

                    // 结果写入全局内存
                    out_normal_x[tid] = nx;
                    out_normal_y[tid] = ny;
                    out_normal_z[tid] = nz;
                    out_curvature[tid] = curvature;
                }


                void launchNormalEstimation(
                    const GpuPointCloud* input,
                    const int nr_k,
                    GpuPointCloudNormal& output
                )
                {
                    int size = input->size();
                    if (size == 0) {

                        std::cerr << "[Warning] NormalEstimation received empty cloud!" << std::endl;
                        return;
                    }
                    pcl::cuda::LBVH bvh;
                    output.alloc(size);
                    bvh.compute(input, size);
                    // 需要分别 gather x, y, z
                    // 这样做虽然多了几次内存操作，但 SOA 的合并访存优势通常能弥补回来
                    thrust::device_vector<float> d_sorted_x(size);
                    thrust::device_vector<float> d_sorted_y(size);
                    thrust::device_vector<float> d_sorted_z(size);

                    const uint32_t* sorted_indices = bvh.getSortedIndices();
                    const int blockSize = 256;
                    const int gridSize = (size + blockSize - 1) / blockSize;
                    gatherPointsKernel4 << <gridSize, blockSize >> > (
                        sorted_indices,
                        input->x(), input->y(), input->z(),
                        thrust::raw_pointer_cast(d_sorted_x.data()), // 传入排序后的 X
                        thrust::raw_pointer_cast(d_sorted_y.data()), // 传入排序后的 Y
                        thrust::raw_pointer_cast(d_sorted_z.data()), // 传入排序后的 Z
                        size
                        );
                    CHECK_CUDA_KERNEL_ERROR();
                    CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	

                    // 准备临时显存：存放计算出的排序后的法向量
                    thrust::device_vector<float> d_sorted_nx(size);
                    thrust::device_vector<float> d_sorted_ny(size);
                    thrust::device_vector<float> d_sorted_nz(size);
                    thrust::device_vector<float> d_sorted_c(size);

                    // 针对常用 K 值优化，避免寄存器溢出
                    if (nr_k <= 32) {
                        computeNormalKernel<32> << <gridSize, 256 >> > (
                            bvh.getNodes(),
                            thrust::raw_pointer_cast(d_sorted_x.data()), // 传入排序后的 X
                            thrust::raw_pointer_cast(d_sorted_y.data()), // 传入排序后的 Y
                            thrust::raw_pointer_cast(d_sorted_z.data()), // 传入排序后的 Z
                            size,
                            nr_k,
                            thrust::raw_pointer_cast(d_sorted_nx.data()), // 传入排序后的 X
                            thrust::raw_pointer_cast(d_sorted_ny.data()), // 传入排序后的 Y
                            thrust::raw_pointer_cast(d_sorted_nz.data()), // 传入排序后的 Z
                            thrust::raw_pointer_cast(d_sorted_c.data()) // 传入排序后的 Z
                            );                    
                    }
                    else if (nr_k > 32 && nr_k <= 64) {
                        computeNormalKernel<64> << <gridSize, 128 >> > (
                            bvh.getNodes(),
                            thrust::raw_pointer_cast(d_sorted_x.data()),
                            thrust::raw_pointer_cast(d_sorted_y.data()),
                            thrust::raw_pointer_cast(d_sorted_z.data()),
                            size,
                            nr_k,
                            thrust::raw_pointer_cast(d_sorted_nx.data()), // 传入排序后的 X
                            thrust::raw_pointer_cast(d_sorted_ny.data()), // 传入排序后的 Y
                            thrust::raw_pointer_cast(d_sorted_nz.data()), // 传入排序后的 Z
                            thrust::raw_pointer_cast(d_sorted_c.data()) // 传入排序后的 Z
                            );
                    }
                    else if (nr_k > 64 && nr_k <= 128) {
                        computeNormalKernel<128> << <gridSize, 64 >> > ( // 减小 BlockSize 应对寄存器压力
                            bvh.getNodes(),
                            thrust::raw_pointer_cast(d_sorted_x.data()),
                            thrust::raw_pointer_cast(d_sorted_y.data()),
                            thrust::raw_pointer_cast(d_sorted_z.data()),
                            size,
                            nr_k,
                            thrust::raw_pointer_cast(d_sorted_nx.data()), // 传入排序后的 X
                            thrust::raw_pointer_cast(d_sorted_ny.data()), // 传入排序后的 Y
                            thrust::raw_pointer_cast(d_sorted_nz.data()), // 传入排序后的 Z
                            thrust::raw_pointer_cast(d_sorted_c.data()) // 传入排序后的 Z
                            );
                    }
                    else {
                        std::cerr << "[NormalEstimation Error] K > 128 is not optimized on GPU." << std::endl;
                        return;
                    }

                    scatterNormalsKernel << <gridSize, blockSize >> > (
                        sorted_indices,
                        thrust::raw_pointer_cast(d_sorted_nx.data()),
                        thrust::raw_pointer_cast(d_sorted_ny.data()),
                        thrust::raw_pointer_cast(d_sorted_nz.data()),
                        thrust::raw_pointer_cast(d_sorted_c.data()),
                        output.nx(), // 最终输出
                        output.ny(),
                        output.nz(),
                        output.curvature(),
                        size
                        );
                    CHECK_CUDA_KERNEL_ERROR();
                    CHECK_CUDA(cudaDeviceSynchronize());
                }
            }
        
    }
}