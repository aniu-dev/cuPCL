#include "icp_point2point.cuh"
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
            __host__ __device__ float distSqPointAABB3(float3 p, const AABB& aabb) {
                float dx = ::fmaxf(0.0f, ::fmaxf(aabb.min.x - p.x, p.x - aabb.max.x));
                float dy = ::fmaxf(0.0f, ::fmaxf(aabb.min.y - p.y, p.y - aabb.max.y));
                float dz = ::fmaxf(0.0f, ::fmaxf(aabb.min.z - p.z, p.z - aabb.max.z));
                return dx * dx + dy * dy + dz * dz;
            }    

            __global__ void subtractCentroid(const float* __restrict__ input_x,
                const float* __restrict__ input_y,
                const float* __restrict__ input_z,
                float* __restrict__ output_x,
                float* __restrict__ output_y,
                float* __restrict__ output_z,
                const int size,
                const PointXYZ centroid)
            {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid >= size) return;
                float x = input_x[tid] - centroid.x;
                float y = input_y[tid] - centroid.y;
                float z = input_z[tid] - centroid.z;

                output_x[tid] = x;
                output_y[tid] = y;
                output_z[tid] = z;
            }


           __global__ void computeMeanDistancesKernel(
                const LBVH::Node* __restrict__ nodes,
                const float* __restrict__ source_x,
                const float* __restrict__ source_y,
                const float* __restrict__ source_z,
                const int num_points,
                const float max_dist_sq,
                double* d_out              

            ) {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                int tx = threadIdx.x;

                // 1. 定义累加器 (初始化为 0)
                double local_vals[17] = { 0.0 };

                // 2. 只有在此范围内的线程才进行搜索计算 (Masking)
                // 【关键修改】不要 return，而是用 bool 控制
                bool is_active = (tid < num_points);

                if (is_active) {
                    float3 q = make_float3(source_x[tid], source_y[tid], source_z[tid]);
                    float3 searchPoint = make_float3(0.0f, 0.0f, 0.0f);
                    float best_dist_sq = max_dist_sq;

                    // 【关键修改】增加栈大小，防止溢出
                    int stack[48];
                    int stack_top = 0;
                    stack[stack_top++] = 0;
                    bool found = false;

                    while (stack_top > 0) {
                        int node_idx = stack[--stack_top];

                        // 边界检查，防止栈溢出后的非法读取
                        // (通常 LBVH 内部节点都在范围内，但这有助于调试)
                        // const LBVH::Node& node = nodes[node_idx]; 

                        // 使用 __ldg 优化只读缓存读取
                        const LBVH::Node node = nodes[node_idx];

                        int left_idx = node.leftIdx;
                        int right_idx = node.rightIdx;
                        bool is_leaf_l = (left_idx & 0x80000000);
                        bool is_leaf_r = (right_idx & 0x80000000);
                        int idx_l = left_idx & 0x7FFFFFFF;
                        int idx_r = right_idx & 0x7FFFFFFF;

                        float dist_l = distSqPointAABB3(q, node.bounds[0]);
                        float dist_r = distSqPointAABB3(q, node.bounds[1]);

                        // --- 左子树 ---
                        if (dist_l < best_dist_sq) {
                            if (is_leaf_l) {
                                best_dist_sq = dist_l;
                                searchPoint = node.bounds[0].min; // 对于 Leaf，bounds.min 就是点坐标
                                found = true;
                            }
                        }
                        else { dist_l = 1e38f; }

                        // --- 右子树 ---
                        if (dist_r < best_dist_sq) {
                            if (is_leaf_r) {
                                best_dist_sq = dist_r;
                                searchPoint = node.bounds[1].min;
                                found = true;
                            }
                        }
                        else { dist_r = 1e38f; }

                        // --- 入栈逻辑 ---
                        bool tr_l = (dist_l < 1e38f) && !is_leaf_l;
                        bool tr_r = (dist_r < 1e38f) && !is_leaf_r;

                        // 检查栈是否溢出 (非常重要！)
                        if (stack_top + 2 >= 64) break;

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

                    // 如果找到了对应点，填充统计数据
                    if (found) {
                        local_vals[0] = q.x ;
                        local_vals[1] = q.y ;
                        local_vals[2] = q.z ;
                        local_vals[3] = searchPoint.x;
                        local_vals[4] = searchPoint.y;
                        local_vals[5] = searchPoint.z;

                        local_vals[6] = (double)q.x * searchPoint.x;
                        local_vals[7] = (double)q.x * searchPoint.y;
                        local_vals[8] = (double)q.x * searchPoint.z;
                        local_vals[9] = (double)q.y * searchPoint.x;
                        local_vals[10] = (double)q.y * searchPoint.y;
                        local_vals[11] = (double)q.y * searchPoint.z;
                        local_vals[12] = (double)q.z * searchPoint.x;
                        local_vals[13] = (double)q.z * searchPoint.y;
                        local_vals[14] = (double)q.z * searchPoint.z;
                        local_vals[15] = 1.0; // count = 1
                        local_vals[16] = (double)best_dist_sq;
                    }
                }
                // is_active 为 false 的线程，local_vals 保持全 0，安全参与后续归约

                // ================================================
                //  并行归约 (即使是空闲线程也要执行，提供 0 值)
                // ================================================

                // 1. Warp 归约
                for (int k = 0; k < 17; k++) {
                    local_vals[k] = warpSum(local_vals[k]);
                }

                // 2. 写入 Shared Memory (每个 Warp 的 Lane 0 写入)
                const int warpId = tx / 32;
                const int laneId = tx % 32;
                // 假设 BlockSize = 256，则有 8 个 Warp
                __shared__ double sm_sum[32][17];

                if (laneId == 0) {
                    for (int k = 0; k < 17; k++) {
                        sm_sum[warpId][k] = local_vals[k];
                    }
                }

                // 【关键修改】现在所有线程都活着，这里不会死锁
                __syncthreads();

                // 3. Block 归约 (由第一个 Warp 完成)
                if (warpId == 0) {
                    for (int k = 0; k < 17; k++) {
                        // 从共享内存读取各个 Warp 的总和
                        // 注意：BlockSize 256 -> blockDim.x/32 = 8 个 warp
                        // 只有前 8 个 lane 需要读取 sm_sum，其他的补 0
                        double val = (laneId < (blockDim.x / 32)) ? sm_sum[laneId][k] : 0.0;
                        local_vals[k] = warpSum(val);
                    }
                }

                // 4. 原子加到全局内存 (Block 的第一个线程执行)
                if (tx == 0) {
                    for (int k = 0; k < 17; k++) {
                        atomicAdd(&d_out[k], local_vals[k]);
                    }
                }
            }
            __global__ void transformPointsKernel(float* x, float* y, float* z, int n, Eigen::Matrix4f mat) {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid >= n) return;

                float px = x[tid], py = y[tid], pz = z[tid];
                // 应用 4x4 矩阵变换
                x[tid] = mat(0, 0) * px + mat(0, 1) * py + mat(0, 2) * pz + mat(0, 3);
                y[tid] = mat(1, 0) * px + mat(1, 1) * py + mat(1, 2) * pz + mat(1, 3);
                z[tid] = mat(2, 0) * px + mat(2, 1) * py + mat(2, 2) * pz + mat(2, 3);
            }

            void launchIterativeClosestPoint(
                const GpuPointCloud* source_input,
                const GpuPointCloud* target_input,
                const int& max_iterations,
                const float& transformation_epsilon,
                const float& euclidean_fitness_epsilon,
                const float& max_correspondence_distance,
                Eigen::Matrix4f& matrix,
                GpuPointCloud& output
            )
            {

                int source_size = source_input->size();
                if (source_size == 0) {

                    std::cerr << "[Warning] IterativeClosestPoint received empty source cloud!" << std::endl;
                    return;
                }
                int target_size = target_input->size();
                if (target_size == 0) {

                    std::cerr << "[Warning] IterativeClosestPoint received empty target cloud!" << std::endl;
                    return;
                }
                PointXYZ centroid;
                computeCentroid3D(*target_input, centroid);

                GpuPointCloud targetPointCloud;
                targetPointCloud.alloc(target_size);
                output.alloc(source_size);

                int blockSize = 256;
                int gridSize = (source_size + blockSize - 1) / blockSize;

                //将源点云和目标点云减去质心
                subtractCentroid << <gridSize,blockSize>> > (source_input->x(),
                    source_input->y(), 
                    source_input->z(),
                    output.x(),
                    output.y(),
                    output.z(), source_size, centroid);
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());
                gridSize = (target_size + blockSize - 1) / blockSize;
                subtractCentroid << < gridSize,blockSize>> > (target_input->x(),
                    target_input->y(),
                    target_input->z(),
                    targetPointCloud.x(),
                    targetPointCloud.y(),
                    targetPointCloud.z(), target_size, centroid);
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());
                pcl::cuda::LBVH bvh;
                bvh.compute(&targetPointCloud, target_size);

                const LBVH::Node* d_nodes = bvh.getNodes();
                // 2. 准备统计缓冲区 (16个double: 3个sum_s, 3个sum_t, 9个sum_st, 1个count)
                double* d_out;
                CHECK_CUDA(cudaMalloc(&d_out, 17 * sizeof(double)));
                double h_out[17];

                // 初始总变换矩阵
                float prev_fitness = 1e38f;
                float dist_sq_threshold = max_correspondence_distance * max_correspondence_distance;

                matrix = Eigen::Matrix4f::Identity();
                gridSize = (source_size + blockSize - 1) / blockSize;

                for (int i = 0; i < max_iterations; i++)
                {

                    CHECK_CUDA(cudaMemset(d_out, 0, 17 * sizeof(double)));

                    computeMeanDistancesKernel << <gridSize, blockSize >> > (
                        d_nodes,
                        output.x(), output.y(), output.z(),
                        source_size,
                        dist_sq_threshold,
                        d_out
                        );
                    CHECK_CUDA_KERNEL_ERROR();
                    CHECK_CUDA(cudaDeviceSynchronize());
  
                    CHECK_CUDA(cudaMemcpy(h_out, d_out, 17 * sizeof(double), cudaMemcpyDeviceToHost));

                    double N = h_out[15];
                    if (N < 3) {
                        std::cerr << "[ICP] Too few correspondences found!" << std::endl;
                        break;
                    }

             
                    Eigen::Vector3d mu_s(h_out[0] / N, h_out[1] / N, h_out[2] / N);
                    Eigen::Vector3d mu_t(h_out[3] / N, h_out[4] / N, h_out[5] / N);

                    // 构造去质心后的协方差矩阵 H (3x3)
                    // H = sum(s_i * t_i^T) - N * mu_s * mu_t^T
                    Eigen::Matrix3d H;
                    H(0, 0) = h_out[6] - N * mu_s.x() * mu_t.x();
                    H(0, 1) = h_out[7] - N * mu_s.x() * mu_t.y();
                    H(0, 2) = h_out[8] - N * mu_s.x() * mu_t.z();
                    H(1, 0) = h_out[9] - N * mu_s.y() * mu_t.x();
                    H(1, 1) = h_out[10] - N * mu_s.y() * mu_t.y();
                    H(1, 2) = h_out[11] - N * mu_s.y() * mu_t.z();
                    H(2, 0) = h_out[12] - N * mu_s.z() * mu_t.x();
                    H(2, 1) = h_out[13] - N * mu_s.z() * mu_t.y();
                    H(2, 2) = h_out[14] - N * mu_s.z() * mu_t.z();

                    // SVD 分解求解旋转 R
                    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    Eigen::Matrix3d U = svd.matrixU();
                    Eigen::Matrix3d V = svd.matrixV();
                    Eigen::Matrix3d R = V * U.transpose();

                    // 处理镜像问题
                    if (R.determinant() < 0) {
                        V.col(2) *= -1;
                        R = V * U.transpose();
                    }

                    // 计算平移 t
                    Eigen::Vector3d t = mu_t - R * mu_s;

                    // 构造本轮迭代的变换矩阵
                    Eigen::Matrix4f T_current = Eigen::Matrix4f::Identity();
                    T_current.block<3, 3>(0, 0) = R.cast<float>();
                    T_current.block<3, 1>(0, 3) = t.cast<float>();



                    // 更新总变换矩阵和 Source 点云位置
                    matrix = T_current * matrix;
                    transformPointsKernel << <gridSize, blockSize >> > (output.x(), output.y(), output.z(), source_size, T_current);
                    CHECK_CUDA_KERNEL_ERROR();
                    CHECK_CUDA(cudaDeviceSynchronize());
                    // 收敛判定 (Fitness & Epsilon)

                    double total_dist_sq_sum = h_out[16];
                    float current_fitness = (float)(total_dist_sq_sum / N);
                    if (std::abs(prev_fitness - current_fitness) < euclidean_fitness_epsilon) {
                        std::cout << "[ICP] Converged: MSE change < epsilon." << std::endl;
                        break;
                    }
                    // 判定 变换矩阵的变化量 (Epsilon)
                    float delta = (T_current - Eigen::Matrix4f::Identity()).norm();
                    if (delta < transformation_epsilon) {
                         std::cout << "[ICP] Converged at iteration " << i << " due to epsilon." << std::endl;
                        break;
                    }


                    prev_fitness = current_fitness;
                }
                Eigen::Matrix3f R_final = matrix.block<3, 3>(0, 0);
                Eigen::Vector3f t_local = matrix.block<3, 1>(0, 3);
                Eigen::Vector3f C(centroid.x, centroid.y, centroid.z);

                //  计算全局平移
                Eigen::Vector3f t_global = t_local + C - (R_final * C);

                //  将修正后的平移向量写回矩阵
                matrix.block<3, 1>(0, 3) = t_global;
                CHECK_CUDA(cudaFree(d_out));
               

            };
        }
    }
}