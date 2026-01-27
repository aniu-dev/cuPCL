#include "voxel_grid.cuh"
#include "internal/error.cuh"
#include "pcl_cuda/common/primitives.h"
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>

namespace pcl
{
    namespace cuda
    {
        namespace device
        {

            // 计算每个点的体素哈希值 (Hash)
            // 作用：把 float 的 (x,y,z) 映射成一个唯一的 unsigned long long 整数 ID
            __global__ void computeVoxelIndicesKernel(
                const float* __restrict__ x_in,
                const float* __restrict__ y_in,
                const float* __restrict__ z_in,
                const float3 min_val, const float3 inv_leaf_size,
                const int3 grid_dim, const int numPoints,
                int* sort_indices,                // 输出：用于初始化的索引 [0, 1, 2, ...]
                unsigned long long* voxelIndices) // 输出：每个点对应的体素 Hash 值
            {
                int tx = threadIdx.x + blockDim.x * blockIdx.x;
                if (tx >= numPoints) return;

                // 读取原始坐标
                float x = x_in[tx];
                float y = y_in[tx];
                float z = z_in[tx];

                // 计算网格坐标 (vx, vy, vz)
                int vx = (int)floorf((x - min_val.x) * inv_leaf_size.x);
                int vy = (int)floorf((y - min_val.y) * inv_leaf_size.y);
                int vz = (int)floorf((z - min_val.z) * inv_leaf_size.z);

                // 边界钳制，防止索引越界
                vx = max(0, min(grid_dim.x - 1, vx));
                vy = max(0, min(grid_dim.y - 1, vy));
                vz = max(0, min(grid_dim.z - 1, vz));

                // 计算线性 Hash 值： z * (W*H) + y * W + x
                unsigned long long hash = (unsigned long long)vx +
                    (unsigned long long)vy * grid_dim.x +
                    (unsigned long long)vz * grid_dim.x * grid_dim.y;

                voxelIndices[tx] = hash; // 记录该点属于哪个体素
                sort_indices[tx] = tx;   // 初始化排序索引，最开始是 0, 1, 2...
            }

            //标记有效体素 (Voxel Mask)
            // 作用：在排序后的 Hash 数组中，找到变化的边界。
            // 例子：Hash数组: [AAABBC] -> Mask: [100101] (表示A的开始, B的开始, C的开始)
            __global__ void gatherVaildVoxelKernel(
                const unsigned long long* __restrict__ voxel_indices, // 排序后的 Hash 数组
                int* vaild_voxel_mask,                               // 输出：Mask 数组
                const int numPoints)
            {
                int tx = blockIdx.x * blockDim.x + threadIdx.x;
                if (tx >= numPoints) return;

                // 第一个点肯定是一个新体素的开始
                if (tx == 0)
                {
                    vaild_voxel_mask[tx] = 1;
                    return;
                }

                // 比较当前点和前一个点的 Hash 值
                unsigned long long curr = voxel_indices[tx];
                unsigned long long prev = voxel_indices[tx - 1];

                // 如果不一样，说明这里是新体素的起点，标记为 1
                vaild_voxel_mask[tx] = (curr != prev) ? 1 : 0;
            }

            // Warp 级归约求和 (核心优化 Kernel)
            // 作用：计算每个体素内点的坐标之和 (Sum X, Sum Y, Sum Z)
            // 优化点：
            // 1. 间接读取：通过 sort_indices 去原始内存读数据，省去了 Gather 的显存读写。
            // 2. Warp Shuffle：在一个 Warp (32线程) 内部先互相加一下，减少对 Global Memory 的原子加法次数。
            __global__ void voxelFilterWarpReduceKernel(
                const float* __restrict__ x_in, // 原始乱序 X (只读)
                const float* __restrict__ y_in, // 原始乱序 Y (只读)
                const float* __restrict__ z_in, // 原始乱序 Z (只读)
                const int* __restrict__ sort_indices,      // 排序后的索引表：告诉我们要去哪里拿数据
                const int* __restrict__ point_to_voxel_id, // 映射表：当前线程处理的点属于第几个输出体素 (0,0,0, 1,1, 2,2...)
                float* x_out_acc,   // 输出：累加器 X
                float* y_out_acc,   // 输出：累加器 Y
                float* z_out_acc,   // 输出：累加器 Z
                int* count_out_acc, // 输出：计数器
                const int numPoints)
            {
                int tid = threadIdx.x + blockDim.x * blockIdx.x;
                int lane_id = threadIdx.x % 32; // 当前线程在 Warp 内的编号 (0-31)

                float rx = 0.0f, ry = 0.0f, rz = 0.0f;
                int my_voxel_id = -1; // 当前点属于哪个输出体素
                int contribution = 0; // 当前线程贡献的点数 (1或0)

                if (tid < numPoints) {
                    // 【关键修改】：间接寻址 (Indirect Addressing)
                    // 不按顺序读 x_in[tid]，而是读 x_in[sort_indices[tid]]
                    // 因为 tid 是按体素排好序的逻辑顺序，但数据还在老地方。
                    int original_idx = sort_indices[tid];

                    // 使用 __ldg 强制走只读缓存 (Read-Only Cache)，提高随机读取效率
                    rx = __ldg(&x_in[original_idx]);
                    ry = __ldg(&y_in[original_idx]);
                    rz = __ldg(&z_in[original_idx]);

                    // 获取当前点属于第几个体素 (由 Scan 算出来的连续 ID)
                    my_voxel_id = point_to_voxel_id[tid];
                    contribution = 1;
                }

                // ----------------------------------------------------
                // Warp 级归约逻辑 (Warp-Level Reduction)
                // 目标：如果 Warp 里相邻的线程属于同一个体素，就把它们的值加在一起，
                // 这样 32 个线程可能只需要做 1 次 atomicAdd，而不是 32 次。
                // ----------------------------------------------------

                unsigned int mask = __activemask(); // 获取当前 Warp 活跃线程掩码

                // Log2 步长归约 (offset = 1, 2, 4, 8, 16)
                for (int offset = 1; offset < 32; offset *= 2) {
                 
                    float other_x = __shfl_down_sync(mask, rx, offset);
                    float other_y = __shfl_down_sync(mask, ry, offset);
                    float other_z = __shfl_down_sync(mask, rz, offset);
                    int other_contribution = __shfl_down_sync(mask, contribution, offset);
                    int other_idx = __shfl_down_sync(mask, my_voxel_id, offset);

                    // 判断：如果邻居也是有效的，并且邻居的体素ID和我一样
                    if (my_voxel_id == other_idx && other_idx != -1) {
                        // 把邻居的值加给我
                        rx += other_x;
                        ry += other_y;
                        rz += other_z;
                        contribution += other_contribution;
                    }
                }

                // ----------------------------------------------------
                // 写入 Global Memory
                // ----------------------------------------------------
                if (tid < numPoints) {
                    // 看看“左边”一个线程 (LaneID - 1) 的体素 ID
                    int prev_idx = __shfl_up_sync(mask, my_voxel_id, 1);
                    bool is_leader = (lane_id == 0) || (my_voxel_id != prev_idx);

                    if (is_leader) {
                        atomicAdd(&x_out_acc[my_voxel_id], rx);
                        atomicAdd(&y_out_acc[my_voxel_id], ry);
                        atomicAdd(&z_out_acc[my_voxel_id], rz);
                        atomicAdd(&count_out_acc[my_voxel_id], contribution);
                    }
                }
            }
            //  归一化 (平均值计算)
            // 作用： sum / count
            __global__ void voxelNormalizationKernel(
                float* x_out, float* y_out, float* z_out,
                const int* count,
                int valid_count)
            {
                int tid = threadIdx.x + blockDim.x * blockIdx.x;
                if (tid >= valid_count) return;

                int c = count[tid];
                if (c > 0) {
                    float inv = 1.0f / (float)c;
                    x_out[tid] *= inv;
                    y_out[tid] *= inv;
                    z_out[tid] *= inv;
                }
            }
  
            void launchVoxelGridFilter(
                const GpuPointCloud* input,
                const float& lx,
                const float& ly,
                const float& lz,
                GpuPointCloud& output)       
            {
            
                int size = input->size();
                if (size == 0) {

                    std::cerr << "[Warning] VoxelGridFilter received empty cloud!" << std::endl;
                    return;
                }

                // 计算包围盒 (Min/Max)
                PointXYZ minVal, maxVal;
                reduceMinMax3D(*input, size, minVal, maxVal);
                float3 min_val = make_float3(minVal.x ,minVal.y, minVal.z);
                float3 max_val = make_float3(maxVal.x, maxVal.y, maxVal.z);
                // 计算体素尺寸倒数，避免后续做除法
                float3 inv_leaf;
                inv_leaf.x = 1.0f / lx;
                inv_leaf.y = 1.0f / ly;
                inv_leaf.z = 1.0f / lz;

                // 计算网格维度
                int3 grid_dim;
                grid_dim.x = (int)floorf((max_val.x - min_val.x) * inv_leaf.x) + 1;
                grid_dim.y = (int)floorf((max_val.y - min_val.y) * inv_leaf.y) + 1;
                grid_dim.z = (int)floorf((max_val.z - min_val.z) * inv_leaf.z) + 1;

                // 分配内存
                thrust::device_vector<unsigned long long> d_voxel_indices(size); // 存 Hash
                thrust::device_vector<int> d_sort_indices(size);                 // 存 0~N 索引

                int blockSize = 256;
                int gridSize = (size + blockSize - 1) / blockSize;

                // 计算每个点的体素索引 (Hash)
                computeVoxelIndicesKernel << <gridSize, blockSize >> > (
                    input->x(), input->y(), input->z(),
                    min_val, inv_leaf, grid_dim, size,
                    thrust::raw_pointer_cast(d_sort_indices.data()),
                    thrust::raw_pointer_cast(d_voxel_indices.data()));
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	


                //  排序 (这是目前最耗时的部分)
                // 根据 Hash 值对 d_voxel_indices 进行排序
                // 同时，d_sort_indices 会跟着一起变，这样我们知道排序后的第i个点对应原始数据的哪个位置
                // thrust::stable_sort 是基数排序(Radix Sort)，在 GPU 上非常快，但因为数据量大，依然是瓶颈
                thrust::stable_sort_by_key(d_voxel_indices.begin(),
                    d_voxel_indices.end(), d_sort_indices.begin());
                //  生成 Mask (找出体素分界线)
                thrust::device_vector<int> d_vaild_voxel_mask(size);
                gatherVaildVoxelKernel << <gridSize, blockSize >> > (
                    thrust::raw_pointer_cast(d_voxel_indices.data()),
                    thrust::raw_pointer_cast(d_vaild_voxel_mask.data()),
                    size
                    );
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	



                // 前缀和 (Scan) 生成紧凑的体素 ID
                // Mask: [1, 0, 1, 0] -> Scan: [1, 1, 2, 2] -> ID: [0, 0, 1, 1]
                // 这样每个点都知道自己应该去 output 的第几个格子累加
                thrust::device_vector<int> d_point_to_voxel_id(size);
                thrust::inclusive_scan(d_vaild_voxel_mask.begin(), d_vaild_voxel_mask.end(), d_point_to_voxel_id.begin());

                // 将 ID 减 1，使其从 0 开始
                using namespace thrust::placeholders;
                thrust::transform(d_point_to_voxel_id.begin(), d_point_to_voxel_id.end(),
                    d_point_to_voxel_id.begin(), _1 - 1);

                // 计算最终输出的体素数量
                int valid_count = d_point_to_voxel_id.back() + 1;

                // 准备输出内存 (必须清零)
                output.alloc(valid_count);
                thrust::device_vector<int> d_voxel_counts(valid_count, 0);

                CHECK_CUDA(cudaMemset(output.x(), 0, valid_count * sizeof(float)));
                CHECK_CUDA(cudaMemset(output.y(), 0, valid_count * sizeof(float)));
                CHECK_CUDA(cudaMemset(output.z(), 0, valid_count * sizeof(float)));


                //  启动核心计算 Kernel (Sum)
                // 注意：d_sort_indices 传进去了，用于间接寻址
                blockSize = 256;
                gridSize = (size + blockSize - 1) / blockSize;

                voxelFilterWarpReduceKernel << <gridSize, blockSize >> > (
                    input->x(), // 原始 X
                    input->y(), // 原始 Y
                    input->z(), // 原始 Z
                    thrust::raw_pointer_cast(d_sort_indices.data()), // 排序后的索引
                    thrust::raw_pointer_cast(d_point_to_voxel_id.data()), // 点到体素ID的映射
                    output.x(), output.y(), output.z(),
                    thrust::raw_pointer_cast(d_voxel_counts.data()), // 计数
                    size
                    );
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	

                // 归一化 (Sum / Count)
                int normGridSize = (valid_count + blockSize - 1) / blockSize;
                voxelNormalizationKernel << <normGridSize, blockSize >> > (
                    output.x(), output.y(), output.z(),
                    thrust::raw_pointer_cast(d_voxel_counts.data()),
                    valid_count
                    );
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	


            }
        }
    }
}