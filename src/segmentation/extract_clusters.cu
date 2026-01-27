#include "extract_clusters.cuh"
#include "pcl_cuda/common/common.h"
#include "lbvh/lbvh.cuh"
#include "internal/cuda_math.cuh"
#include "internal/error.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <map>
#include <chrono>
// --- Thrust 核心算法头文件 ---
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
namespace pcl
{
    namespace cuda
    {
        namespace device
        {
            __host__ __device__ float distSqPointAABB5(float3 p, const AABB& aabb) {
                float dx = ::fmaxf(0.0f, ::fmaxf(aabb.min.x - p.x, p.x - aabb.max.x));
                float dy = ::fmaxf(0.0f, ::fmaxf(aabb.min.y - p.y, p.y - aabb.max.y));
                float dz = ::fmaxf(0.0f, ::fmaxf(aabb.min.z - p.z, p.z - aabb.max.z));
                return dx * dx + dy * dy + dz * dz;
            }
            // Kernel 函数定义
            __global__ void gatherPointsKernel5(
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



            // 并查集基础操作
            __device__ int find_root(int* parent, int i) {
                while (parent[i] != i) {
                    // 路径压缩：使当前节点指向其爷爷节点，缩短树高
                    parent[i] = parent[parent[i]];
                    i = parent[i];
                }
                return i;
            }

            __device__ void union_nodes(int* parent, int i, int j) {
                int root_i = find_root(parent, i);
                int root_j = find_root(parent, j);
                if (root_i != root_j) {
                    // 使用 atomicMin 保证确定性，总是将较大的根指向较小的根
                    atomicMin(&parent[max(root_i, root_j)], min(root_i, root_j));
                }
            }
            __global__ void euclideanClusteringKernel(
                const float* __restrict__ x_in,
                const float* __restrict__ y_in,
                const float* __restrict__ z_in,
                int numPoints,
                float rSq,
                const LBVH::Node* d_nodes,
                int* d_parent  // 并查集数组
            ) {
                int tx = threadIdx.x + blockIdx.x * blockDim.x;
                if (tx >= numPoints) return;

                float3 q = make_float3(x_in[tx], y_in[tx], z_in[tx]);

                uint32_t stack[48];
                int stack_top = 0;
                stack[stack_top++] = 0;

                while (stack_top > 0) {
                    uint32_t node_idx = stack[--stack_top];
                    bool is_leaf = (node_idx & 0x80000000);
                    uint32_t real_idx = node_idx & 0x7FFFFFFF;
                    auto node = d_nodes[real_idx];
                    float dis_l = distSqPointAABB5(q, node.bounds[0]);
                    float dis_r = distSqPointAABB5(q, node.bounds[1]);
                    if (is_leaf) {
                        // 如果两个点距离小于半径，就在并查集中合并它们
                        float3 p = make_float3(x_in[real_idx], y_in[real_idx], z_in[real_idx]);
                        float3 d = q - p;
                        if (dot(d, d) <= rSq) {
                            union_nodes(d_parent, tx, real_idx);
                        }
                    }
                    else {
                        
                        if (dis_l <= rSq) stack[stack_top++] = node.leftIdx;
                        if (dis_r <= rSq) stack[stack_top++] = node.rightIdx;
                    }
                }
            }



            __global__ void flattenParentKernel(int* d_parent, int numPoints) {
                int tx = threadIdx.x + blockIdx.x * blockDim.x;
                if (tx >= numPoints) return;
                d_parent[tx] = find_root(d_parent, tx);
            }

            void launchEuclideanClusterExtraction(
                const GpuPointCloud* input,
                const float& radius,
                const int& min_cluster_size,
                const int& max_cluster_size,
                std::vector<std::vector<int>>& clusters
            )
            {

                int size = input->size();
                if (size == 0) {

                    std::cerr << "[Warning] EuclideanClusterExtraction received empty cloud!" << std::endl;
                    return;
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                pcl::cuda::LBVH bvh;
                bvh.compute(input, size);
                // 需要分别 gather x, y, z
                // 这样做虽然多了几次内存操作，但 SOA 的合并访存优势通常能弥补回来
                thrust::device_vector<float> d_sorted_x(size);
                thrust::device_vector<float> d_sorted_y(size);
                thrust::device_vector<float> d_sorted_z(size);

                //const uint32_t* sorted_indices = bvh.getSortedIndices();
                thrust::device_vector<uint32_t> d_mapped_indices(bvh.getSortedIndices(), bvh.getSortedIndices() + size);
                const int blockSize = 256;
                const int gridSize = (size + blockSize - 1) / blockSize;
                gatherPointsKernel5 << <gridSize, blockSize >> > (
                    thrust::raw_pointer_cast(d_mapped_indices.data()),
                    input->x(), input->y(), input->z(),
                    thrust::raw_pointer_cast(d_sorted_x.data()), // 传入排序后的 X
                    thrust::raw_pointer_cast(d_sorted_y.data()), // 传入排序后的 Y
                    thrust::raw_pointer_cast(d_sorted_z.data()), // 传入排序后的 Z
                    size
                    );
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	
                int* d_parent;
                cudaMalloc(&d_parent, sizeof(int) * size);
                // 初始状态：每个点的父节点是自己
                thrust::sequence(thrust::device, d_parent, d_parent + size);

                // 3. 执行聚类合并
                euclideanClusteringKernel << <gridSize, blockSize >> > (
                    thrust::raw_pointer_cast(d_sorted_x.data()), // 传入排序后的 X
                    thrust::raw_pointer_cast(d_sorted_y.data()), // 传入排序后的 Y
                    thrust::raw_pointer_cast(d_sorted_z.data()), // 传入排序后的 Z
                    size, radius * radius, bvh.getNodes(), d_parent
                    );
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());

                // 4. 路径压缩（拍扁树结构）
                flattenParentKernel << <gridSize, blockSize >> > (d_parent, size);
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());

                // A. 将 parent 指针和原始索引表包装成 Thrust 迭代器
                thrust::device_ptr<int> dev_parent_ptr(d_parent);

                // B. 排序后，相同 Parent 的点索引会排在一起
                thrust::sort_by_key(dev_parent_ptr, dev_parent_ptr + size, d_mapped_indices.begin());
                // C. 统计每个簇的大小 (Reduce by Key)
                thrust::device_vector<int> d_unique_roots(size);
                thrust::device_vector<int> d_counts(size);
                auto new_end = thrust::reduce_by_key(
                    dev_parent_ptr, dev_parent_ptr + size,
                    thrust::make_constant_iterator(1),
                    d_unique_roots.begin(),
                    d_counts.begin()
                );
                int num_unique_clusters = new_end.first - d_unique_roots.begin();

                // D. 计算每个簇在排序后数组中的起始偏移 (Exclusive Scan)
                thrust::device_vector<int> d_offsets(num_unique_clusters);
                thrust::exclusive_scan(d_counts.begin(), d_counts.begin() + num_unique_clusters, d_offsets.begin());

                // E. 将元数据拷回 CPU 进行过滤（这部分数据量极小，千万级点云通常也只有几万个簇）
                std::vector<int> h_unique_counts(num_unique_clusters);
                std::vector<int> h_unique_offsets(num_unique_clusters);
                thrust::copy(d_counts.begin(), d_counts.begin() + num_unique_clusters, h_unique_counts.begin());
                thrust::copy(d_offsets.begin(), d_offsets.begin() + num_unique_clusters, h_unique_offsets.begin());

                // F. 准备最终的点索引大数组拷回 CPU (一次性拷贝，避开碎片化)
                std::vector<uint32_t> h_all_sorted_indices(size);
                thrust::copy(d_mapped_indices.begin(), d_mapped_indices.end(), h_all_sorted_indices.begin());

                // G. 根据大小过滤并构建输出 vector
                clusters.clear();
                for (int i = 0; i < num_unique_clusters; ++i) {
                    int c_size = h_unique_counts[i];
                    if (c_size >= min_cluster_size && c_size <= max_cluster_size) {
                        int start_idx = h_unique_offsets[i];

                        // 直接从大数组中截取这一段，非常快
                        std::vector<int> cluster;
                        cluster.reserve(c_size);
                        for (int j = 0; j < c_size; ++j) {
                            cluster.push_back(static_cast<int>(h_all_sorted_indices[start_idx + j]));
                        }
                        clusters.push_back(std::move(cluster));
                    }
                }

                cudaFree(d_parent);

            };
        }
    }
}