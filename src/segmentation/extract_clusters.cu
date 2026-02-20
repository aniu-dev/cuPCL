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
namespace pcl {
    namespace cuda {
        namespace device {


            
            __host__ __device__ float distSqPointAABB5(float3 p, const AABB& aabb) {
                float dx = ::fmaxf(0.0f, ::fmaxf(aabb.min.x - p.x, p.x - aabb.max.x));
                float dy = ::fmaxf(0.0f, ::fmaxf(aabb.min.y - p.y, p.y - aabb.max.y));
                float dz = ::fmaxf(0.0f, ::fmaxf(aabb.min.z - p.z, p.z - aabb.max.z));
                return dx * dx + dy * dy + dz * dz;
            }
            
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
            __device__ int find_root(int* parent, int i) {
                while (parent[i] != i) {
                    parent[i] = parent[parent[i]];
                    i = parent[i];
                }
                return i;
            }

            __device__ void union_nodes(int* parent, int i, int j) {
                int root_i = find_root(parent, i);
                int root_j = find_root(parent, j);
                if (root_i != root_j) {
                    if (root_i < root_j) atomicMin(&parent[root_j], root_i);
                    else atomicMin(&parent[root_i], root_j);
                }
            }
            __global__ void warpMergeKernel(const float * __restrict__ d_in_x,
                const float* __restrict__ d_in_y, const float* __restrict__ d_in_z,
                int numPoints,float rSq, int* d_parent)
            {
                const int tid = threadIdx.x + blockDim.x * blockIdx.x;
                const int tx = threadIdx.x;
                if (tid >= numPoints) return;

                __shared__ float s_x[256];
                __shared__ float s_y[256];
                __shared__ float s_z[256];
                __shared__ int s_root[256];
                float x_in = d_in_x[tid];
                float y_in = d_in_y[tid];
                float z_in = d_in_z[tid];

                int my_root = tid;


                for (int mask = 1; mask <= 16; mask++)
                {
                    float ox = __shfl_up_sync(0xffffffff, x_in, mask);
                    float oy = __shfl_up_sync(0xffffffff, y_in, mask);
                    float oz = __shfl_up_sync(0xffffffff, z_in, mask);
                    int oroot = __shfl_up_sync(0xffffffff, my_root, mask);

                    float dis = (ox - x_in) * (ox - x_in) + (oy - y_in) * (oy - y_in) + (oz - z_in) * (oz - z_in);
                    if (dis < rSq)
                    {
                        if (my_root > oroot)
                        {
                            my_root = oroot;
                        }
                    }
                }

                s_x[tx] = x_in;
                s_y[tx] = y_in;
                s_z[tx] = z_in;
                s_root[tx] = my_root - blockIdx.x * blockDim.x;
                __syncthreads();


                const int laneId = tx % 32;
                const int warpId = tx / 32;


               
                if (warpId > 0 && laneId < 4) {
                   
                    int prev_warp_end = (warpId << 5) - 1;
                    for (int i = 0; i < 4; i++) {
                        int target_tx = prev_warp_end - i;
                        float dx = x_in - s_x[target_tx];
                        float dy = y_in - s_y[target_tx];
                        float dz = z_in - s_z[target_tx];

                        if ((dx * dx + dy * dy + dz * dz) < rSq) {
                           
                            int root_current = tx;
                            int root_target = target_tx;

                            
                            while (s_root[root_current] != root_current) root_current = s_root[root_current];
                            while (s_root[root_target] != root_target) root_target = s_root[root_target];

                            if (root_current != root_target) {
                                if (root_current < root_target) atomicMin(&s_root[root_target], root_current);
                                else atomicMin(&s_root[root_current], root_target);
                            }
                        }
                    }
                }
                __syncthreads();

                
#pragma unroll
                for (int i = 0; i < 8; i++) { 
                    s_root[tx] = s_root[s_root[tx]];
                    __syncthreads();
                }

               
                d_parent[tid] = s_root[tx] + blockIdx.x * blockDim.x;
            }
            __global__ void blockBridgeKernel(
                const float* sx, const float* sy, const float* sz,
                int numPoints, float rSq, int* d_parent, int blockSize)
            {
                
                int bIdx = blockIdx.x * blockDim.x + threadIdx.x;
                int start_of_next_block = (bIdx + 1) * blockSize;

                if (start_of_next_block >= numPoints) return;

                for (int i = 1; i <= 4; ++i) {
                    int idxA = start_of_next_block - i;
                    if (idxA < 0) break;

                    float ax = sx[idxA]; float ay = sy[idxA]; float az = sz[idxA];

                    for (int j = 0; j < 4; ++j) {
                        int idxB = start_of_next_block + j;
                        if (idxB >= numPoints) break;

                        float dx = ax - sx[idxB];
                        float dy = ay - sy[idxB];
                        float dz = az - sz[idxB];

                        if ((dx * dx + dy * dy + dz * dz) < rSq) {
                            
                            union_nodes(d_parent, idxA, idxB);
                        }
                    }
                }
            }

            __global__ void clusteringKernel(
                const float* sx, const float* sy, const float* sz,
                int numPoints, float rSq, const LBVH::Node* d_nodes, int* d_parent)
            {
                int tx = blockIdx.x * blockDim.x + threadIdx.x;
                if (tx >= numPoints) return;

     
                int root_tx_cached = d_parent[tx];
                float3 q = make_float3(sx[tx], sy[tx], sz[tx]);
                const float eps_rSq = rSq + 1e-6f;

                uint32_t stack[30];
                int stack_top = 0;
                stack[stack_top++] = 0;

                while (stack_top > 0) {
                    uint32_t node_idx = stack[--stack_top];
                    bool is_leaf = (node_idx & 0x80000000);
                    uint32_t real_idx = node_idx & 0x7FFFFFFF;

                    if (is_leaf) {
                       
                        int neighbor_idx = (int)real_idx;
                        int p_neighbor = d_parent[neighbor_idx];
                                   
                           
                            if (p_neighbor != root_tx_cached) {
                                // 只有父节点不等，才进行find_root 和 距离计算
                                int root_neighbor = find_root(d_parent, neighbor_idx);

                                if (root_tx_cached != root_neighbor) {
                                    float3 p = make_float3(sx[neighbor_idx], sy[neighbor_idx], sz[neighbor_idx]);
                                    float3 d = q - p;
                                    if (dot(d, d) <= eps_rSq) {
                                        // 执行原子合并
                                        union_nodes(d_parent, tx, neighbor_idx);
                                        // 合并后，我的 root 可能会变小，更新缓存
                                        root_tx_cached = find_root(d_parent, tx);
                                    }
                                }
                            }
                                    
                    }
                    else {
                        auto node = d_nodes[real_idx];
                        if (distSqPointAABB5(q, node.bounds[0]) <= eps_rSq) stack[stack_top++] = node.leftIdx;
                        if (distSqPointAABB5(q, node.bounds[1]) <= eps_rSq) stack[stack_top++] = node.rightIdx;
                    }
                }
            }
            __global__ void flattenKernel(int* d_parent, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) d_parent[i] = find_root(d_parent, i);
            }
            void launchEuclideanClusterExtraction(
                const GpuPointCloud* input,
                const float& radius,
                const int& min_cluster_size,
                const int& max_cluster_size,
                std::vector<std::vector<int>>& clusters)
            {
                int size = input->size();
                if (size == 0) {

                    std::cerr << "[Warning] EuclideanClusterExtraction received empty cloud!" << std::endl;
                    return;
                }

                // 1. LBVH 构建
                pcl::cuda::LBVH bvh;
                bvh.compute(input, size);
               
                const uint32_t* d_sorted_indices = bvh.getSortedIndices();

                // 2. 准备 LBVH 排序后的坐标
                thrust::device_vector<float> sx(size), sy(size), sz(size);
                const int blockSize = 256;
                const int gridSize = (size + blockSize - 1) / blockSize;
                gatherPointsKernel5 << <gridSize, blockSize >> > (
                    d_sorted_indices,
                    input->x(), input->y(), input->z(),
                    thrust::raw_pointer_cast(sx.data()), // 传入排序后的 X
                    thrust::raw_pointer_cast(sy.data()), // 传入排序后的 Y
                    thrust::raw_pointer_cast(sz.data()), // 传入排序后的 Z
                    size
                    );
                
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());
                // 3. DSU 初始化与预合并
                int* d_parent;
                CHECK_CUDA(cudaMalloc(&d_parent, sizeof(int) * size));
                warpMergeKernel << <gridSize, blockSize >> > (
                    thrust::raw_pointer_cast(sx.data()), thrust::raw_pointer_cast(sy.data()), thrust::raw_pointer_cast(sz.data()),
                    size, radius * radius, d_parent);
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());

                // 4. 核心聚类搜索
                clusteringKernel << <gridSize, blockSize >> > (
                    thrust::raw_pointer_cast(sx.data()), thrust::raw_pointer_cast(sy.data()), thrust::raw_pointer_cast(sz.data()),
                    size, radius * radius, bvh.getNodes(), d_parent);
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());
                flattenKernel << <gridSize, blockSize >> > (d_parent, size);
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());
                // 5.映射回原始索引并分组

                thrust::device_vector<int> d_mapped_indices(size);
                thrust::copy(thrust::device, d_sorted_indices, d_sorted_indices + size, d_mapped_indices.begin());

                thrust::device_ptr<int> dev_parent_ptr(d_parent);
              
                thrust::sort_by_key(dev_parent_ptr, dev_parent_ptr + size, d_mapped_indices.begin());

                // 6. 规约统计
                thrust::device_vector<int> d_unique_roots(size);
                thrust::device_vector<int> d_counts(size);
                auto res = thrust::reduce_by_key(dev_parent_ptr, dev_parent_ptr + size, thrust::make_constant_iterator(1), d_unique_roots.begin(), d_counts.begin());
                int num_clusters = res.first - d_unique_roots.begin();

                std::vector<int> h_counts(num_clusters);
                thrust::copy(d_counts.begin(), d_counts.begin() + num_clusters, h_counts.begin());
                thrust::device_vector<int> d_offsets(num_clusters);
                thrust::exclusive_scan(d_counts.begin(), d_counts.begin() + num_clusters, d_offsets.begin());
                std::vector<int> h_offsets(num_clusters);
                thrust::copy(d_offsets.begin(), d_offsets.begin() + num_clusters, h_offsets.begin());
                std::vector<int> h_all_ids(size);
                thrust::copy(d_mapped_indices.begin(), d_mapped_indices.end(), h_all_ids.begin());

                clusters.clear();
                for (int i = 0; i < num_clusters; ++i) {
                    if (h_counts[i] >= min_cluster_size && h_counts[i] <= max_cluster_size) {
                        std::vector<int> cluster(h_all_ids.begin() + h_offsets[i], h_all_ids.begin() + h_offsets[i] + h_counts[i]);
                        clusters.push_back(std::move(cluster));
                    }
                }

                CHECK_CUDA(cudaFree(d_parent));
            }

        } // namespace device
    } // namespace cuda
} // namespace pcl


