#include "radius_outlier_removal.cuh"
#include "pcl_cuda/common/common.h"
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

            __host__ __device__ float distSqPointAABB(float3 p, const AABB& aabb) {
                float dx = ::fmaxf(0.0f, ::fmaxf(aabb.min.x - p.x, p.x - aabb.max.x));
                float dy = ::fmaxf(0.0f, ::fmaxf(aabb.min.y - p.y, p.y - aabb.max.y));
                float dz = ::fmaxf(0.0f, ::fmaxf(aabb.min.z - p.z, p.z - aabb.max.z));
                return dx * dx + dy * dy + dz * dz;
            }
            // Kernel 函数定义
            __global__ void gatherPointsKernel(
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
            __global__ void countNeighborsSortedSOAKernel(
                const float* __restrict__ x_in, // 排序后的 X
                const float* __restrict__ y_in, // 排序后的 Y
                const float* __restrict__ z_in, // 排序后的 Z
                int numPoints,
                float rSq,
                const LBVH::Node* d_nodes,
                int* d_mask,
                int min_pts
            ) {
                int tx = threadIdx.x + blockIdx.x * blockDim.x;
                if (tx >= numPoints) return;
                float3 q = make_float3(x_in[tx], y_in[tx], z_in[tx]);
                int count = -1;

                uint32_t stack[48];
                int stack_top = 0;
                stack[stack_top++] = 0;
                bool is_iniler = false;
                while (stack_top > 0) {
                    uint32_t node_idx = stack[--stack_top];
                    bool is_leaf = (node_idx & 0x80000000);
                    uint32_t real_idx = node_idx & 0x7FFFFFFF;

                    if (is_leaf) {
                        float3 p = make_float3(x_in[real_idx], y_in[real_idx], z_in[real_idx]);

                        float3 d = q - p;
                        if (dot(d, d) <= rSq) {
                            count++;
                            if (count >= min_pts) {
                                is_iniler = true;
                                stack_top = 0;
                                break;
                            }
                        }
                        //}
                    }
                    else {
                        auto node = d_nodes[real_idx];
                        bool l = distSqPointAABB(q, node.bounds[0]) <= rSq;
                        bool r = distSqPointAABB(q, node.bounds[1]) <= rSq;
                        if (l) stack[stack_top++] = node.leftIdx;
                        if (r) stack[stack_top++] = node.rightIdx;
                    }
                }

                d_mask[tx] = is_iniler ? 1 : 0;
            }


            void launchRadiusOutlierRemovalFilter(
                const GpuPointCloud* input,
                const float& radius,
                const int& min_pts,
                GpuPointCloud& output
            )
            {

                int size = input->size();
                if (size == 0) {

                    std::cerr << "[Warning] RadiusOutlierRemovalFilter received empty cloud!" << std::endl;
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
                gatherPointsKernel << <gridSize, blockSize >> > (
                    sorted_indices,
                    input->x(), input->y(), input->z(),
                    thrust::raw_pointer_cast(d_sorted_x.data()), // 传入排序后的 X
                    thrust::raw_pointer_cast(d_sorted_y.data()), // 传入排序后的 Y
                    thrust::raw_pointer_cast(d_sorted_z.data()), // 传入排序后的 Z
                    size
                    );
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	

                // thrust::device_vector<int> d_mask(size);
                int* d_mask;
                CHECK_CUDA(cudaMalloc(&d_mask, sizeof(int) * size));
                CHECK_CUDA(cudaMemset(d_mask, 0, sizeof(int) * size));
                countNeighborsSortedSOAKernel << <gridSize, blockSize >> > (
                    thrust::raw_pointer_cast(d_sorted_x.data()), // 传入排序后的 X
                    thrust::raw_pointer_cast(d_sorted_y.data()), // 传入排序后的 Y
                    thrust::raw_pointer_cast(d_sorted_z.data()), // 传入排序后的 Z
                    size,
                    radius * radius,
                    bvh.getNodes(),
                    d_mask,
                    min_pts
                    );
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	
                
                copyPointCloud(thrust::raw_pointer_cast(d_sorted_x.data()),
                    thrust::raw_pointer_cast(d_sorted_y.data()),
                    thrust::raw_pointer_cast(d_sorted_z.data()),
                    size, output, d_mask);
                
                                          
                CHECK_CUDA(cudaFree(d_mask));
            
            };
        }
    }
}