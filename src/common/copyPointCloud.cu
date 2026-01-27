#include "pcl_cuda/common/common.h"
#include "pcl_cuda/common/primitives.h"
#include <device_launch_parameters.h>
#include "internal/error.cuh"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
namespace pcl {
    namespace cuda {
        __global__ void copyPointCloudMaskToIndicesKernel(int* d_outIndices, int* d_mask, int* d_indices, int numPoints)
        {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid >= numPoints) return;
            int mask = d_mask[tid];
            if (mask == 1)
            {
                int index = d_indices[tid];
                d_outIndices[index] = tid;
            }
        }
        __global__ void copyPointCloudKernel(const float* d_in_x, const float* d_in_y, const float* d_in_z,
            float* d_out_x, float* d_out_y, float* d_out_z, int* d_indices, int numPoints)
        {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid >= numPoints) return;
            int index = d_indices[tid];
            d_out_x[tid] = d_in_x[index];
            d_out_y[tid] = d_in_y[index];
            d_out_z[tid] = d_in_z[index];
        }
        void copyPointCloud(const GpuPointCloud& cloud_in, GpuPointCloud& cloud_out, int* d_mask)
        {
            int size = cloud_in.size();
            if (size == 0) {

                std::cerr << "[Warning] copyPointCloud received empty cloud!" << std::endl;
                return;
            }
            int blockSize = 256;
            int gridSize = (size + blockSize - 1) / blockSize;
            int valid_count = 0;
            int* d_outIndices;
            int* d_indices;
            CHECK_CUDA(cudaMalloc(&d_indices, sizeof(int) * size));
            scanExclusive(d_mask, d_indices, size);
            int last_mask, last_index;
            CHECK_CUDA(cudaMemcpy(&last_mask, d_mask + size - 1, sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(&last_index, d_indices + size - 1, sizeof(int), cudaMemcpyDeviceToHost));
            valid_count = last_index + last_mask;

            CHECK_CUDA(cudaMalloc(&d_outIndices, sizeof(int) * valid_count));
            copyPointCloudMaskToIndicesKernel << <gridSize, blockSize >> > (d_outIndices, d_mask, d_indices, size);
            CHECK_CUDA_KERNEL_ERROR();
            CHECK_CUDA(cudaDeviceSynchronize());
            cloud_out.alloc(valid_count);
            copyPointCloudKernel << <gridSize, blockSize >> > (cloud_in.x(), cloud_in.y(), cloud_in.z(),
                cloud_out.x(), cloud_out.y(), cloud_out.z(), d_outIndices, valid_count);
            CHECK_CUDA_KERNEL_ERROR();
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaFree(d_indices));
            CHECK_CUDA(cudaFree(d_outIndices));
        }
        __global__ void copyPointCloudIndicesToMaskKernel(int* d_mask, int* d_indices, int numPoints)
        {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid >= numPoints) return;
            int index = d_indices[tid];
            d_mask[index] = 0;
        }
        void copyPointCloud(const GpuPointCloud& cloud_in, GpuPointCloud& cloud_out, int* d_indices, int numIndices ,bool negative)
        {
            int size = cloud_in.size();
            if (size == 0) {

                std::cerr << "[Warning] copyPointCloud received empty cloud!" << std::endl;
                return;
            }
            int valid_count = 0;
            int* d_outIndices;
            int blockSize = 256;
            int gridSize;
            if (negative == true)
            {
                int* d_mask;
                CHECK_CUDA(cudaMalloc(&d_mask, sizeof(int) * size));
                CHECK_CUDA(cudaMemset(d_mask, 1, sizeof(int) * size));
                valid_count = numIndices;

                gridSize = (valid_count + blockSize - 1) / blockSize;
                copyPointCloudIndicesToMaskKernel << <gridSize, blockSize >> > (d_mask, d_indices, valid_count);
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());
                int* d_indicesTemp;
                CHECK_CUDA(cudaMalloc(&d_indicesTemp, sizeof(int) * size));
                scanExclusive(d_mask, d_indicesTemp, size);
                //cudaMemcpy(&valid_count, d_indicesTemp + (size - 1), sizeof(int), cudaMemcpyDeviceToHost);
                int last_mask, last_index;
                CHECK_CUDA(cudaMemcpy(&last_mask, d_mask + size - 1, sizeof(int), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(&last_index, d_indicesTemp + size - 1, sizeof(int), cudaMemcpyDeviceToHost));
                valid_count = last_index + last_mask;


                CHECK_CUDA(cudaMalloc(&d_outIndices, sizeof(int) * valid_count));
                gridSize = (size + blockSize - 1) / blockSize;
                copyPointCloudMaskToIndicesKernel << <gridSize, blockSize >> > (d_outIndices, d_mask, d_indices, size);
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaFree(d_mask));
                CHECK_CUDA(cudaFree(d_indicesTemp));

            }
            else
            {
                valid_count = numIndices;
                gridSize = (valid_count + blockSize - 1) / blockSize;
                CHECK_CUDA(cudaMalloc(&d_outIndices, sizeof(int) * valid_count));
                CHECK_CUDA(cudaMemcpy(&d_outIndices, d_indices, sizeof(int) * valid_count, cudaMemcpyDeviceToDevice));
            }
            cloud_out.alloc(valid_count);
            copyPointCloudKernel << <gridSize, blockSize >> > (cloud_in.x(), cloud_in.y(), cloud_in.z(),
                cloud_out.x(), cloud_out.y(), cloud_out.z(), d_outIndices, valid_count);
            CHECK_CUDA_KERNEL_ERROR();
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaFree(d_outIndices));
        }

        void copyPointCloud(const GpuPointCloud& cloud_in, GpuPointCloud& cloud_out, std::vector<int>h_indices, bool negative)
        {
            int size = cloud_in.size();
            if (size == 0) {

                std::cerr << "[Warning] copyPointCloud received empty cloud!" << std::endl;
                return;
            }
            int valid_count = 0;

            int blockSize = 256;
            int gridSize;

            thrust::device_vector<int>d_indices(h_indices.begin(), h_indices.end());
          
            if (negative == true)
            {
                int* d_outIndices;
                int* d_mask;
                CHECK_CUDA(cudaMalloc(&d_mask, sizeof(int) * size));
                CHECK_CUDA(cudaMemset(d_mask, 1, sizeof(int) * size));
                valid_count = d_indices.size();

                gridSize = (valid_count + blockSize - 1) / blockSize;
                copyPointCloudIndicesToMaskKernel << <gridSize, blockSize >> > (d_mask, thrust::raw_pointer_cast(d_indices.data()), valid_count);
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());
                int* d_indicesTemp;
                CHECK_CUDA(cudaMalloc(&d_indicesTemp, sizeof(int) * size));
                scanExclusive(d_mask, d_indicesTemp, size);
                int last_mask, last_index;
                CHECK_CUDA(cudaMemcpy(&last_mask, d_mask + size - 1, sizeof(int), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(&last_index, d_indicesTemp + size - 1, sizeof(int), cudaMemcpyDeviceToHost));
                valid_count = last_index + last_mask;

                //cudaMemcpy(&valid_count, d_indicesTemp + (size - 1), sizeof(int), cudaMemcpyDeviceToHost);
                CHECK_CUDA(cudaMalloc(&d_outIndices, sizeof(int) * valid_count));
                gridSize = (size + blockSize - 1) / blockSize;
                copyPointCloudMaskToIndicesKernel << <gridSize, blockSize >> > (d_outIndices, d_mask, thrust::raw_pointer_cast(d_indices.data()), size);
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());
                cloud_out.alloc(valid_count);
                copyPointCloudKernel << <gridSize, blockSize >> > (cloud_in.x(), cloud_in.y(), cloud_in.z(),
                    cloud_out.x(), cloud_out.y(), cloud_out.z(), d_outIndices, valid_count);
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaFree(d_outIndices));
                CHECK_CUDA(cudaFree(d_mask));
                CHECK_CUDA(cudaFree(d_indicesTemp));

            }
            else
            {
                valid_count = d_indices.size();
                gridSize = (valid_count + blockSize - 1) / blockSize;
                cloud_out.alloc(valid_count);

                copyPointCloudKernel << <gridSize, blockSize >> > (cloud_in.x(), cloud_in.y(), cloud_in.z(),
                    cloud_out.x(), cloud_out.y(), cloud_out.z(), thrust::raw_pointer_cast(d_indices.data()), valid_count);
                CHECK_CUDA_KERNEL_ERROR();
                cudaDeviceSynchronize();
              
            }

        }    

        void copyPointCloud(const float* d_x, const float* d_y, const float* d_z,const int numPoints , GpuPointCloud& cloud_out, int* d_mask)
        {
            int size = numPoints;
            if (size == 0) {

                std::cerr << "[Warning] copyPointCloud received empty cloud!" << std::endl;
                return;
            }
            int blockSize = 256;
            int gridSize = (size + blockSize - 1) / blockSize;
            int valid_count = 0;
            int* d_outIndices;
            int* d_indices;
            CHECK_CUDA(cudaMalloc(&d_indices, sizeof(int) * size));
            scanExclusive(d_mask, d_indices, size);
            int last_mask, last_index;
            CHECK_CUDA(cudaMemcpy(&last_mask, d_mask + size - 1, sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(&last_index, d_indices + size - 1, sizeof(int), cudaMemcpyDeviceToHost));
            valid_count = last_index + last_mask;

            CHECK_CUDA(cudaMalloc(&d_outIndices, sizeof(int) * valid_count));
            copyPointCloudMaskToIndicesKernel << <gridSize, blockSize >> > (d_outIndices, d_mask, d_indices, size);
            CHECK_CUDA_KERNEL_ERROR();
            CHECK_CUDA(cudaDeviceSynchronize());
            cloud_out.alloc(valid_count);
            copyPointCloudKernel << <gridSize, blockSize >> > (d_x, d_y, d_z,
                cloud_out.x(), cloud_out.y(), cloud_out.z(), d_outIndices, valid_count);
            CHECK_CUDA_KERNEL_ERROR();
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaFree(d_indices));
            CHECK_CUDA(cudaFree(d_outIndices));
        }


    } // namespace cuda
} // namespace pcl