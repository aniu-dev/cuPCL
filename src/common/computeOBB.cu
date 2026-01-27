#include "pcl_cuda/common/common.h"
#include"internal/device_primitives.cuh"
#include "internal/error.cuh"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include <thrust/count.h>
namespace pcl {
    namespace cuda {
        __global__ void projectAndFindBoundsKernel(
            const float* __restrict__ d_in_x,
            const float* __restrict__ d_in_y,
            const float* __restrict__ d_in_z, // 输入点云  
            float* __restrict__ d_out,
            int num_points,
            float cx, float cy, float cz,       // 质心
            float r00, float r10, float r20,    // Matrix col 0 (Eigenvector X)
            float r01, float r11, float r21,    // Matrix col 1 (Eigenvector Y)
            float r02, float r12, float r22)   // Matrix col 2 (Eigenvector Z)                     
        {
            int tx = threadIdx.x;
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= num_points) return;
            // 读取点
            float px = d_in_x[tid];
            float py = d_in_y[tid];
            float pz = d_in_z[tid];

            // 去中心化
            float dx = px - cx;
            float dy = py - cy;
            float dz = pz - cz;

            // 投影到局部坐标系 (Dot product with eigenvectors)
            // local_x = dot(vec, eigen_vec_0)
            float lx = dx * r00 + dy * r10 + dz * r20;
            float ly = dx * r01 + dy * r11 + dz * r21;
            float lz = dx * r02 + dy * r12 + dz * r22;

            float3 min_temp = make_float3(INFINITY, INFINITY, INFINITY);
            float3 max_temp = make_float3(-INFINITY, -INFINITY, -INFINITY);

            min_temp.x = fmin(min_temp.x, lx);
            min_temp.y = fmin(min_temp.y, ly);
            min_temp.z = fmin(min_temp.z, lz);

            max_temp.x = fmax(max_temp.x, lx);
            max_temp.y = fmax(max_temp.y, ly);
            max_temp.z = fmax(max_temp.z, lz);
            min_temp.x = warpMin(min_temp.x);
            max_temp.x = warpMax(max_temp.x);

            min_temp.y = warpMin(min_temp.y);
            max_temp.y = warpMax(max_temp.y);

            min_temp.z = warpMin(min_temp.z);
            max_temp.z = warpMax(max_temp.z);


            const int warpId = tx / 32;
            const int laneId = tx % 32;
            const int warpNum = blockDim.x / 32;
            __shared__ float3 sx_min[32];
            __shared__ float3 sx_max[32];

            if (laneId == 0)
            {
                sx_min[warpId] = min_temp;
                sx_max[warpId] = max_temp;

            }
            __syncthreads();
            if (warpId == 0)
            {
                min_temp = (laneId < warpNum) ? sx_min[laneId] : make_float3(INFINITY, INFINITY, INFINITY);
                max_temp = (laneId < warpNum) ? sx_max[laneId] : make_float3(-INFINITY, -INFINITY, -INFINITY);

                min_temp.x = warpMin(min_temp.x);
                max_temp.x = warpMax(max_temp.x);

                min_temp.y = warpMin(min_temp.y);
                max_temp.y = warpMax(max_temp.y);

                min_temp.z = warpMin(min_temp.z);
                max_temp.z = warpMax(max_temp.z);
            }

            if (tx == 0)
            {
                atomicMinFloat(&d_out[0], min_temp.x);
                atomicMinFloat(&d_out[1], min_temp.y);
                atomicMinFloat(&d_out[2], min_temp.z);
                atomicMaxFloat(&d_out[0], max_temp.x);
                atomicMaxFloat(&d_out[1], max_temp.y);
                atomicMaxFloat(&d_out[2], max_temp.z);
            }
        }

        void computeOBB(
            const GpuPointCloud& cloud_in,
            const Eigen::Matrix3f& covariance,
            const PointXYZ& centroid,
            PointXYZ& min_point_OBB,
            PointXYZ& max_point_OBB,
            PointXYZ& position_OBB,
            Eigen::Matrix3f& rotational_matrix_OBB
        ) {
            int size = cloud_in.size();
            if (size == 0) {

                std::cerr << "[Warning] computeOBB received empty cloud!" << std::endl;
                return;
            }
            // 1. 特征分解
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance);
            Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();
            Eigen::Vector3f eigen_values = eigen_solver.eigenvalues();

            // 2. 修正坐标轴 (PCA)
            Eigen::Matrix3f transformation_matrix;
            transformation_matrix.col(0) = eigen_vectors.col(2);
            transformation_matrix.col(1) = eigen_vectors.col(1);
            transformation_matrix.col(2) = eigen_vectors.col(0);

            // 确保右手坐标系: v_z = v_x cross v_y
            if (transformation_matrix.col(0).cross(transformation_matrix.col(1)).dot(transformation_matrix.col(2)) < 0) {
                transformation_matrix.col(2) = -transformation_matrix.col(2);
            }
            rotational_matrix_OBB = transformation_matrix;

            float* d_buffer;
            CHECK_CUDA(cudaMalloc(&d_buffer, 6 * sizeof(float)));
            float h_init[6] = { FLT_MAX,FLT_MAX ,FLT_MAX ,-FLT_MAX ,-FLT_MAX ,-FLT_MAX };

            CHECK_CUDA(cudaMemcpy(d_buffer, h_init, 6 * sizeof(float), cudaMemcpyHostToDevice));
            // 5. 启动 Kernel
            int blockSize = 256;
            int gridSize = (size + blockSize - 1) / blockSize;

            projectAndFindBoundsKernel << <gridSize, blockSize >> > (
                cloud_in.x(), cloud_in.y(), cloud_in.z(),
                d_buffer, size, centroid.x, centroid.y, centroid.z,
                transformation_matrix(0, 0), transformation_matrix(1, 0), transformation_matrix(2, 0), // Col 0
                transformation_matrix(0, 1), transformation_matrix(1, 1), transformation_matrix(2, 1), // Col 1
                transformation_matrix(0, 2), transformation_matrix(1, 2), transformation_matrix(2, 2) // Col 2
                );
            CHECK_CUDA_KERNEL_ERROR();
            CHECK_CUDA(cudaDeviceSynchronize());
           

            float h_res[6];
            CHECK_CUDA(cudaMemcpy(h_res, d_buffer, 6 * sizeof(float), cudaMemcpyDeviceToHost));
            min_point_OBB.x = h_res[0];  max_point_OBB.x = h_res[3];
            min_point_OBB.y = h_res[1];  max_point_OBB.y = h_res[4];
            min_point_OBB.z = h_res[2];  max_point_OBB.z = h_res[5];

            // 计算局部中心
            Eigen::Vector3f local_center(
                (min_point_OBB.x + max_point_OBB.x) / 2.0f,
                (min_point_OBB.y + max_point_OBB.y) / 2.0f,
                (min_point_OBB.z + max_point_OBB.z) / 2.0f
            );

            // 计算世界坐标系下的 OBB 中心
            // WorldCenter = Centroid + Rotation * LocalCenter
            Eigen::Vector3f world_center = transformation_matrix * local_center;
            position_OBB.x = centroid.x + world_center.x();
            position_OBB.y = centroid.y + world_center.y();
            position_OBB.z = centroid.z + world_center.z();

            CHECK_CUDA(cudaFree(d_buffer));
        }
    } // namespace cuda
} // namespace pcl