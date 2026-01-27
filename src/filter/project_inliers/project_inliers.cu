#include "project_inliers.cuh"
#include "internal/error.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
namespace pcl
{
    namespace cuda
    {
        namespace device
        {   
            struct PlaneModel {
                __device__ __forceinline__
                    static void project(float x, float y, float z,
                        float& out_x, float& out_y, float& out_z,
                        const float* coeff)
                {
                    float A = coeff[0], B = coeff[1], C = coeff[2], D = coeff[3];
                    float norm_sq = A * A + B * B + C * C;
                    if (norm_sq > 1e-6f) {
                        float val = A * x + B * y + C * z + D;
                        out_x = x - (A * val) / norm_sq;
                        out_y = y - (B * val) / norm_sq;
                        out_z = z - (C * val) / norm_sq;
                    }
                    else {
                        out_x = x; out_y = y; out_z = z;
                    }
                }
            };

            struct SphereModel {
                __device__ __forceinline__
                    static void project(float x, float y, float z,
                        float& out_x, float& out_y, float& out_z,
                        const float* coeff)
                {
                    float cx = coeff[0], cy = coeff[1], cz = coeff[2], r = coeff[3];
                    float dx = x - cx;
                    float dy = y - cy;
                    float dz = z - cz;
                    float dist = sqrtf(dx * dx + dy * dy + dz * dz);

                    if (dist > 1e-6f) {
                        float factor = r / dist;
                        out_x = cx + dx * factor;
                        out_y = cy + dy * factor;
                        out_z = cz + dz * factor;
                    }
                    else {
                        out_x = x; out_y = y; out_z = z;
                    }

                }

            };

            struct LineModel {
                __device__ __forceinline__
                    static void project(float x, float y, float z,
                        float& out_x, float& out_y, float& out_z,
                        const float* coeff)
                {
                    // 点 (px, py, pz)，方向 (m, n, p)
                    float px = coeff[0], py = coeff[1], pz = coeff[2];
                    float m = coeff[3], n = coeff[4], p = coeff[5];

                    float dx = x - px;
                    float dy = y - py;
                    float dz = z - pz;

                    float k = dx * m + dy * n + dz * p;
                    float norm_dir = m * m + n * n + p * p;

                    if (norm_dir > 1e-6f) {
                        float t = k / norm_dir;
                        out_x = px + t * m;
                        out_y = py + t * n;
                        
                        out_z = pz + t * p;
                    }
                    else {
                        out_x = x; out_y = y; out_z = z;
                    }
                }
            };
            struct CylinderModel {
                __device__ __forceinline__
                    static void project(float x, float y, float z,
                        float& out_x, float& out_y, float& out_z,
                        const float* coeff)
                {
                    float px = coeff[0], py = coeff[1], pz = coeff[2];
                    float m = coeff[3], n = coeff[4], p = coeff[5];
                    float r = coeff[6];

                    //  计算轴线上最近点 (垂足)
                    float dx = x - px, dy = y - py, dz = z - pz;
                    float k = dx * m + dy * n + dz * p;
                    float norm_dir = m * m + n * n + p * p;

                    if (norm_dir > 1e-6f) {
                        float t = k / norm_dir;
                        float mx = px + t * m;
                        float my = py + t * n;
                        float mz = pz + t * p;

                        // 从垂足指向当前点的向量
                        float vec_x = x - mx;
                        float vec_y = y - my;
                        float vec_z = z - mz;
                        float vec_norm = sqrtf(vec_x * vec_x + vec_y * vec_y + vec_z * vec_z);

                        //缩放到半径长度
                        if (vec_norm > 1e-6f) {
                            float factor = r / vec_norm;
                            out_x = mx + vec_x * factor;
                            out_y = my + vec_y * factor;
                            out_z = mz + vec_z * factor;
                        }
                        else {
                            out_x = x; out_y = y; out_z = z;
                        }
                    }
                    else {
                        out_x = x; out_y = y; out_z = z;
                    }
                }
            };

            template <typename ModelPolicy>
            __global__ void projectInliersKernel(
                const float* __restrict__ d_in_x,
                const float* __restrict__ d_in_y,
                const float* __restrict__ d_in_z,
                float* __restrict__ d_out_x,
                float* __restrict__ d_out_y,
                float* __restrict__ d_out_z,
                const float* __restrict__ coefficients, 
                const int numPoints)
            {
                const int tid = threadIdx.x + blockDim.x * blockIdx.x;
                if (tid >= numPoints) return;

                // 读取数据
                float x = d_in_x[tid];
                float y = d_in_y[tid];
                float z = d_in_z[tid];
                float ox, oy, oz;

                ModelPolicy::project(x, y, z, ox, oy, oz, coefficients);

                //写入数据
                d_out_x[tid] = ox;
                d_out_y[tid] = oy;
                d_out_z[tid] = oz;
            }
            void launchProjectInliersFilter(
                const GpuPointCloud* input,
                const ModelType& model_type,
                const ModelCoefficients& model_cofficients,
                GpuPointCloud& output
            )
            {
         
                int size = input->size();
                if (size == 0) {

                    std::cerr << "[Warning] ProjectInliersFilter received empty cloud!" << std::endl;
                    return;
                }

                int blockSize = 512;
                int gridSize = (size + blockSize - 1) / blockSize;
                int cofficients_size = model_cofficients.values.size();
                output.alloc(size);

                float* cofficients;
                CHECK_CUDA(cudaMalloc(&cofficients, sizeof(float) * cofficients_size));
                CHECK_CUDA(cudaMemcpy(cofficients, model_cofficients.values.data(), sizeof(float) * cofficients_size, cudaMemcpyHostToDevice));

                switch (model_type)
                {
                case SACMODEL_PLANE:
                    if (cofficients_size != 4) return;
                    projectInliersKernel<PlaneModel> << <gridSize, blockSize >> > (input->x(), input->y(), input->z(), output.x(), output.y(), output.z(),
                        cofficients, size);                   
                    break;
                case SACMODEL_SPHERE:
                    if (cofficients_size != 4) return;
                    projectInliersKernel<SphereModel> << <gridSize, blockSize >> > (input->x(), input->y(), input->z(), output.x(), output.y(), output.z(),
                        cofficients, size);                   
                    break;
                case SACMODEL_LINE:
                    if (cofficients_size != 6) return;
                    projectInliersKernel <LineModel><< <gridSize, blockSize >> > (input->x(), input->y(), input->z(), output.x(), output.y(), output.z(),
                        cofficients, size);                  
                    break;
                case SACMODEL_CYLINDER:
                    if (cofficients_size != 7) return;

                    projectInliersKernel<CylinderModel> << <gridSize, blockSize >> > (input->x(), input->y(), input->z(), output.x(), output.y(), output.z(),
                        cofficients, size);
                    
                    break;
                default:
                    break;
                }
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成	
                CHECK_CUDA(cudaFree(cofficients));
            };
        }
        
    }
}