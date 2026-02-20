#include "ransac.cuh"
#include "internal/cuda_math.cuh"
#include "pcl_cuda/common/common.h"
#include "internal/error.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <random> 
#include <ctime> 
#define MAX_CONST_SIZE 2048 
__constant__ float4 c_model_storage[MAX_CONST_SIZE];
namespace pcl {
    namespace cuda {  
        namespace device
        {

            template <typename Trait>
            __global__ void generateModelsKernel(
                const float* __restrict__  x_in, const float* __restrict__  y_in, const float* __restrict__  z_in,
                const int* d_sample_indices,
                typename Trait::Model* d_out_models, const int max_iterations
            )
            {
                const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                if (tid >= max_iterations) return;
                float3 pts[Trait::SAMPLE_SIZE];

                // 读取采样点
#pragma unroll
                for (int i = 0; i < Trait::SAMPLE_SIZE; ++i) {
                    int p_idx = d_sample_indices[tid * Trait::SAMPLE_SIZE + i];
                    pts[i] = make_float3(x_in[p_idx], y_in[p_idx], z_in[p_idx]);
                }
                d_out_models[tid] = Trait::compute(pts);
            }
            template <typename Trait>
            __global__ void ransacScoringKernel(
                const float* __restrict__  x_in, const float* __restrict__  y_in, const float* __restrict__  z_in, const int num_points,
                int* d_out_scores, const float threshold, const int max_iterations
            ) {

                const int tid = threadIdx.x;
                const int bid = blockIdx.x;
                const int tx = threadIdx.x + blockIdx.x * blockDim.x;
                extern __shared__ int sm_scores[];
                if (tx >= num_points) return;
                // 2. 初始化共享内存
                for (int i = tid; i < max_iterations; i += blockDim.x) {
                    sm_scores[i] = 0;
                }
                __syncthreads(); // 必须同步，确保清零完成

                float3 p = make_float3(x_in[tx], y_in[tx], z_in[tx]);
                const int warpId = tid / 32;
                const int laneId = tid % 32;
                const int warpNum = blockDim.x / 32;
#pragma unroll 4
                for (int i = 0; i < max_iterations; i += 4)
                {
                    // 批量加载 (利用 Constant Cache)
                    typename Trait::Model m0 = Trait::load_model(i);
                    typename Trait::Model m1 = Trait::load_model(i + 1);
                    typename Trait::Model m2 = Trait::load_model(i + 2);
                    typename Trait::Model m3 = Trait::load_model(i + 3);

                    // 批量计算 (延迟掩盖)
                    bool b0 = Trait::check_inlier(p, m0, threshold);
                    bool b1 = Trait::check_inlier(p, m1, threshold);
                    bool b2 = Trait::check_inlier(p, m2, threshold);
                    bool b3 = Trait::check_inlier(p, m3, threshold);

                    // 批量投票
                    unsigned int mask0 = __ballot_sync(0xFFFFFFFF, b0);
                    unsigned int mask1 = __ballot_sync(0xFFFFFFFF, b1);
                    unsigned int mask2 = __ballot_sync(0xFFFFFFFF, b2);
                    unsigned int mask3 = __ballot_sync(0xFFFFFFFF, b3);


                    // 统计个数并写入
                    // 只有 Warp 的第一个线程负责原子加，减少 32 倍冲突
                    if (laneId == 0) {
                        int warp_count0 = __popc(mask0); // 硬件指令计数
                        int warp_count1 = __popc(mask1); // 硬件指令计数
                        int warp_count2 = __popc(mask2); // 硬件指令计数
                        int warp_count3 = __popc(mask3); // 硬件指令计数

                        atomicAdd(&sm_scores[i], warp_count0);
                        atomicAdd(&sm_scores[i + 1], warp_count1);
                        atomicAdd(&sm_scores[i + 2], warp_count2);
                        atomicAdd(&sm_scores[i + 3], warp_count3);

                    }

                }

                __syncthreads(); // 等待所有 Warp 处理完所有模型

                // 将 Block 结果汇总到 Global Memory
                // 每个线程负责搬运一部分结果，减少串行
                for (int i = tid; i < max_iterations; i += blockDim.x) {
                    int block_total = sm_scores[i];
                    if (block_total > 0) {
                    
                        atomicAdd(&d_out_scores[i], block_total);
                    }
                }




            }
            template <typename Trait>
            __global__ void computePointsKernel(const float* x_in, const  float* y_in, const float* z_in, const int num_points,
                typename Trait::Model best_model, float threshold, int* is_inlier)
            {
                const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                if (tid >= num_points) return;

                float3 pt = make_float3(x_in[tid], y_in[tid], z_in[tid]);
                bool is_inlier_temp = Trait::check_inlier(pt, best_model, threshold);
                int  dist_out = is_inlier_temp ? 1 : 0;
                is_inlier[tid] = dist_out;
            }
            struct PlaneTrait {
                static const int SAMPLE_SIZE = 3;

                struct Model {
                    float4 coeff; // ax + by + cz + d = 0, 其中 (a,b,c) 为单位法向量
                };
                static void fillCoefficients(const Model& model, ModelCoefficients& coeffs) {

                    coeffs.values = { model.coeff.x, model.coeff.y, model.coeff.z, model.coeff.w };
                }

                __device__ static Model compute(const float3* p) {
                    //  计算两个向量
                    float3 v1 = p[1] - p[0];
                    float3 v2 = p[2] - p[0];

                    // 通过叉积计算法向量
                    float3 n = cross(v1, v2);
                    float len = length(n);

                    // 退化检查：如果三点共线，叉积模长接近 0
                    if (len < 1e-9f) {
                        // 返回一个无效平面（d设为极大值）
                        return { make_float4(0.0f, 0.0f, 1.0f, 1e10f) };
                    }

                    // 单位化法向量 (a, b, c)
                    float inv_len = 1.0f / len;
                    float3 unit_n = make_float3(n.x * inv_len, n.y * inv_len, n.z * inv_len);

                    //  计算 d 系数: d = -(a*x0 + b*y0 + c*z0)
                    float d = -(unit_n.x * p[0].x + unit_n.y * p[0].y + unit_n.z * p[0].z);

                    return { make_float4(unit_n.x, unit_n.y, unit_n.z, d) };
                }
                // 从常量内存加载
                __device__ __forceinline__ static Model load_model(int idx) {
                    Model m;
                    m.coeff = c_model_storage[idx]; // 平面只占1个槽位
                    return m;
                }

                // 快速检查内点
                __device__ __forceinline__ static bool check_inlier(float3 p, Model m, float threshold) {
                    // |ax+by+cz+d| < th
                    float dis = fabsf(p.x * m.coeff.x + p.y * m.coeff.y + p.z * m.coeff.z + m.coeff.w);
                    return dis < threshold;
                }

            };
            struct LineTrait {
                static const int SAMPLE_SIZE = 2;

                struct Model {
                    float4 origin;    // 直线上的一点 (P0)
                    float4 direction; // 直线的单位方向向量 (d)
                };
                static void fillCoefficients(const Model& model, ModelCoefficients& coeffs) {

                    coeffs.values = { model.origin.x, model.origin.y, model.origin.z, model.direction.x, model.direction.y, model.direction.z };
                }
                __device__ static Model compute(const float3* p) {
                    float3 diff = p[1] - p[0];
                    float len = length(diff);
                    // 单位化方向向量
                    float3 dir = make_float3(diff.x / len, diff.y / len, diff.z / len);
                    return { make_float4(p[0].x, p[0].y, p[0].z, 0.f),
                             make_float4(dir.x, dir.y, dir.z, 0.f) };
                }

                // 加载 (占用2个槽位)
                __device__ __forceinline__ static Model load_model(int idx) {
                    Model m;
                    m.origin = c_model_storage[idx * 2];     // 偶数位
                    m.direction = c_model_storage[idx * 2 + 1]; // 奇数位
                    return m;
                }

                __device__ __forceinline__ static bool check_inlier(float3 p, Model m, float threshold) {
                    float3 P0 = make_float3(m.origin.x, m.origin.y, m.origin.z);
                    float3 u = make_float3(m.direction.x, m.direction.y, m.direction.z);
                    float3 vec = p - P0;
                    float3 cp = cross(vec, u);
                    return dot(cp, cp) < (threshold * threshold);
                }
            };
            struct SphereTrait {
                static const int SAMPLE_SIZE = 4;
                struct Model {
                    float4 data; // x, y, z 为球心，w 为半径
                };
                static void fillCoefficients(const Model& model, ModelCoefficients& coeffs) {

                    coeffs.values = { model.data.x, model.data.y, model.data.z, model.data.w };
                }

                __device__ static Model compute(const float3* p) {

                    float3 v1 = p[1] - p[0];
                    float3 v2 = p[2] - p[0];
                    float3 v3 = p[3] - p[0];

                    // 构造 3x3 矩阵 A 和向量 B
                    // A 的每一行是 2 * (P_i - P_1)
                    float m11 = 2.0f * v1.x, m12 = 2.0f * v1.y, m13 = 2.0f * v1.z;
                    float m21 = 2.0f * v2.x, m22 = 2.0f * v2.y, m23 = 2.0f * v2.z;
                    float m31 = 2.0f * v3.x, m32 = 2.0f * v3.y, m33 = 2.0f * v3.z;

                    // B 的每一项是点到原点的平方距离之差
                    float b1 = dot(p[1], p[1]) - dot(p[0], p[0]);
                    float b2 = dot(p[2], p[2]) - dot(p[0], p[0]);
                    float b3 = dot(p[3], p[3]) - dot(p[0], p[0]);

                    // 计算矩阵 A 的行列式 (Determinant)
                    float det = m11 * (m22 * m33 - m23 * m32) -
                        m12 * (m21 * m33 - m23 * m31) +
                        m13 * (m21 * m32 - m22 * m31);

                    // 如果行列式接近0，说明四点共面或退化，无法成球
                    if (fabsf(det) < 1e-7f) {
                        return { make_float4(0.0f, 0.0f, 0.0f, -1.0f) }; // 返回无效半径
                    }

                    // 使用克莱姆法则解 xc, yc, zc
                    float detX = b1 * (m22 * m33 - m23 * m32) -
                        m12 * (b2 * m33 - m23 * b3) +
                        m13 * (b2 * m32 - m22 * b3);

                    float detY = m11 * (b2 * m33 - m23 * b3) -
                        b1 * (m21 * m33 - m23 * m31) +
                        m13 * (m21 * b3 - b2 * m31);

                    float detZ = m11 * (m22 * b3 - b2 * m32) -
                        m12 * (m21 * b3 - b2 * m31) +
                        b1 * (m21 * m32 - m22 * m31);

                    float invDet = 1.0f / det;
                    float3 center = make_float3(detX * invDet, detY * invDet, detZ * invDet);

                    // 计算半径：球心到任意采样点的距离
                    float radius = length(p[0] - center);

                    return { make_float4(center.x, center.y, center.z, radius) };
                }
                __device__ __forceinline__ static Model load_model(int idx) {
                    Model m;
                    m.data = c_model_storage[idx]; // 平面只占1个槽位
                    return m;
                }
                __device__ static float check_inlier(float3 p, Model m, float threshold)
                {
                    float dx = p.x - m.data.x;
                    float dy = p.y - m.data.y;
                    float dz = p.z - m.data.z;
                    float dist_sq = dx * dx + dy * dy + dz * dz;
                    float r = m.data.w;

                    float r_min = r - threshold;
                    float r_max = r + threshold;
                    float min_sq = (r_min < 0.0f) ? 0.0f : r_min * r_min;
                    float max_sq = r_max * r_max;

                    return (dist_sq >= min_sq) && (dist_sq <= max_sq);
                }
            };
            struct Circle2DTrait {
                static const int SAMPLE_SIZE = 3;
                struct Model { float4 data; }; // x, y, radius
                static void fillCoefficients(const Model& model, ModelCoefficients& coeffs) {

                    coeffs.values = { model.data.x, model.data.y, model.data.z };
                }
                __device__ static Model compute(const float3* p) {
                    float x1 = p[0].x, y1 = p[0].y;
                    float x2 = p[1].x, y2 = p[1].y;
                    float x3 = p[2].x, y3 = p[2].y;

                    float D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
                    if (fabsf(D) < 1e-6f) return { make_float4(0,0,0,0) }; // 共线

                    float cx = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / D;
                    float cy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / D;
                    float r = sqrtf((x1 - cx) * (x1 - cx) + (y1 - cy) * (y1 - cy));
                    return { make_float4(cx, cy, r,0) };
                }
                __device__ __forceinline__ static Model load_model(int idx) {
                    Model m;
                    m.data = c_model_storage[idx]; // 平面只占1个槽位
                    return m;
                }
                __device__ static float check_inlier(float3 p, Model m, float threshold)
                {
                    float dx = p.x - m.data.x;
                    float dy = p.y - m.data.y;

                    float dist_sq = dx * dx + dy * dy;
                    float r = m.data.z;

                    float r_min = r - threshold;
                    float r_max = r + threshold;
                    float min_sq = (r_min < 0.0f) ? 0.0f : r_min * r_min;
                    float max_sq = r_max * r_max;

                    return (dist_sq >= min_sq) && (dist_sq <= max_sq);
                }
            };
            struct Circle3DTrait {
                static const int SAMPLE_SIZE = 3;

                struct Model {
                    float4 center; // 圆心
                    float4 normal; // 法向量
                };

                static void fillCoefficients(const Model& model, ModelCoefficients& coeffs) {

                    coeffs.values = {
                        model.center.x, model.center.y, model.center.z,
                        model.center.w,
                        model.normal.x, model.normal.y, model.normal.z
                    };
                }

                __device__ static Model compute(const float3* p) {
                    // 向量边缘
                    float3 v1 = p[1] - p[0];
                    float3 v2 = p[2] - p[0];

                    // 计算法向量
                    float3 v1xv2 = cross(v1, v2);
                    float area2 = dot(v1xv2, v1xv2);

                    // 共线检查
                    if (area2 < 1e-9f) {
                        return { make_float4(0,0,0,-1), make_float4(0,0,1,0) };
                    }

                    float3 normal = normalize(v1xv2);


                    float v1_sq = dot(v1, v1);
                    float v2_sq = dot(v2, v2);


                    float3 part1 = make_float3(
                        v2_sq * v1.x - v1_sq * v2.x,
                        v2_sq * v1.y - v1_sq * v2.y,
                        v2_sq * v1.z - v1_sq * v2.z
                    );



                    float3 numerator = cross(v1xv2, part1); // 交换了叉积顺序
                    float denom = 2.0f * area2;

                    float3 center = make_float3(
                        p[0].x + numerator.x / denom,
                        p[0].y + numerator.y / denom,
                        p[0].z + numerator.z / denom
                    );

                    // 3. 计算半径
                    float r = length(p[0] - center);

                    return { make_float4(center.x ,center.y,center.z,r),
                        make_float4(normal.x ,normal.y,normal.z,0.0) };
                }

                __device__ __forceinline__ static Model load_model(int idx) {
                    Model m;
                    m.center = c_model_storage[idx * 2];     // 偶数位
                    m.normal = c_model_storage[idx * 2 + 1]; // 奇数位
                    return m;
                }
                __device__ static float check_inlier(float3 p, Model m, float threshold) {

                    float radius = m.center.w;
                    if (radius < 0) return 1e10f;
                    float3 center = { m.center.x ,m.center.y,m.center.z };
                    float3 normal = { m.normal.x ,m.normal.y,m.normal.z };

                    float3 diff = p - center;

                    float h = dot(diff, normal);

                    if (fabsf(h) > threshold) return false;

                    float dist_sq_total = dot(diff, diff);
                    float dist_sq_proj = dist_sq_total - h * h;


                    if (dist_sq_proj < 0.0f) dist_sq_proj = 0.0f;
                    float d_proj = sqrtf(dist_sq_proj);
                    float err_radial = fabsf(d_proj - radius);

                    return (h * h + err_radial * err_radial) < (threshold * threshold);
                }
            };

            template <typename Trait>
            void computeModel(
                const GpuPointCloud* input,
                const int& max_iterations,
                const float& threshold,
                GpuPointCloud& output,
                ModelCoefficients& coefficients)
            {
                int numPoints = input->size();
                // 分配设备内存
                std::vector<int> h_indices(max_iterations * Trait::SAMPLE_SIZE);
                std::mt19937 gen(time(0));
                std::uniform_int_distribution<> dis(0, numPoints - 1);
                for (int& idx : h_indices) idx = dis(gen);

                thrust::device_vector<int> d_indices = h_indices;
                thrust::device_vector<typename Trait::Model> d_models(max_iterations);

                // 生成模型 (Kernel 1) 
                int blockSize = 256;
                int gridSize = (max_iterations + blockSize - 1) / blockSize;

                generateModelsKernel<Trait> << <gridSize, blockSize >> > (
                    input->x(), input->y(), input->z(),
                    thrust::raw_pointer_cast(d_indices.data()), thrust::raw_pointer_cast(d_models.data()) , max_iterations
                    );
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());
                thrust::host_vector<typename Trait::Model> h_models = d_models;

                std::vector<float4> packed_buffer;
                for (const auto& m : h_models) {
                    // 强制类型转换，将模型数据视为 float4 数组
                    const float4* ptr = reinterpret_cast<const float4*>(&m);
                    // 计算该模型占几个 float4 (Plane=1, Line=2)
                    int slots = sizeof(typename Trait::Model) / sizeof(float4);
                    for (int k = 0; k < slots; k++) packed_buffer.push_back(ptr[k]);
                }

                if (packed_buffer.size() > MAX_CONST_SIZE) {
                    std::cerr << "Error: Too many models for constant memory!" << std::endl;
                    return;
                }
                CHECK_CUDA(cudaMemcpyToSymbol(c_model_storage, packed_buffer.data(), packed_buffer.size() * sizeof(float4)));

                thrust::device_vector<int> d_scores(max_iterations, 0);

                gridSize = (numPoints + blockSize - 1) / blockSize;
                const int BATCH_SIZE = 128;
                int shared_mem_size = (256 / 32) * BATCH_SIZE * sizeof(int); // 8 * 128 * 4 = 4096 bytes
                ransacScoringKernel<Trait> << <gridSize, blockSize, shared_mem_size >> > (
                    input->x(), input->y(), input->z(), numPoints,
                    thrust::raw_pointer_cast(d_scores.data()), threshold, max_iterations
                    );
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());
                auto max_iter = thrust::max_element(d_scores.begin(), d_scores.end());
                int max_index = max_iter - d_scores.begin();
                typename Trait::Model modelBest = h_models[max_index];

                thrust::device_vector<int> d_mask(numPoints);
                computePointsKernel <Trait> << <gridSize, blockSize >> > (input->x(), input->y(), input->z(), 
                    numPoints, modelBest, threshold, thrust::raw_pointer_cast(d_mask.data()) );
                CHECK_CUDA_KERNEL_ERROR();
                CHECK_CUDA(cudaDeviceSynchronize());

                typename Trait::fillCoefficients(modelBest, coefficients);

                copyPointCloud(*input, output, thrust::raw_pointer_cast(d_mask.data()));
            }

            void launchRansacSegment(
                const GpuPointCloud* input,
                const ModelType& model_type,
                const int& max_iterations,
                const float& threshold,
                GpuPointCloud& output,
                ModelCoefficients& coefficients
            )
            {
                size_t size = input->size();
                if (size == 0) {

                    std::cerr << "[Warning] RansacSegment received empty cloud!" << std::endl;
                    return;
                }
                switch (model_type)
                {
                case SACMODEL_PLANE:

                    computeModel<PlaneTrait>(input, max_iterations, threshold, output, coefficients);
                    break;
                case SACMODEL_LINE:
                    computeModel<LineTrait>(input, max_iterations, threshold, output, coefficients);
                    break;
                case SACMODEL_CIRCLE2D:
                    computeModel<Circle2DTrait>(input, max_iterations, threshold, output, coefficients);
                    break;
                case SACMODEL_CIRCLE3D:
                    computeModel<Circle3DTrait>(input, max_iterations, threshold, output, coefficients);
                    break;
                case SACMODEL_SPHERE:
                    computeModel<SphereTrait>(input, max_iterations, threshold, output, coefficients);
                    break;
                default:
                    break;
                }
            }

        }
    } // namespace cuda
} // namespace pcl




































