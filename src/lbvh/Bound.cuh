#pragma once
#include "internal/cuda_math.cuh" 
namespace pcl {
    namespace cuda
    {
        struct AABB {
            float3 min; // 也就是 lower
            float3 max; // 也就是 upper

            //  默认构造：初始化为"无效"状态 (min无穷大，max无穷小)
            __host__ __device__ AABB() {
                min = make_float3(1e30f, 1e30f, 1e30f);
                max = make_float3(-1e30f, -1e30f, -1e30f);
            }

            // 点云专用构造：一个点就是一个包围盒
            __host__ __device__ AABB(const float3& p) {
                min = p;
                max = p;
            }

            //完整构造
            __host__ __device__ AABB(const float3& min_v, const float3& max_v) {
                min = min_v;
                max = max_v;
            }

            // 计算中心点 (Morton Code 排序必须用)
            __host__ __device__ float3 center() const {
                return (min + max) * 0.5f;
            }

            // 合并/吞噬另一个盒子 (构建 BVH 树的核心)
            __host__ __device__  void merge(const AABB& b) {
                min = fminf(min, b.min); // 这里的 fminf 来自 cuda_math.h
                max = fmaxf(max, b.max);
            }

            // 判断两个盒子是否相交 (查询/遍历树的核心)
            __host__ __device__ bool intersects(const AABB& b) const {
                if (max.x < b.min.x || min.x > b.max.x) return false;
                if (max.y < b.min.y || min.y > b.max.y) return false;
                if (max.z < b.min.z || min.z > b.max.z) return false;
                return true;
            }

            // 7. 归一化坐标 (计算 Morton Code 必须用)
            // 把世界坐标映射到 [0, 1] 区间
            __host__ __device__ float3 normCoord(const float3& pos) const {
                float3 diff = max - min;
                return (pos - min) / diff;
            }
        };
    }
}