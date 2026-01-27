#pragma once
#include "../types/device_cloud.h"

namespace pcl {
    namespace cuda {
        

        // 示例：规约求和
        float reduceBlockSum(const float* d_data, size_t size);
        float reduceWarpSum(const float* d_data, size_t size);
        void reduceWarpSum3D(const GpuPointCloud& cloud, size_t size, PointXYZ& Val);
        // 示例：规约求最大值
        float reduceMax(const float* d_data, size_t size);
        // 示例：规约求最小值
        float reduceMin(const float* d_data, size_t size);
        // 示例：规约求最小值最大值
        void reduceMinMax(const float* d_data, size_t size, float& minVal, float& maxVal);
        void reduceMinMax3D(const GpuPointCloud& cloud, size_t size, PointXYZ& minVal, PointXYZ& maxVal);

        // 示例：排他前缀和
        void scanExclusive(const int* d_data, int* d_out, size_t size);
        // 示例：包含前缀和 未实现
        //void scanInclusive(const int* d_data, int* d_out, size_t size);

        //排序算法 未实现
        //void radixSort(const int* d_data, int* d_out, size_t size);
    } // namespace cuda
} // namespace pcl