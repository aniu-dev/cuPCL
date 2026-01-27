#pragma once
#include <device_launch_parameters.h>

namespace pcl {
    namespace cuda {
        template <typename T>
        __device__ __forceinline__ T warpSum(T val) {
            for (int mask = 16; mask > 0; mask >>= 1)
                val += __shfl_xor_sync(0xffffffff, val, mask);
            return val;
        }

        __device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
            float old;
            old = *addr;
            while (old > value && atomicCAS((unsigned int*)addr,
                __float_as_uint(old),
                __float_as_uint(value)) != __float_as_uint(old)) {
                old = *addr;
            }
            return old;
        }

        __device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
            float old;
            old = *addr;
            while (old < value&& atomicCAS((unsigned int*)addr,
                __float_as_uint(old),
                __float_as_uint(value)) != __float_as_uint(old)) {
                old = *addr;
            }
            return old;
        }

        template <typename T>
        __device__ __forceinline__ T warpMin(T val) {
            for (int mask = 16; mask > 0; mask >>= 1)
                val = fmin(val, __shfl_xor_sync(0xffffffff, val, mask));
            return val;
        }

        template <typename T>
        __device__ __forceinline__ T warpMax(T val) {
            for (int mask = 16; mask > 0; mask >>= 1)
                val = fmax(val, __shfl_xor_sync(0xffffffff, val, mask));
            return val;
        }
        

    } // namespace cuda
} // namespace pcl