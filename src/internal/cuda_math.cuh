#pragma once
#include <cuda_runtime.h>
#include <math.h>
namespace pcl {
    namespace cuda
    {
        #define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
        #define INT4(value) (reinterpret_cast<int4*>(&(value))[0])

        __host__ __device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
            return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
        }
        __host__ __device__ __forceinline__ float3 operator+=(const float3& a, const float3& b) {
            return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
        }
        __host__ __device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
            return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
        }
        __host__ __device__ __forceinline__ float3 operator*(const float3& a, const float3& b) {
            return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
        }
        __host__ __device__ __forceinline__ float3 operator*(const float3& a, float s) {
            return make_float3(a.x * s, a.y * s, a.z * s);
        }
        __host__ __device__ __forceinline__ float3 operator/(const float3& a, float s) {
            float inv = 1.0f / s;
            return a * inv;
        }
        __host__ __device__ __forceinline__ float3 operator/(const float3& a, const float3& b) {
            return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
        }
        __host__ __device__ __forceinline__ float dot(const float3 a, const  float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
        __host__ __device__ __forceinline__ float dot4(const float4 a, const float4 b) {
            return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        __host__ __device__ __forceinline__ float3 cross(const float3 a, const  float3 b) {
            return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
        }
        __host__ __device__ __forceinline__ float length(const float3 a) { return sqrtf(dot(a, a)); }
        __host__ __device__ __forceinline__ float3 normalize(const float3 a) {
            float invLen = 1.0/sqrtf(dot(a, a) + 1e-9f);
            return make_float3(a.x * invLen, a.y * invLen, a.z * invLen);
        }
        __host__ __device__ __forceinline__ float3 fminf(const float3& a, const float3& b) {
            return make_float3(::fminf(a.x, b.x), ::fminf(a.y, b.y), ::fminf(a.z, b.z));
        }

        __host__ __device__ __forceinline__ float3 fmaxf(const float3& a, const float3& b) {
            return make_float3(::fmaxf(a.x, b.x), ::fmaxf(a.y, b.y), ::fmaxf(a.z, b.z));
        }
    }
}