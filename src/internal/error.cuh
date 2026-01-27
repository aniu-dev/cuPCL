#pragma once
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>
#include <sstream>
#include <iostream>

namespace pcl {
    namespace cuda {

        // 自定义异常
        class CudaException : public std::runtime_error {
        public:
            CudaException(const std::string& msg) : std::runtime_error(msg) {}
        };

        namespace internal {
            inline void check_cuda_error(cudaError_t error, const char* file, int line, const char* func_name) {
                if (error != cudaSuccess) {
                    std::stringstream ss;
                    ss << "\n[CUDA ERROR] at file " << file << ":" << line << "\n"
                        << "  Function: " << func_name << "\n"
                        << "  Error Name: " << cudaGetErrorName(error) << "\n"
                        << "  Error String: " << cudaGetErrorString(error) << "\n";
                    throw CudaException(ss.str());
                }
            }
        }

        // 宏定义
#define CHECK_CUDA(call) do { \
        cudaError_t err = call; \
        pcl::cuda::internal::check_cuda_error(err, __FILE__, __LINE__, #call); \
    } while(0)

#define CHECK_CUDA_KERNEL_ERROR() do { \
        cudaError_t err = cudaGetLastError(); \
        pcl::cuda::internal::check_cuda_error(err, __FILE__, __LINE__, "Kernel Launch"); \
    } while(0)

    } // namespace cuda
} // namespace pcl