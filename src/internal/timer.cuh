#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <string>

namespace pcl {
    namespace cuda {

        struct GpuTimer {
            cudaEvent_t start;
            cudaEvent_t stop;

            GpuTimer() {
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
            }

            ~GpuTimer() {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }

            void Start(cudaStream_t stream = 0) {
                cudaEventRecord(start, stream);
            }

            void Stop(cudaStream_t stream = 0) {
                cudaEventRecord(stop, stream);
            }

            float Elapsed() {
                cudaEventSynchronize(stop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                return milliseconds;
            }

            void Print(const std::string& name = "Kernel") {
                float ms = Elapsed();
                std::cout << "[" << name << "] Time: " << ms << " ms" << std::endl;
            }
        };

    } // namespace cuda
} // namespace pcl