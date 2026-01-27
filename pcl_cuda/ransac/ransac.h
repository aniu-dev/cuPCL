#pragma once
#include "../types/device_cloud.h"
#include "../types/model_types.h"
#include <memory>
namespace pcl {
    namespace cuda {

        class Ransac {
        public:
            Ransac();
            ~Ransac();
            Ransac(const Ransac&) = delete;
            Ransac& operator=(const Ransac&) = delete;
            Ransac(Ransac&&) noexcept;
            Ransac& operator=(Ransac&&) noexcept;

            void setInput(const GpuPointCloud* cloud);
            void setDistanceThreshold(float threshold);
            void setMaxIterations(int max_iterations);
            void setModelType(ModelType model_type);
            void segment(GpuPointCloud& cloud, ModelCoefficients& model_coeffs);

        private:
            struct Impl;
            std::unique_ptr<Impl> pimpl_;

           
        };

    } // namespace cuda
} // namespace pcl