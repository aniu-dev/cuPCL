#pragma once
#include "../types/device_cloud.h"
#include "../types/model_types.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>
namespace pcl {
    namespace cuda {
        class NormalEstimation {
        public:
            NormalEstimation();
            ~NormalEstimation();
            NormalEstimation(const NormalEstimation&) = delete;
            NormalEstimation& operator=(const NormalEstimation&) = delete;
            NormalEstimation(NormalEstimation&&) noexcept;
            NormalEstimation& operator=(NormalEstimation&&) noexcept;
            void setInputCloud(const GpuPointCloud& input); // 'x', 'y', 'z'
            void setKSearch(const int nr_k);
            void compute(GpuPointCloudNormal& output);
        private:

            struct Impl;
            std::unique_ptr<Impl> pimpl_;
        };

    } // namespace cuda
} // namespace pcl