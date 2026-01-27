#pragma once
#include "filters.h"
namespace pcl {
    namespace cuda {

        // Í³¼ÆÂË²¨
        class StatisticalOutlierRemoval : public Filter {
        public:
            StatisticalOutlierRemoval();
            ~StatisticalOutlierRemoval();
            StatisticalOutlierRemoval(const StatisticalOutlierRemoval&) = delete;
            StatisticalOutlierRemoval& operator=(const StatisticalOutlierRemoval&) = delete;
            StatisticalOutlierRemoval(StatisticalOutlierRemoval&&) noexcept;
            StatisticalOutlierRemoval& operator=(StatisticalOutlierRemoval&&) noexcept;
            void setMeanK(const int& nr_k);      
            void setStddevMulThresh(const float& stddev_mult);      
            void filter(GpuPointCloud& output) override;

        private:
            struct Impl;
            std::unique_ptr<Impl> pimpl_;

        };
    }
}
