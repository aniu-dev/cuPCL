#pragma once
#include "filters.h"
namespace pcl {
    namespace cuda {

        // 半径滤波
        class RadiusOutlierRemoval : public Filter {
        public:
            RadiusOutlierRemoval();
            ~RadiusOutlierRemoval();
            RadiusOutlierRemoval(const RadiusOutlierRemoval&) = delete;
            RadiusOutlierRemoval& operator=(const RadiusOutlierRemoval&) = delete;
            RadiusOutlierRemoval(RadiusOutlierRemoval&&) noexcept;
            RadiusOutlierRemoval& operator=(RadiusOutlierRemoval&&) noexcept;
            void setRadiusSearch(const float &radius);        // 设置半径为0.1
            void setMinNeighborsInRadius(const int &min_pts); // 设置查询点的邻域点个数      
            void filter(GpuPointCloud& output) override;

        private:
            struct Impl;
            std::unique_ptr<Impl> pimpl_;

        };
    }
}
