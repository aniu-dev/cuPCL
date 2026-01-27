#pragma once
#include "../types/device_cloud.h"
#include "../types/model_types.h"
#include <memory>
namespace pcl {
    namespace cuda {

        // 欧式聚类
        class EuclideanClusterExtraction  {
        public:
            EuclideanClusterExtraction();
            ~EuclideanClusterExtraction();
            EuclideanClusterExtraction(const EuclideanClusterExtraction&) = delete;
            EuclideanClusterExtraction& operator=(const EuclideanClusterExtraction&) = delete;
            EuclideanClusterExtraction(EuclideanClusterExtraction&&) noexcept;
            EuclideanClusterExtraction& operator=(EuclideanClusterExtraction&&) noexcept;
            void setInputCloud(const GpuPointCloud& input);
            void setClusterTolerance(const float& radius);        // 设置半径为0.1
            void setMinClusterSize(const int& min_cluster_size); // 设置查询点的邻域点个数  
            void setMaxClusterSize(const int& max_cluster_size); // 设置查询点的邻域点个数  
            void extract(std::vector<std::vector<int>>& clusters);

        private:
            struct Impl;
            std::unique_ptr<Impl> pimpl_;

        };
    }
}
