
#include "pcl_cuda/segmentation/extract_clusters.h"
#include "extract_clusters.cuh"
namespace pcl
{
    namespace cuda
    {

        struct EuclideanClusterExtraction::Impl
        {
            float radius_ = 0.1f;
            int min_cluster_size_ = 1000;
            int max_cluster_size_ = 10000;
            const GpuPointCloud* input_ = nullptr;
        };


        EuclideanClusterExtraction::EuclideanClusterExtraction() : pimpl_(std::make_unique<Impl>()) {}
        EuclideanClusterExtraction::~EuclideanClusterExtraction() = default;
        // 移动构造函数
        EuclideanClusterExtraction::EuclideanClusterExtraction(EuclideanClusterExtraction&&) noexcept = default;
        EuclideanClusterExtraction& EuclideanClusterExtraction::operator=(EuclideanClusterExtraction&&) noexcept = default;

        void EuclideanClusterExtraction::setInputCloud(const GpuPointCloud& input) { pimpl_->input_ = &input; };
        void EuclideanClusterExtraction::setClusterTolerance(const float& radius) { pimpl_->radius_ = radius; };
        
        void EuclideanClusterExtraction::setMinClusterSize(const int& min_cluster_size) { pimpl_->min_cluster_size_ = min_cluster_size; };
        void EuclideanClusterExtraction::setMaxClusterSize(const int& max_cluster_size) { pimpl_->max_cluster_size_ = max_cluster_size; };
        void EuclideanClusterExtraction::extract(std::vector<std::vector<int>>& clusters)
        {
            device::launchEuclideanClusterExtraction(
                pimpl_->input_,
                pimpl_->radius_,
                pimpl_->min_cluster_size_,
                pimpl_->max_cluster_size_,
                clusters);
        }





    }


}



