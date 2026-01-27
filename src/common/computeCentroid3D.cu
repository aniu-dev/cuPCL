#include "pcl_cuda/common/common.h"
#include "pcl_cuda/common/primitives.h"
namespace pcl {
    namespace cuda {
        // 质心可类似实现...
        void computeCentroid3D(const GpuPointCloud& cloud_in, PointXYZ& centroid)
        {
            size_t size = cloud_in.size();
            if (size == 0) {
                              
                std::cerr << "[Warning] computeCentroid3D received empty cloud!" << std::endl;
                return;
            }
                                           
            PointXYZ valPt;
            reduceWarpSum3D(cloud_in, size, valPt);
            float inv_N = 1.0f / static_cast<float>(size);
            centroid.x = valPt.x * inv_N;
            centroid.y = valPt.y * inv_N;
            centroid.z = valPt.z * inv_N;

            // 如果输入的点云里本身包含 NaN 点，或者数据溢出，算出来的质心可能是无效的
            if (!std::isfinite(centroid.x) || !std::isfinite(centroid.y) || !std::isfinite(centroid.z))
            {  
                 std::cerr << "[Warning] Computed centroid is NaN or Inf. Input cloud might contain invalid points." << std::endl;
            }


        }
    } // namespace cuda
} // namespace pcl