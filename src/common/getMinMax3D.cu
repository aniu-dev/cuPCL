#include "pcl_cuda/common/common.h"
#include "pcl_cuda/common/primitives.h"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include <thrust/count.h>
namespace pcl {
    namespace cuda {
        void getMinMax3D(const GpuPointCloud& cloud_in, PointXYZ& min_pt, PointXYZ& max_pt)
        {
            size_t size = cloud_in.size();
            if (size == 0) {

                std::cerr << "[Warning] getMinMax3D received empty cloud!" << std::endl;
                return;
            }
            PointXYZ minPt, maxPt;

            reduceMinMax3D(cloud_in, size, minPt, maxPt);
            min_pt.x = minPt.x;  max_pt.x = maxPt.x;
            min_pt.y = minPt.y;  max_pt.y = maxPt.y;
            min_pt.z = minPt.z;  max_pt.z = maxPt.z;
        }

    } // namespace cuda
} // namespace pcl