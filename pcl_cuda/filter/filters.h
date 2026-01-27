#pragma once
#include "../types/device_cloud.h"
#include "../types/model_types.h"
#include <memory>
namespace pcl {
    namespace cuda {

        // ÂË²¨»ùÀà
        class Filter {
        public:
            virtual ~Filter() {}
            void setInputCloud(const GpuPointCloud* cloud) { input_ = cloud; }
            virtual void filter(GpuPointCloud& output) = 0; // ´¿Ðéº¯Êý

        protected:
            const GpuPointCloud* input_ = nullptr;
        };          
    } // namespace cuda
} // namespace pcl