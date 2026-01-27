#pragma once
#include "filters.h"

namespace pcl {
    namespace cuda {

        // Ö±Í¨ÂË²¨
        class PassThrough : public Filter {
        public:
            PassThrough();
            ~PassThrough();
            PassThrough(const PassThrough&) = delete;
            PassThrough& operator=(const PassThrough&) = delete;
            PassThrough(PassThrough&&) noexcept;
            PassThrough& operator=(PassThrough&&) noexcept;
            void setFilterFieldName(const std::string &axis); // 'x', 'y', 'z'
            void setFilterLimits(const float& min_limit, const  float& max_limit);
            void setNegative(const bool& negative);
            void filter(GpuPointCloud& output) override;

        private:

            struct Impl;
            std::unique_ptr<Impl> pimpl_; 
        };


    }
}