#pragma once
#include "filters.h"

namespace pcl {
    namespace cuda {

        class ProjectInliersFilter : public Filter
        {

        public:
            ProjectInliersFilter();
            ~ProjectInliersFilter();
            ProjectInliersFilter(const ProjectInliersFilter&) = delete;
            ProjectInliersFilter& operator=(const ProjectInliersFilter&) = delete;
            ProjectInliersFilter(ProjectInliersFilter&&) noexcept;
            ProjectInliersFilter& operator=(ProjectInliersFilter&&) noexcept;
            void setModelType(const ModelType &model_type);
            void setModelCoefficients(const ModelCoefficients &model_cofficients);
            void filter(GpuPointCloud& output) override;

        private:
            struct Impl;
            std::unique_ptr<Impl> pimpl_;
          
        };
    }
}