#pragma once
#include <vector>

namespace pcl {
    namespace cuda {

        enum ModelType {
            SACMODEL_PLANE,
            SACMODEL_SPHERE,
            SACMODEL_LINE,
            SACMODEL_CIRCLE2D,
            SACMODEL_CIRCLE3D,
            SACMODEL_CYLINDER
        };

        struct ModelCoefficients {
            std::vector<float> values;
        };

    } // namespace cuda
} // namespace pcl