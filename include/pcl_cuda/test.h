
#include "cuPCL.hpp"

void compute3DCentroid_test(size_t numPoints, float range);
void getMinMax3D_test(size_t numPoints, float range);
void computeCentroidAndCovariance_test(size_t numPoints, float range);
void computeOBB_test(size_t numPoints, float range);
void PassThrough_test(size_t numPoints, float range);
void transformPointCloud_test(size_t numPoints, float range);
void ransacPlane_test(size_t numPoints);
void ransacLine_test(size_t numPoints);
void ransacSphere_test(size_t numPoints);
void ransacCircle2D_test(size_t numPoints);
void ransacCircle3D_test(size_t numPoints);
void pointCloudToDeepMat_test(size_t numPoints);
void projectPlane_test(size_t numPoints, float range);
void projectCylinder_test(size_t numPoints, float range);
void projectLine_test(size_t numPoints, float range);
void projectSphere_test(size_t numPoints, float range);
void NormalEstimation_test(size_t numPoints, float range);
void RadiusOutlierRemoval_test(size_t numPoints, float range);
void VoxelGrid_test(size_t numPoints, float range);
void StatisticalOutlierRemoval_test(size_t numPoints, float range);
void ICP_test(size_t numPoints, float range);
void EuclideanClusterExtraction_test(size_t numPoints);
void copyPointCloud_test(size_t numPoints, float range);