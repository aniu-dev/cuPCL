
#include "include/pcl_cuda/test.h"
int main()
{
	size_t numPoints = 1000000;
	float range = 50.0;
	//getMinMax3D_test(numPoints, range);
	//compute3DCentroid_test(numPoints, range);


	//computeCentroidAndCovariance_test(numPoints, range);

	//computeOBB_test(numPoints, range);
	//PassThrough_test(numPoints, range);

	//transformPointCloud_test(numPoints, range);

	/*ransacPlane_test(numPoints);
	ransacLine_test(numPoints);
	ransacCircle2D_test(numPoints);
	ransacCircle3D_test(numPoints);
	ransacSphere_test(numPoints);*/
	//pointCloudToDeepMat_test(numPoints);

	 /*projectPlane_test( numPoints,  range); 
	 projectLine_test( numPoints,  range);
	 projectSphere_test( numPoints,  range);
	 projectCylinder_test(numPoints, range);*/


	 //NormalEstimation_test( numPoints,range);

	 //RadiusOutlierRemoval_test(numPoints, range);
	 //VoxelGrid_test( numPoints, range);
	//StatisticalOutlierRemoval_test( numPoints,range);

	 //ICP_test(numPoints,range);


	  EuclideanClusterExtraction_test( numPoints);
	  //copyPointCloud_test( numPoints,  range);
	return 0;
}




