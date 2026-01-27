#pragma once
#include "../types/device_cloud.h"
#include <Eigen/Core>
#include <Eigen/Dense>
namespace pcl {
    namespace cuda {
        void getMinMax3D(const GpuPointCloud& cloud_in, PointXYZ& min_pt, PointXYZ& max_pt);
        void computeCentroid3D(const GpuPointCloud& cloud_in, PointXYZ& centroid);
        void copyPointCloud(const GpuPointCloud& cloud_in, GpuPointCloud& cloud_out, int* d_mask);
        void copyPointCloud(const GpuPointCloud& cloud_in, GpuPointCloud& cloud_out, int* d_indices, int size, bool negative);
        void copyPointCloud(const GpuPointCloud& cloud_in, GpuPointCloud& cloud_out, std::vector<int>h_indices, bool negative);
        void copyPointCloud(const float* d_x, const float * d_y, const float * d_z, const int numPoints, GpuPointCloud& cloud_out, int* d_mask);
        void transformPointCloud(const GpuPointCloud& cloud_in, GpuPointCloud& cloud_out, Eigen::Matrix4f matrix_);

        void pointCloudToDeepMat(const GpuPointCloud& cloud_in, const int width, const int height, float* mat_out, float resolution, float min_x, float min_y);

        void computeCentroidAndCovariance(const GpuPointCloud& cloud_in, Eigen::Matrix3f& covariance_, PointXYZ& centoird_);

        void computeOBB(
            const GpuPointCloud& cloud_in,
            const Eigen::Matrix3f& covariance,
            const PointXYZ& centroid,
            PointXYZ& min_point_OBB,
            PointXYZ& max_point_OBB,
            PointXYZ& position_OBB,
            Eigen::Matrix3f& rotational_matrix_OBB
        );


    } // namespace cuda
} // namespace pcl