#include "pcl_cuda/types/device_cloud.h"
#include "internal/error.cuh"

namespace pcl {
    namespace cuda {

        // 构造
        GpuPointCloud::GpuPointCloud() : size_(0), capacity_(0), x_(nullptr), y_(nullptr), z_(nullptr) {}

        // 析构：核心防泄漏逻辑
        GpuPointCloud::~GpuPointCloud() {
            clear();
        }

        // 移动构造：接管别人的资源
        GpuPointCloud::GpuPointCloud(GpuPointCloud&& other) noexcept {
            swap(other);
        }

        // 移动赋值：先释放自己的，再接管别人的
        GpuPointCloud& GpuPointCloud::operator=(GpuPointCloud&& other) noexcept {
            if (this != &other) {
                clear(); // 这一步至关重要，防止原来的显存泄漏
                swap(other);
            }
            return *this;
        }


        void GpuPointCloud::alloc(size_t size) {
            if (size <= capacity_) {
                size_ = size;
                return;
            }
            // 重新分配
            clear(); // 先清空旧的
            CHECK_CUDA(cudaMalloc(&x_, size * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&y_, size * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&z_, size * sizeof(float)));
            size_ = size;
            capacity_ = size;
        }
        void GpuPointCloud::clear() {
            if (x_) CHECK_CUDA(cudaFree(x_));
            if (y_) CHECK_CUDA(cudaFree(y_));
            if (z_) CHECK_CUDA(cudaFree(z_));
            x_ = y_ = z_ = nullptr;
            size_ = 0;
            capacity_ = 0;
        }
        void GpuPointCloud::upload(const std::vector<float>& h_x, const std::vector<float>& h_y, const std::vector<float>& h_z) {
            
            size_t size = h_x.size();
            if (size == 0) {

                std::cerr << "[Warning] upload received empty cloud!" << std::endl;
                return;
            }
            
            alloc(size);
            CHECK_CUDA(cudaMemcpy(x_, h_x.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(y_, h_y.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(z_, h_z.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
        }

        void GpuPointCloud::download(std::vector<float>& h_x, std::vector<float>& h_y, std::vector<float>& h_z) const {
            h_x.resize(size_); h_y.resize(size_); h_z.resize(size_);
            CHECK_CUDA(cudaMemcpy(h_x.data(), x_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_y.data(), y_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_z.data(), z_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
        }
        void GpuPointCloud::fromDevicePointers(const float* d_x, const float* d_y, const float* d_z, size_t count) {
            // 1. 分配空间 (alloc 内部应该判断 capacity 是否足够，不够才重新 malloc)
            this->alloc(count);

            this->size_ = count;

            if (count == 0) return;

            CHECK_CUDA(cudaMemcpy(this->x_, d_x, count * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(this->y_, d_y, count * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(this->z_, d_z, count * sizeof(float), cudaMemcpyDeviceToDevice));

        }
        void GpuPointCloud::swap(GpuPointCloud& other) {
            std::swap(x_, other.x_);
            std::swap(y_, other.y_);
            std::swap(z_, other.z_);
            std::swap(size_, other.size_);
            std::swap(capacity_, other.capacity_);
        }

        // 构造
        GpuPointCloudNormal::GpuPointCloudNormal() : size_(0), capacity_(0), nx_(nullptr), ny_(nullptr), nz_(nullptr), curvature_(nullptr) {}

        GpuPointCloudNormal::~GpuPointCloudNormal() {
            clear();
        }


        GpuPointCloudNormal::GpuPointCloudNormal(GpuPointCloudNormal&& other) noexcept {
            swap(other);
        }


        GpuPointCloudNormal& GpuPointCloudNormal::operator=(GpuPointCloudNormal&& other) noexcept {
            if (this != &other) {
                clear(); 
                swap(other);
            }
            return *this;
        }


        void GpuPointCloudNormal::alloc(size_t size) {
            if (size <= capacity_) {
                size_ = size;
                return;
            }
            // 重新分配
            clear(); // 先清空旧的
            CHECK_CUDA(cudaMalloc(&nx_, size * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&ny_, size * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&nz_, size * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&curvature_, size * sizeof(float)));
            size_ = size;
            capacity_ = size;
        }
        void GpuPointCloudNormal::clear() {
            if (nx_) CHECK_CUDA(cudaFree(nx_));
            if (ny_) CHECK_CUDA(cudaFree(ny_));
            if (nz_) CHECK_CUDA(cudaFree(nz_));
            if (curvature_) CHECK_CUDA(cudaFree(curvature_));
            nx_ = ny_ = nz_ = curvature_ = nullptr;
            size_ = 0;
            capacity_ = 0;
        }
        void GpuPointCloudNormal::upload(const std::vector<float>& h_nx, const std::vector<float>& h_ny, const std::vector<float>& h_nz, const std::vector<float>& h_curvature) {

            size_t size = h_nx.size();
            if (size == 0) {

                std::cerr << "[Warning] upload received empty cloud!" << std::endl;
                return;
            }

            alloc(size);
            CHECK_CUDA(cudaMemcpy(nx_, h_nx.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(ny_, h_ny.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(nz_, h_nz.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(curvature_, h_curvature.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
        }

        void GpuPointCloudNormal::download(std::vector<float>& h_nx, std::vector<float>& h_ny, std::vector<float>& h_nz, std::vector<float>& h_curvature) const {
            h_nx.resize(size_); h_ny.resize(size_); h_nz.resize(size_); h_curvature.resize(size_);
            CHECK_CUDA(cudaMemcpy(h_nx.data(), nx_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_ny.data(), ny_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_nz.data(), nz_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_curvature.data(), curvature_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
        }
        void GpuPointCloudNormal::fromDevicePointers(const float* d_nx, const float* d_ny, const float* d_nz, const float* d_curvature, size_t count) {
            // 1. 分配空间 (alloc 内部应该判断 capacity 是否足够，不够才重新 malloc)
            this->alloc(count);

            // 2. 更新大小
            this->size_ = count;

            if (count == 0) return;


            CHECK_CUDA(cudaMemcpy(this->nx_, d_nx, count * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(this->ny_, d_ny, count * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(this->nz_, d_nz, count * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(this->curvature_, d_curvature, count * sizeof(float), cudaMemcpyDeviceToDevice));

        }
        void GpuPointCloudNormal::swap(GpuPointCloudNormal& other) {
            std::swap(nx_, other.nx_);
            std::swap(ny_, other.ny_);
            std::swap(nz_, other.nz_);
            std::swap(curvature_, other.curvature_);
            std::swap(size_, other.size_);
            std::swap(capacity_, other.capacity_);
        }








    } // namespace cuda
} // namespace pcl