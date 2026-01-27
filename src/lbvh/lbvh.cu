#include "pcl_cuda/common/primitives.h"
#include "lbvh.cuh"
#include "internal/error.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <device_launch_parameters.h>
namespace pcl {
    namespace cuda {
        //  Morton Code 位运算核心
        // 将一个 10位的整数 扩展成 30位 (例如 001 -> 001001)
        __host__ __device__ inline uint32_t mortonExpandBits(uint32_t v) {
            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;
            return v;
        }

        // 计算 3D Morton Code
        // 输入：int3 坐标 (范围通常是 0~1023)
        // 输出：30位 整数 (用于排序)
        __host__ __device__ inline uint32_t getMorton(int3 cell) {
            const uint32_t xx = mortonExpandBits((uint32_t)cell.x);
            const uint32_t yy = mortonExpandBits((uint32_t)cell.y);
            const uint32_t zz = mortonExpandBits((uint32_t)cell.z);
            return xx * 4 + yy * 2 + zz;
        }

        struct LBVH::ThrustImpl {
            // 原始数据引用 (不拥有内存)
            thrust::device_vector<AabbType> vector_objects;
            AabbType* d_objects_ptr = nullptr;
            // --- 构建过程中的缓冲区 ---
            // 1. Morton Codes: 用于 Z-order 曲线排序
            thrust::device_vector<uint32_t> d_morton_codes;

            // 2. Object IDs: 排序后的索引映射 
            thrust::device_vector<uint32_t> d_sorted_indices;

            // 3. Nodes: 存储内部节点 (大小 N-1)
            thrust::device_vector<Node> d_nodes;

            // 4. Leaf Parents: 记录叶子节点的父节点索引，用于自底向上计算 AABB
            thrust::device_vector<uint32_t> d_leaf_parents;

            // 5. Construction Flags: 原子计数器，用于 Refit 阶段的同步
            thrust::device_vector<int> d_flags;
        };
            __global__ void createAABBsKernel(
                const float* __restrict__ x_in,
                const float* __restrict__ y_in,
                const float* __restrict__ z_in,
                AABB* bounds,
                int numPoints)
            {
                int tx = blockIdx.x * blockDim.x + threadIdx.x;
                if (tx >= numPoints) return;
                float x = x_in[tx];
                float y = y_in[tx];
                float z = z_in[tx];
                float3 point = make_float3(x, y, z);

                bounds[tx] = AABB(point);

            }
            // 计算两个 64位 Key 的前导零个数 (Count Leading Zeros)
            // 用于判断两个 Morton Code 有多少位是相同的 (Common Prefix)
            __device__ inline int countCommonUpperBits(const uint64_t lhs, const uint64_t rhs) {
                return ::__clzll(lhs ^ rhs);
            }
            // 将 30位的 Morton Code 和 32位的 Index 合并为一个 64位的 Key
            // 这样可以处理 Code 相同的情况 (Index 作为 Tie-breaker)
            __device__ inline uint64_t mergeCodeAndIndex(const uint32_t code, const int idx) {
                return ((uint64_t)code << 32ul) | (uint64_t)idx;
            }

            // 核心逻辑 1：确定当前节点覆盖的范围 [i, j]
            // 给定一个排序后的索引 idx，向左或向右扩展，找到具有相同公共前缀的最远范围
            __device__ inline int2 determineRange(uint32_t const* mortonCodes, const uint32_t numObjs, uint32_t idx) {
                if (idx == 0) return make_int2(0, numObjs - 1);

                // 1. 确定搜索方向 (d) 和公共前缀长度 (min_delta)
                const uint64_t self_code = mergeCodeAndIndex(mortonCodes[idx], idx);
                const int l_delta = countCommonUpperBits(self_code, mergeCodeAndIndex(mortonCodes[idx - 1], idx - 1));
                const int r_delta = countCommonUpperBits(self_code, mergeCodeAndIndex(mortonCodes[idx + 1], idx + 1));
                const int d = (r_delta > l_delta) ? 1 : -1;
                const int min_delta = min(l_delta, r_delta);

                // 2. 指数步长搜索，找到范围的上界
                int l_max = 2;
                int i;
                // 向外扩张，直到公共前缀小于 min_delta
                while ((i = idx + d * l_max) >= 0 && i < numObjs) {
                    if (countCommonUpperBits(self_code, mergeCodeAndIndex(mortonCodes[i], i)) <= min_delta) break;
                    l_max <<= 1;
                }
                // 3. 二分查找，精确定位范围边界
                int t = l_max >> 1;
                int l = 0;
                while (t > 0) {
                    i = idx + (l + t) * d;
                    if (0 <= i && i < numObjs) 
                    {
                        if (countCommonUpperBits(self_code, mergeCodeAndIndex(mortonCodes[i], i)) > min_delta)
                            l += t;
                    }
                    t >>= 1;
                }

                unsigned int j = idx + l * d;
                if (d < 0) {
                    // 确保返回 (min, max)
                    unsigned int temp = idx; idx = j; j = temp;
                }
                return make_int2(idx, j);
            }

            // 核心逻辑 ：寻找分割点 (Split Position)
            // 在范围 [first, last] 内找到一个分割点 gamma，使得 common prefix 发生变化
            __device__ inline uint32_t findSplit(uint32_t const* mortonCodes, const uint32_t first, const uint32_t last) {
                const uint64_t first_code = mergeCodeAndIndex(mortonCodes[first], first);
                const uint64_t last_code = mergeCodeAndIndex(mortonCodes[last], last);

                // 整个范围共有的前缀长度
                const int common_prefix = countCommonUpperBits(first_code, last_code);
                // 二分查找分割点
                int split = first;
                int step = last - first;
                do {
                    step = (step + 1) >> 1;
                    const int new_split = split + step; // 尝试的中点

                    if (new_split < last) {
                        uint64_t split_code = mergeCodeAndIndex(mortonCodes[new_split], new_split);
                        int split_prefix = countCommonUpperBits(first_code, split_code);

                        // 如果左半部分的前缀依然很长，说明分割点还在右边
                        if (split_prefix > common_prefix)
                            split = new_split;
                    }
                } while (step > 1);

                return split;
            }
        

        // Kernel 1: 计算每个对象的 Morton Code

        __global__ void computeMortonCodesKernel(const LBVH::AabbType* objects, uint32_t* codes, 
            uint32_t* indices, LBVH::AabbType rootAABB, int numObjects) {
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid >= numObjects) return;

            // 1. 归一化坐标到 [0, 1]
            float3 norm_pos = rootAABB.normCoord(objects[tid].center());

            // 2. 映射到 [0, 1023] 的整数网格
            float3 grid_pos = norm_pos * 1024.0f;
            int3 c;
            c.x = min(max((int)grid_pos.x, 0), 1023);
            c.y = min(max((int)grid_pos.y, 0), 1023);
            c.z = min(max((int)grid_pos.z, 0), 1023);

            // 3. 计算 30-bit Morton Code
            codes[tid] = getMorton(c); // 假设 getMorton 在 Common.h 中定义
            indices[tid] = tid;        // 初始化索引
        }


        // Kernel : 构建树的拓扑结构 (内部节点)
        // 每个线程负责一个内部节点 (总共 N-1 个)

        __global__ void buildInternalHierarchyKernel(LBVH::Node* internalNodes, uint32_t* leafParents,
            const uint32_t* mortonCodes, int numObjects) {
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid >= numObjects - 1) return; // 内部节点只有 N-1 个

            // 1. 确定当前节点覆盖的索引范围 [first, last]
            int2 range = determineRange(mortonCodes, numObjects, tid);

            // 2. 找到分割点 gamma
            const int gamma = findSplit(mortonCodes, range.x, range.y);

            // 3. 设置左子节点
            if (range.x == gamma) {
                // 左孩子是叶子节点
                // 高位 0x80000000 标记为叶子
                internalNodes[tid].leftIdx = gamma | 0x80000000;
                leafParents[gamma] = (uint32_t)tid; // 记录该叶子的父节点
            }
            else {
                // 左孩子是内部节点
                internalNodes[tid].leftIdx = gamma;
                internalNodes[gamma].parentIdx = (uint32_t)tid;
            }

            // 4. 设置右子节点
            if (range.y == gamma + 1) {
                // 右孩子是叶子节点
                internalNodes[tid].rightIdx = (gamma + 1) | 0x80000000;
                leafParents[gamma + 1] = (uint32_t)tid | 0x80000000; // 标记该叶子是右孩子
            }
            else {
                // 右孩子是内部节点
                internalNodes[tid].rightIdx = gamma + 1;
                // 标记该内部节点是其父节点的右孩子 (高位 flag 用于 AABB 合并时的索引)
                internalNodes[gamma + 1].parentIdx = (uint32_t)tid | 0x80000000;
            }
        }

        
        // Kernel 3: 自底向上计算并合并 AABB (Refit)

        __global__ void calculateInternalAABBsKernel(LBVH::Node* internalNodes, uint32_t* leafParents,
            const LBVH::AabbType* rawObjects, uint32_t* sortedIndices, int* flags, int numObjects) {
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid >= numObjects) return;

            // 每个线程从一个叶子节点开始，向上爬升
            uint32_t sorted_idx = tid;
            uint32_t original_idx = sortedIndices[sorted_idx]; // 获取原始对象 ID

            // 加载叶子节点的包围盒
            LBVH::AabbType current_bound = rawObjects[original_idx];

            // 获取父节点信息
            uint32_t parent_info = leafParents[sorted_idx];
            uint32_t parent_idx = parent_info & 0x7FFFFFFF;
            bool is_right_child = (parent_info & 0x80000000) != 0;

            // 循环向上合并，直到根节点
            while (true) {
                // 1. 将自己的包围盒写入父节点的对应槽位 (Bounds[0] 或 Bounds[1])
                internalNodes[parent_idx].bounds[is_right_child] = current_bound;
                __threadfence();
                // 2. 原子操作：标记自己已经到达父节点
                // atomicOr 返回旧值。如果旧值是 0，说明我是第一个到达的，我要挂起(return)。
                // 如果旧值是 1，说明兄弟节点已经把它的包围盒写好了，我是第二个到达的，我负责合并并继续向上。
                int old_flag = atomicOr(flags + parent_idx, 1);

                if (old_flag == 0) {
                    // 我是第一个到的，由于缺乏兄弟节点的数据，无法计算父节点 AABB，任务结束。
                    return;
                }

                // 3. 我是第二个到的，兄弟已经准备好了。开始合并。
                // 此时内存屏障保证了兄弟的数据已写入显存。
                LBVH::AabbType left_bound = internalNodes[parent_idx].bounds[0];
                LBVH::AabbType right_bound = internalNodes[parent_idx].bounds[1];

                current_bound = left_bound;
                current_bound.merge(right_bound); // 合并左右，得到父节点的新 AABB

                // 4. 检查是否到达根节点
                if (parent_idx == 0) return;

                // 5. 准备下一轮循环：移动到爷爷节点
                uint32_t curr_node_idx = parent_idx;
                uint32_t parent_raw = internalNodes[curr_node_idx].parentIdx;

                parent_idx = parent_raw & 0x7FFFFFFF;
                is_right_child = (parent_raw & 0x80000000) != 0;
            }
        }

        LBVH::LBVH() : impl(std::make_unique<ThrustImpl>()) {}
        LBVH::~LBVH() = default;

        // Getters
        const LBVH::Node* LBVH::getNodes() const { return thrust::raw_pointer_cast(impl->d_nodes.data()); }
        const LBVH::AabbType* LBVH::getObjects() const { return impl->d_objects_ptr; }
        const uint32_t* LBVH::getSortedIndices() const { return thrust::raw_pointer_cast(impl->d_sorted_indices.data()); }

        void LBVH::compute(const GpuPointCloud* cloud, size_t size) {
           
            if (size == 0) {

                std::cerr << "[Warning] LBVH received empty cloud!" << std::endl;
                return;
            }
            numObjects = size;       
            const int blockSize = 256;
            const int gridSize = (numObjects + blockSize - 1) / blockSize;

            PointXYZ minVal, maxVal;
            reduceMinMax3D(*cloud, numObjects, minVal, maxVal);
            float3 min_val = make_float3(minVal.x, minVal.y, minVal.z);
            float3 max_val = make_float3(maxVal.x, maxVal.y, maxVal.z);
            rootAABB = AabbType(min_val, max_val);
            
            impl->vector_objects.resize(numObjects);
            //  调整显存缓冲区大小
            impl->d_morton_codes.resize(numObjects);
            impl->d_sorted_indices.resize(numObjects);
            impl->d_leaf_parents.resize(numObjects);
            impl->d_nodes.resize(numObjects - 1); // 内部节点数总是 N-1
            impl->d_flags.resize(numObjects - 1);

            // 原子计数器必须清零
            thrust::fill(impl->d_flags.begin(), impl->d_flags.end(), 0);
            // 更新缓存的裸指针
            impl->d_objects_ptr = thrust::raw_pointer_cast(impl->vector_objects.data());
            
            createAABBsKernel << <gridSize, blockSize >> > (
                cloud->x(), cloud->y(), cloud->z(), impl->d_objects_ptr, numObjects);
            CHECK_CUDA_KERNEL_ERROR();
            CHECK_CUDA(cudaDeviceSynchronize());
            computeMortonCodesKernel << <gridSize, blockSize >> > (
                impl->d_objects_ptr,
                thrust::raw_pointer_cast(impl->d_morton_codes.data()),
                thrust::raw_pointer_cast(impl->d_sorted_indices.data()),
                rootAABB,
                (int)numObjects
                );
            CHECK_CUDA_KERNEL_ERROR();
            CHECK_CUDA(cudaDeviceSynchronize());
            // 排序 (Sort by Key)
            // Key: Morton Code, Value: Original Index
            // 排序后，空间位置相近的物体在数组中也相邻
            thrust::stable_sort_by_key(
                impl->d_morton_codes.begin(),
                impl->d_morton_codes.end(),
                impl->d_sorted_indices.begin()
            );
 
            // 构建树的拓扑结构 (生成内部节点)
            int internalBlocks = (int)((numObjects - 1 + blockSize - 1) / blockSize);
            buildInternalHierarchyKernel << <internalBlocks, blockSize >> > (
                thrust::raw_pointer_cast(impl->d_nodes.data()),
                thrust::raw_pointer_cast(impl->d_leaf_parents.data()),
                thrust::raw_pointer_cast(impl->d_morton_codes.data()),
                (int)numObjects
                );
            CHECK_CUDA_KERNEL_ERROR();
            CHECK_CUDA(cudaDeviceSynchronize());
            //计算每个节点的 AABB (Refit 过程)
            calculateInternalAABBsKernel << <gridSize, blockSize >> > (
                thrust::raw_pointer_cast(impl->d_nodes.data()),
                thrust::raw_pointer_cast(impl->d_leaf_parents.data()),
                impl->d_objects_ptr,
                thrust::raw_pointer_cast(impl->d_sorted_indices.data()),
                thrust::raw_pointer_cast(impl->d_flags.data()),
                (int)numObjects
                );

            CHECK_CUDA_KERNEL_ERROR();
            CHECK_CUDA(cudaDeviceSynchronize());

        }

    } // namespace cuda
} // namespace pcl