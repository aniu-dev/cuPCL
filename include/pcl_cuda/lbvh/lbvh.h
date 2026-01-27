#pragma once
#include <memory>
#include <cstdint>
#include "Bound.h" 
#include "../common/device_cloud.h"
namespace pcl {
    namespace cuda {

        /// <summary>
        /// 基于线性包围体层次结构 (LBVH) 的 GPU 加速空间索引。
        /// 算法基于 Tero Karras (2012) 的论文：
        /// "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
        /// </summary>
        class LBVH {
        public:
            // 使用 AABB 作为包围盒类型
            using AabbType = AABB;

            /// <summary>
            /// BVH 内部节点结构 (64字节对齐以优化缓存行读取)
            /// </summary>
            struct alignas(64) Node {
                uint32_t parentIdx; // 父节点索引
                uint32_t leftIdx;   // 左子节点索引 (最高位为1表示叶子)
                uint32_t rightIdx;  // 右子节点索引 (最高位为1表示叶子)
                uint32_t fence;     // 占位符/构建栅栏 (用于对齐)

                AabbType bounds[2]; // 左右子节点的包围盒 [0]:Left, [1]:Right
            };

        private:
            // PImpl 模式隐藏 Thrust 实现细节
            struct ThrustImpl;
            std::unique_ptr<ThrustImpl> impl;

            size_t numObjects = 0; // 对象总数
            AabbType rootAABB;     // 整个场景的根包围盒

        public:
            LBVH();
            ~LBVH();

            /// <summary>
            /// 在 GPU 上构建 BVH 树
            /// </summary>
            /// <param name="d_objects">设备端 AABB 数组指针</param>
            /// <param name="count">对象数量</param>
            //void Compute(const AabbType* d_objects, size_t count);
            
           void compute(const GpuPointCloud* cloud, size_t count);

            // --- Getters (用于在自定义 Kernel 中访问树) ---
            /// <summary>获取内部节点数组 (大小为 N-1)</summary>
            const Node* getNodes() const;

            /// <summary>获取原始对象数组 (未排序)</summary>
            const AabbType* getObjects() const;

            /// <summary>获取排序后的索引映射 (Sorted Index -> Original Index)</summary>
            const uint32_t* getSortedIndices() const;
            /// <summary>获取对象数量</summary>
            size_t getNumObjects() const { return numObjects; }
            /// <summary>获取根节点包围盒</summary>
            AabbType getRootAABB() const { return rootAABB; }
        };
    }
}