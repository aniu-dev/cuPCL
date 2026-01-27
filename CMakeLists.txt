cmake_minimum_required(VERSION 3.18)

# 1. 消除可能的 Policy 警告
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

project(cuPCL LANGUAGES CXX CUDA)

# 2. 必须引入 CUDA Toolkit 才能使用 CUDA::cudart
find_package(CUDAToolkit REQUIRED)

# ==========================================
# 配置选项
# ==========================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# ==========================================
# 查找依赖库
# ==========================================
find_package(PCL 1.10 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# ==========================================
# 核心修改：收集源文件
# ==========================================
# 1. 收集 src 目录下的文件
file(GLOB_RECURSE SRC_FOLDER_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

# 2. 收集 根目录 下的 .cu 和 .cpp 文件 (features.cu, segmentation.cu 等)
file(GLOB ROOT_SOURCE_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

# 3. 合并所有源文件
set(ALL_SRC_FILES ${SRC_FOLDER_FILES} ${ROOT_SOURCE_FILES})

# 4. 【关键步骤】从库的源列表中剔除 main.cpp
# 因为 main.cpp 是生成可执行文件的，不能放进库里，否则会报错 "main defined multiple times"
list(REMOVE_ITEM ALL_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

# 收集头文件
file(GLOB_RECURSE HEADER_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h" # 根目录下的头文件
)

# ==========================================
# IDE 视图优化
# ==========================================
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ALL_SRC_FILES} ${HEADER_FILES})

# ==========================================
# 定义库 (cuPCL)
# ==========================================
# 建议改为 STATIC (静态库)，避免 Windows 下 DLL 导出的麻烦
add_library(${PROJECT_NAME} STATIC
    ${ALL_SRC_FILES}
    ${HEADER_FILES}
)

# 设置包含路径
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> # 允许引用根目录头文件
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${PCL_LIBRARIES}
    CUDA::cudart
)

# ==========================================
# 定义可执行文件 (注意：这里改成了 main.cu)
# ==========================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cu") # <--- 检查 .cu
    # 使用 .cu 文件生成可执行文件，这会强制触发 nvcc 编译器
    add_executable(cuPCL_App main.cu) 
    
    # 链接库
    target_link_libraries(cuPCL_App PRIVATE 
        ${PROJECT_NAME}
        CUDA::cudart 
    )
    
    # 只要源文件是 .cu，CMake 就会自动处理设备链接
    # 但因为链接的是静态库，还是建议保留这两行属性以防万一
    set_target_properties(cuPCL_App PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()