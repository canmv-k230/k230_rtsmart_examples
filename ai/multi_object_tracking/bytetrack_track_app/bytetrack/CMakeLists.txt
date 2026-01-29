# bytetrack/CMakeLists.txt
# file(GLOB BYTETRACK_SRC
#     src/*.cpp
# )

# include_directories(.)

# add_library(bytetrack STATIC ${BYTETRACK_SRC})
cmake_minimum_required(VERSION 3.12)

get_filename_component(MODULE_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
project(${MODULE_NAME} LANGUAGES C CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 源文件
file(GLOB_RECURSE STRONGSORT_SOURCES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE STRONGSORT_HEADERS CONFIGURE_DEPENDS include/*.h include/*.hpp)

add_library(${MODULE_NAME} STATIC ${STRONGSORT_SOURCES})

# 现代 CMake：针对目标设置 include
target_include_directories(${MODULE_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/Eigen32>
        $<INSTALL_INTERFACE:include>
)

# 使用 Generator Expressions 设置输出路径
set_target_properties(${MODULE_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libs/${MODULE_NAME}/lib"
)

set(INCLUDE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/libs/${MODULE_NAME}/include")

# 自动拷贝头文件
add_custom_command(TARGET ${MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${INCLUDE_OUTPUT_DIR}"
)
