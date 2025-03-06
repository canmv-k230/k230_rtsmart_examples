cmake_minimum_required(VERSION 3.2)
project(opencv_demo C CXX)


include(CMakeForceCompiler)

# target system
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# cross compiler
set(CMAKE_C_COMPILER "$ENV{CROSS_COMPILE_DIR}/riscv64-unknown-linux-musl-gcc")
set(CMAKE_CXX_COMPILER "$ENV{CROSS_COMPILE_DIR}/riscv64-unknown-linux-musl-g++")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_C_FLAGS_RELEASE "-O2 -s")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -DHAVE_CCONFIG_H")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -DHAVE_CCONFIG_H")


# set(k230_sdk ${nncase_sdk_root}/../../../)
set(CMAKE_EXE_LINKER_FLAGS "-T ${PROJECT_SOURCE_DIR}/../usage_ai2d/cmake/link.lds --static")

# # set opencv
# set(k230_opencv $ENV{OPENCV_SRC_DIR})

# include_directories(${k230_opencv}/include/opencv4/)
# link_directories(${k230_opencv}/lib ${k230_opencv}/lib/opencv4/3rdparty)
# # set mmz
# set(k230_mpp_root "$ENV{MPP_SRC_DIR}")
# link_directories(${k230_mpp_root}/userapps/lib)

if(NOT DEFINED ENV{MPP_SRC_DIR})
  set(k230_mpp_root  ../../mpp)
else()
  set(k230_mpp_root "$ENV{MPP_SRC_DIR}")
endif()



include_directories(${k230_mpp_root}/userapps/api/)
include_directories(${k230_mpp_root}/include)
include_directories(${k230_mpp_root}/include/comm)

include_directories(../../rtsmart/userapps/sdk/rt-thread/include/)  #rtsmart user include;
include_directories(../../rtsmart/userapps)
include_directories(../../rtsmart/userapps/sdk/rt-thread/components/drivers/)

add_subdirectory(uart)
add_subdirectory(i2c)
add_subdirectory(spi)
