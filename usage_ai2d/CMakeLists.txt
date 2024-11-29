cmake_minimum_required(VERSION 3.2)
project(nncase_sdk C CXX)

if(NOT DEFINED ENV{NNCASE_SRC_DIR})
  set(nncase_sdk_root "${PROJECT_SOURCE_DIR}/../../big/nncase/")
else()
  set(nncase_sdk_root $ENV{NNCASE_SRC_DIR})
endif()

# set(k230_sdk ${nncase_sdk_root}/../../../)
set(CMAKE_EXE_LINKER_FLAGS "-T ${PROJECT_SOURCE_DIR}/cmake/link.lds --static")

# set opencv
if(NOT DEFINED ENV{OPENCV_SRC_DIR})
  set(k230_opencv ${k230_sdk}/src/big/utils/lib/opencv)
else()
  set(k230_opencv $ENV{OPENCV_SRC_DIR})
endif()

include_directories(${k230_opencv}/include/opencv4/)
link_directories(${k230_opencv}/lib ${k230_opencv}/lib/opencv4/3rdparty)

if(NOT DEFINED ENV{MPP_SRC_DIR})
  set(k230_mpp_root ${nncase_sdk_root}/../../../src/big/mpp/)
else()
  set(k230_mpp_root "$ENV{MPP_SRC_DIR}")
endif()

# set mmz
link_directories(${k230_mpp_root}/userapps/lib)

# set nncase
include_directories(${nncase_sdk_root}/riscv64)
include_directories(${nncase_sdk_root}/riscv64/nncase/include)
include_directories(${nncase_sdk_root}/riscv64/nncase/include/nncase/runtime)
link_directories(${nncase_sdk_root}/riscv64/nncase/lib/)

add_subdirectory(test_crop) 
add_subdirectory(test_pad)
add_subdirectory(test_resize)
add_subdirectory(test_affine)
add_subdirectory(test_shift)