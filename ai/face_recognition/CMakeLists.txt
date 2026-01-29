cmake_minimum_required(VERSION 3.2)
project(face_recognition C CXX)

set(CMAKE_EXE_LINKER_FLAGS "-T ${PROJECT_SOURCE_DIR}/cmake/link.lds --static")

set(k230_opencv $ENV{OPENCV_SRC_DIR})
include_directories(${k230_opencv}/include/opencv4/)
link_directories(${k230_opencv}/lib ${k230_opencv}/lib/opencv4/3rdparty)

set(k230_mpp_root "$ENV{MPP_SRC_DIR}")


# set mmz
link_directories(${k230_mpp_root}/userapps/lib)

# set nncase
set(nncase_sdk_root $ENV{NNCASE_SRC_DIR})
include_directories(${nncase_sdk_root}/riscv64)
include_directories(${nncase_sdk_root}/riscv64/nncase/include)
include_directories(${nncase_sdk_root}/riscv64/nncase/include/nncase/runtime)
link_directories(${nncase_sdk_root}/riscv64/nncase/lib/)

add_subdirectory(src) 