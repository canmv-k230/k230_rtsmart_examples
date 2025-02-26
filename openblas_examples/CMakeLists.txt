cmake_minimum_required(VERSION 3.2)
project(openblas_demo C CXX)

# set(k230_sdk ${nncase_sdk_root}/../../../)
set(CMAKE_EXE_LINKER_FLAGS "-T ${PROJECT_SOURCE_DIR}/cmake/link.lds --static")

# set opencv
set(k230_openblas $ENV{OPENBLAS_SRC_DIR})

include_directories(${k230_openblas}/include)
link_directories(${k230_openblas}/lib)
# set mmz
set(k230_mpp_root "$ENV{MPP_SRC_DIR}")
link_directories(${k230_mpp_root}/userapps/lib)

add_subdirectory(openblas_cblas_saxpy)
add_subdirectory(openblas_cblas_sger)
add_subdirectory(openblas_cblas_sgemm)
add_subdirectory(openblas_fortran_dgemm)
