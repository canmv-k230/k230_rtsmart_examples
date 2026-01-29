cmake_minimum_required(VERSION 3.2)
project(opencv_demo C CXX)

set(CMAKE_EXE_LINKER_FLAGS "-T ${PROJECT_SOURCE_DIR}/cmake/link.lds --static")

# set opencv
set(k230_opencv $ENV{OPENCV_SRC_DIR})

include_directories(${k230_opencv}/include/opencv4/)
link_directories(${k230_opencv}/lib ${k230_opencv}/lib/opencv4/3rdparty)
# set mmz
set(k230_mpp_root "$ENV{MPP_SRC_DIR}")
link_directories(${k230_mpp_root}/userapps/lib)

add_subdirectory(opencv_calculate_hist)
add_subdirectory(opencv_detect_features2d)
add_subdirectory(opencv_find_contours)
add_subdirectory(opencv_grayscale_binarize)
add_subdirectory(opencv_obj_detect)
