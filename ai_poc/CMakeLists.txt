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

# set vi_vo
include_directories(./vi_vo)

if (DEFINED DISPLAY_MODE)
  message(${DISPLAY_MODE})
  if (${DISPLAY_MODE} STREQUAL "hdmi")
    add_definitions(-DSTUDIO_HDMI)
  endif()
endif()

# 定义要排除的目录列表
set(exclude_dirs "cmake" "k230_bin" "shell" "out" "llamac" ".git" "vi_vo")
file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*")
foreach(file ${files})
    # 判断当前文件是否是一个目录
    if(IS_DIRECTORY ${file})
        # 获取目录名称
        get_filename_component(dir_name ${file} NAME)
        # 检查目录是否在排除列表中
        list(FIND exclude_dirs ${dir_name} is_exclude_dir)
        # 如果目录不在排除列表中，则处理该目录
        if(${is_exclude_dir} EQUAL -1)
            # 处理该目录，例如添加到项目中
            if(${dir_name})
                add_subdirectory(${dir_name}) 
            endif()
        endif()
    endif()
endforeach()