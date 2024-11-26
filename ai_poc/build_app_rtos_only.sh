#!/bin/bash
# set -x

# Get the full path of this script
SCRIPT=$(realpath -s "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

# Define SDK_SRC_ROOT_DIR as the base root directory
export SDK_SRC_ROOT_DIR=$(realpath "${SCRIPTPATH}/../../../../")

# Define other paths relative to SDK_SRC_ROOT_DIR
export SDK_RTSMART_SRC_DIR="${SDK_SRC_ROOT_DIR}/src/rtsmart/"
export MPP_SRC_DIR="${SDK_RTSMART_SRC_DIR}/mpp/"
export NNCASE_SRC_DIR="${SDK_RTSMART_SRC_DIR}/libs/nncase/"
export OPENCV_SRC_DIR="${SDK_RTSMART_SRC_DIR}/libs/opencv/"

# 定义.h文件的路径
header_file="${MPP_SRC_DIR}/include/comm/k_autoconf_comm.h"

if grep -q "k230_canmv_01studio" "$header_file"; then
    if [ $# -eq 0 ]; then
        ./build_app_sub_rtos_only.sh all hdmi
        ./build_app_sub_rtos_only.sh all lcd
    else
        if [ $# -eq 1 ]; then
            ./build_app_sub_rtos_only.sh "$1"
        elif [ $# -eq 2 ]; then
            ./build_app_sub_rtos_only.sh "$1" "$2"
        else
            echo "参数太多，最多支持两个参数"
        fi
    fi
else
    if [ $# -eq 0 ]; then
        ./build_app_sub_rtos_only.sh
    elif [ $# -eq 1 ]; then
        ./build_app_sub_rtos_only.sh "$1"
    else
         echo "参数太多，最多支持一个参数"
    fi
fi
