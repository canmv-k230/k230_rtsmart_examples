#!/bin/bash
set -x

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

# set cross build toolchain
export PATH=$PATH:~/.kendryte/k230_toolchains/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin

clear
rm -rf out
rm -rf k230_bin
mkdir out
pushd out

cmake -DCMAKE_BUILD_TYPE=Release                 \
      -DCMAKE_INSTALL_PREFIX=`pwd`               \
      -DCMAKE_TOOLCHAIN_FILE=cmake/Riscv64.cmake \
      ..

make -j && make install
popd

# assemble all test cases
k230_bin=`pwd`/k230_bin

mkdir -p ${k230_bin}
if [ -f out/bin/test_crop.elf ]; then
      cp out/bin/test_crop.elf ${k230_bin}
fi

if [ -f out/bin/test_pad.elf ]; then
      cp out/bin/test_pad.elf ${k230_bin}
fi

if [ -f out/bin/test_resize.elf ]; then
      cp out/bin/test_resize.elf ${k230_bin}
fi

if [ -f out/bin/test_affine.elf ]; then
      cp out/bin/test_affine.elf ${k230_bin}
fi

if [ -f out/bin/test_shift.elf ]; then
      cp out/bin/test_shift.elf ${k230_bin}
fi

cp utils/test.jpg ${k230_bin}
rm -rf out

chmod 777 -R k230_bin