#!/bin/bash
set -x
set +e

# Get the full path of this script
SCRIPT=$(realpath -s "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

# Define SDK_SRC_ROOT_DIR as the base root directory
export SDK_SRC_ROOT_DIR=$(realpath "${SCRIPTPATH}/../../../../../")

# Define other paths relative to SDK_SRC_ROOT_DIR
export SDK_RTSMART_SRC_DIR="${SDK_SRC_ROOT_DIR}/src/rtsmart/"
export MPP_SRC_DIR="${SDK_RTSMART_SRC_DIR}/mpp/"
export NNCASE_SRC_DIR="${SDK_RTSMART_SRC_DIR}/libs/nncase/"
export OPENCV_SRC_DIR="${SDK_RTSMART_SRC_DIR}/libs/opencv/"

# set cross build toolchain
export PATH=$PATH:~/.kendryte/k230_toolchains/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin

rm -rf out
rm -rf k230_bin
mkdir out
pushd out

cmake -DCMAKE_BUILD_TYPE=Release                 \
      -DCMAKE_INSTALL_PREFIX=`pwd`               \
      -DCMAKE_TOOLCHAIN_FILE=cmake/Riscv64.cmake \
      ..

make -j || exit $?
make install || exit $?
popd

k230_bin=`pwd`/k230_bin/

mkdir -p ${k230_bin}
if [ -f out/bin/deepsort_track.elf ]; then
      cp out/bin/deepsort_track.elf ${k230_bin}
      cp -r utils/* ${k230_bin}
fi

rm -rf out

chmod 777 -R k230_bin