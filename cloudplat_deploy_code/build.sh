SCRIPTPATH=$(dirname "$SCRIPT")

# Define SDK_SRC_ROOT_DIR as the base root directory
export SDK_SRC_ROOT_DIR=$(realpath "${SCRIPTPATH}/../../../../")

# Define other paths relative to SDK_SRC_ROOT_DIR
export SDK_RTSMART_SRC_DIR="${SDK_SRC_ROOT_DIR}/src/rtsmart/"
export MPP_SRC_DIR="${SDK_RTSMART_SRC_DIR}/mpp/"
export FREETYPE_SRC_DIR="${SDK_RTSMART_SRC_DIR}/libs/3rd-party/freetype/"
export NNCASE_SRC_DIR="${SDK_RTSMART_SRC_DIR}/libs/nncase/"
export OPENCV_SRC_DIR="${SDK_RTSMART_SRC_DIR}/libs/opencv/"

# set cross build toolchain
export PATH=$PATH:~/.kendryte/k230_toolchains/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin

# 获取传入的目标目录（如果有）
TARGET_DIR=$1

rm -rf out
rm -rf k230_bin
mkdir out

pushd out
if [ -n "${TARGET_DIR}" ]; then
    echo "构建指定目录: ${TARGET_DIR}"
    cmake -DCMAKE_BUILD_TYPE=Release                 \
        -DCMAKE_INSTALL_PREFIX=`pwd`               \
        -DCMAKE_TOOLCHAIN_FILE=cmake/Riscv64.cmake \
        -DTARGET_DIR=${TARGET_DIR} \
        ..
else
    echo "构建所有模块"
    cmake -DCMAKE_BUILD_TYPE=Release                 \
        -DCMAKE_INSTALL_PREFIX=`pwd`               \
        -DCMAKE_TOOLCHAIN_FILE=cmake/Riscv64.cmake \
        ..
fi

make -j  || exit $?
make install || exit $?
popd

k230_bin=`pwd`/k230_bin

mkdir -p ${k230_bin}
# 拷贝结果
cp out/bin/*.elf ${k230_bin} 2>/dev/null
cp utils/* ${k230_bin} 2>/dev/null

rm -rf out