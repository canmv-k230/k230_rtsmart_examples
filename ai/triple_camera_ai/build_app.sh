#!/bin/bash
# Build K230 example programs (incremental build version)
# Author: wang yan

set -euo pipefail
# set -x

# =======================
# Script & SDK paths
# =======================
SCRIPT=$(realpath -s "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

export SDK_SRC_ROOT_DIR=$(realpath "${SCRIPTPATH}/../../../../../")
export SDK_RTSMART_SRC_DIR="${SDK_SRC_ROOT_DIR}/src/rtsmart/"
export MPP_SRC_DIR="${SDK_RTSMART_SRC_DIR}/mpp/"
export NNCASE_SRC_DIR="${SDK_RTSMART_SRC_DIR}/libs/nncase/"
export OPENCV_SRC_DIR="${SDK_RTSMART_SRC_DIR}/libs/opencv/"

export CUR_MPP_BUILD_DIR="${SDK_SRC_ROOT_DIR}/output/$(grep ^CONFIG_BOARD_CONFIG_NAME ${SDK_SRC_ROOT_DIR}/.config | cut -d= -f2 | tr -d '"')"

# Toolchain
export PATH=$PATH:~/.kendryte/k230_toolchains/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin

# =======================
# Output directories
# =======================
BUILD_DIR="${SCRIPTPATH}/build"
OUTPUT_DIR="${SCRIPTPATH}/k230_bin"

mkdir -p "${BUILD_DIR}"
mkdir -p "${OUTPUT_DIR}"

# =======================
# Build function
# =======================
build_project() {
    pushd "${BUILD_DIR}" > /dev/null

    if [ ! -f "CMakeCache.txt" ]; then
        echo "[INFO] First-time CMake configuration..."
        cmake -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_INSTALL_PREFIX="$(pwd)" \
              -DCMAKE_TOOLCHAIN_FILE=../cmake/Riscv64.cmake \
              ..
    else
        echo "[INFO] Using existing CMake configuration (incremental build)"
    fi

    echo "[INFO] Building project..."
    make -j --no-print-directory

    echo "[INFO] Installing project..."
    make install --no-print-directory

    popd > /dev/null
}

# =======================
# Collect outputs
# =======================
collect_outputs() {
    local elf_file="${BUILD_DIR}/bin/triple_cam_ai.elf"

    if [ -f "${elf_file}" ]; then
        echo "[INFO] Collecting build artifacts..."
        cp -u "${elf_file}" "${OUTPUT_DIR}/"
        cp -u utils/* "${OUTPUT_DIR}/" 2>/dev/null || true
    else
        echo "[WARN] ELF not found: ${elf_file}"
    fi
}

# =======================
# Main
# =======================
build_project
collect_outputs

echo "[INFO] Incremental build finished. Output dir: ${OUTPUT_DIR}"
