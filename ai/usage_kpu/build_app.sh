#!/bin/bash
# Build and package K230 KPU Run programs (incremental build)
# Function: Use CMake and cross-compile toolchain to kpu run examples, and collect ELF and utility files

set -e  # Exit immediately if a command fails
# set -x  # Enable debug mode

# =======================
# Script directory and SDK paths
# =======================
SCRIPT=$(realpath -s "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

export SDK_SRC_ROOT_DIR=$(realpath "${SCRIPTPATH}/../../../../../")
export SDK_RTSMART_SRC_DIR="${SDK_SRC_ROOT_DIR}/src/rtsmart/"
export MPP_SRC_DIR="${SDK_RTSMART_SRC_DIR}/mpp/"
export NNCASE_SRC_DIR="${SDK_RTSMART_SRC_DIR}/libs/nncase/"
export OPENCV_SRC_DIR="${SDK_RTSMART_SRC_DIR}/libs/opencv/"

# Set cross-compile toolchain path
export PATH=$PATH:~/.kendryte/k230_toolchains/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin

# =======================
# Output directories
# =======================
BUILD_DIR="${SCRIPTPATH}/build"
K230_BIN_DIR="${SCRIPTPATH}/k230_bin"
mkdir -p "${BUILD_DIR}"
mkdir -p "${K230_BIN_DIR}"

# =======================
# Build function
# =======================
build_project() {
    pushd "${BUILD_DIR}"

    # Incremental build: only run CMake if CMakeCache.txt does not exist
    if [ ! -f "CMakeCache.txt" ]; then
        echo "[INFO] Running initial CMake configuration..."
        cmake -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_INSTALL_PREFIX="$(pwd)" \
              -DCMAKE_TOOLCHAIN_FILE=../cmake/Riscv64.cmake \
              ..
    else
        echo "[INFO] Using existing CMake configuration for incremental build..."
    fi

    echo "[INFO] Starting build..."
    make -j --no-print-directory

    echo "[INFO] Installing build artifacts..."
    make install --no-print-directory

    popd
}

# =======================
# Collect ELF and utility files
# =======================
collect_outputs() {
    local elf_files=("yolov8_image.elf" "yolov8_camera.elf")

    echo "[INFO] Collecting ELF files to ${K230_BIN_DIR}..."
    for bin in "${elf_files[@]}"; do
        local elf_file="${BUILD_DIR}/bin/${bin}"
        if [ -f "${elf_file}" ]; then
            cp -u "${elf_file}" "${K230_BIN_DIR}/"
            echo "[INFO] Copied ${bin}"
        else
            echo "[WARN] ELF file not found: ${bin}"
        fi
    done

    # Copy utility files
    if compgen -G "utils/*" > /dev/null; then
        cp -u utils/* "${K230_BIN_DIR}/"
        echo "[INFO] Copied utility files"
    fi
}

# =======================
# Main process
# =======================
build_project
collect_outputs

echo "[INFO] Build finished. Output directory: ${K230_BIN_DIR}"
