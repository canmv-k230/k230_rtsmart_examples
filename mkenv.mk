ifneq ($(MKENV_INCLUDED),1)
export SDK_SRC_ROOT_DIR := $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST))))/../../../)
endif

include $(SDK_SRC_ROOT_DIR)/tools/mkenv.mk

include $(SDK_SRC_ROOT_DIR)/.config

export MPP_SRC_DIR := $(SDK_RTSMART_SRC_DIR)/mpp/
export NNCASE_SRC_DIR := $(SDK_RTSMART_SRC_DIR)/libs/nncase
export OPENCV_SRC_DIR := $(SDK_RTSMART_SRC_DIR)/libs/opencv
export OPENBLAS_SRC_DIR := $(SDK_RTSMART_SRC_DIR)/libs/openblas

export RTT_3RD_PARTY_EXAMPLES_ELF_INSTALL_PATH := $(SDK_RTSMART_SRC_DIR)/examples/elf/3rd_party/
export RTT_PERIPHERAL_EXAMPLES_ELF_INSTALL_PATH := $(SDK_RTSMART_SRC_DIR)/examples/elf/peripheral/
export RTT_MPP_EXAMPLES_ELF_INSTALL_PATH := $(SDK_RTSMART_SRC_DIR)/examples/elf/mpp/
export RTT_AI_EXAMPLES_ELF_INSTALL_PATH := $(SDK_RTSMART_SRC_DIR)/examples/elf/ai/
export RTT_EXAMPLES_ELF_INSTALL_PATH_INTERGRATED_POC := $(SDK_RTSMART_SRC_DIR)/examples/elf/integrated_poc/

include $(SDK_TOOLS_DIR)/toolchain_rtsmart.mk
export PATH:="$(CROSS_COMPILE_DIR):$(PATH)"

export MKENV_INCLUDED_RTSMART_EXAMPLE=1
