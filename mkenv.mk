ifneq ($(MKENV_INCLUDED),1)
export SDK_SRC_ROOT_DIR := $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST))))/../../../)

include $(SDK_SRC_ROOT_DIR)/tools/mkenv.mk
endif

export MPP_SRC_DIR := $(SDK_RTSMART_SRC_DIR)/mpp/
export NNCASE_SRC_DIR := $(SDK_RTSMART_SRC_DIR)/libs/nncase
export OPENCV_SRC_DIR := $(SDK_RTSMART_SRC_DIR)/libs/opencv

export RTT_EXAMPLES_ELF_INSTALL_PATH_AI_POC := $(SDK_RTSMART_SRC_DIR)/examples/elf/ai_poc/
$(shell if [ ! -e $(RTT_EXAMPLES_ELF_INSTALL_PATH_AI_POC) ];then mkdir -p $(RTT_EXAMPLES_ELF_INSTALL_PATH_AI_POC); fi)

include $(SDK_TOOLS_DIR)/toolchain_rtsmart.mk
export PATH:=$(CROSS_COMPILE_DIR):$(PATH)
