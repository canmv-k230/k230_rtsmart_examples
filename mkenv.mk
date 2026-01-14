ifneq ($(MKENV_INCLUDED),1)
export SDK_SRC_ROOT_DIR := $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST))))/../../../)
endif

include $(SDK_SRC_ROOT_DIR)/tools/mkenv.mk

include $(SDK_SRC_ROOT_DIR)/.config

export MPP_SRC_DIR := $(SDK_RTSMART_SRC_DIR)/mpp/
export NNCASE_SRC_DIR := $(SDK_RTSMART_SRC_DIR)/libs/nncase
export OPENCV_SRC_DIR := $(SDK_RTSMART_SRC_DIR)/libs/opencv
export OPENBLAS_SRC_DIR := $(SDK_RTSMART_SRC_DIR)/libs/openblas

export RTT_EXAMPLES_ELF_INSTALL_PATH_AI_POC := $(SDK_RTSMART_SRC_DIR)/examples/elf/ai_poc/
export RTT_EXAMPLES_ELF_INSTALL_PATH_KPU_RUN_YOLOV8 := $(SDK_RTSMART_SRC_DIR)/examples/elf/kpu_run_yolov8/
export RTT_EXAMPLES_ELF_INSTALL_PATH_USAGE_AI2D := $(SDK_RTSMART_SRC_DIR)/examples/elf/usage_ai2d/
export RTT_EXAMPLES_ELF_INSTALL_PATH_INTERGRATED_POC := $(SDK_RTSMART_SRC_DIR)/examples/elf/integrated_poc/
export RTT_EXAMPLES_ELF_INSTALL_PATH_FACE_DETECTION := $(SDK_RTSMART_SRC_DIR)/examples/elf/face_detection/
export RTT_EXAMPLES_ELF_INSTALL_PATH_FACE_RECOGNITION := $(SDK_RTSMART_SRC_DIR)/examples/elf/face_recognition/
export RTT_EXAMPLES_ELF_INSTALL_PATH_YOLO := $(SDK_RTSMART_SRC_DIR)/examples/elf/yolo/
export RTT_EXAMPLES_ELF_INSTALL_PATH_OPENCV_EXAMPLES := $(SDK_RTSMART_SRC_DIR)/examples/elf/opencv_examples/
export RTT_EXAMPLES_ELF_INSTALL_PATH_OPENBLAS_EXAMPLES := $(SDK_RTSMART_SRC_DIR)/examples/elf/openblas_examples/
export RTT_EXAMPLES_ELF_INSTALL_PATH_PERIPHERAL_EXAMPLES := $(SDK_RTSMART_SRC_DIR)/examples/elf/peripheral/
export RTT_EXAMPLES_ELF_INSTALL_PATH_UVC_FACE_DETECTION := $(SDK_RTSMART_SRC_DIR)/examples/elf/uvc_face_detection/

include $(SDK_TOOLS_DIR)/toolchain_rtsmart.mk
export PATH:="$(CROSS_COMPILE_DIR):$(PATH)"

export MKENV_INCLUDED_RTSMART_EXAMPLE=1
