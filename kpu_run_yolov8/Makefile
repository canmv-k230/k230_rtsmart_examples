include ../mkenv.mk

dir-y :=

dir-$(CONFIG_RTT_KPU_RUN_YOLOV8_CAMERA) += yolov8_run_camera
dir-$(CONFIG_RTT_KPU_RUN_YOLOV8_IMAGE) += yolov8_run_image

# Add directories to the build system
dirs := $(sort $(dir-y))

.PHONY: all clean

# Default build arguments
BUILD_OUT ?= k230_bin

# Default target
all:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_KPU_RUN_YOLOV8)/*

ifeq ($(CONFIG_RTT_ENABLE_BUILD_KPU_RUN_EXAMPLES),y)
	./build_app.sh
	if [ -d $(BUILD_OUT) ]; then \
		echo "Copying $(BUILD_OUT) to $(RTT_EXAMPLES_ELF_INSTALL_PATH_KPU_RUN_YOLOV8)"; \
		cp -r $(BUILD_OUT)/* $(RTT_EXAMPLES_ELF_INSTALL_PATH_KPU_RUN_YOLOV8); \
	else \
		echo "No build output found in $(BUILD_OUT), skipping copy."; \
	fi
endif
	@echo "Make kpu run samples done."

clean:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_KPU_RUN_YOLOV8)/*
	@rm -rf out k230_bin
	@echo "Clean kpu run samples done."
