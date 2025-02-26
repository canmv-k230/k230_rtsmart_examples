include ../mkenv.mk

dir-y :=

dir-$(CONFIG_RTT_OPENCV_CALCULATE_HIST) += opencv_calculate_hist
dir-$(CONFIG_RTT_OPENCV_DETECT_FEATURES2D) += opencv_detect_features2d
dir-$(CONFIG_RTT_OPENCV_FIND_CONTOURS) += opencv_find_contours
dir-$(CONFIG_RTT_OPENCV_GRAYSCALE_BINARIZE) += opencv_grayscale_binarize
dir-$(CONFIG_RTT_OPENCV_OBJ_DETECT) += opencv_obj_detect
# Add directories to the build system
dirs := $(sort $(dir-y))

.PHONY: all clean

# Default build arguments
BUILD_OUT ?= k230_bin

# Default target
all:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_OPENCV_EXAMPLES)/*

ifeq ($(CONFIG_RTT_ENABLE_BUILD_OPENCV_EXAMPLES),y)
	@rm -rf $(BUILD_OUT)
	@mkdir -p $(RTT_EXAMPLES_ELF_INSTALL_PATH_OPENCV_EXAMPLES)

	./build_app.sh
	if [ -d $(BUILD_OUT) ]; then \
		echo "Copying $(BUILD_OUT) to $(RTT_EXAMPLES_ELF_INSTALL_PATH_OPENCV_EXAMPLES)"; \
		cp -r $(BUILD_OUT)/* $(RTT_EXAMPLES_ELF_INSTALL_PATH_OPENCV_EXAMPLES); \
	else \
		echo "No build output found in $(BUILD_OUT), skipping copy."; \
	fi
endif
	@echo "Make opencv samples done."

clean:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_OPENCV_EXAMPLES)/*
	@rm -rf $(BUILD_OUT)
	@echo "Clean opencv samples done."
