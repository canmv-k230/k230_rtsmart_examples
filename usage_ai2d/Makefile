include ../mkenv.mk

dir-y :=

dir-$(CONFIG_RTT_AI2D_AFFINE) += test_affine
dir-$(CONFIG_RTT_AI2D_CROP) += test_crop
dir-$(CONFIG_RTT_AI2D_PAD) += test_pad
dir-$(CONFIG_RTT_AI2D_RESIZE) += test_resize
dir-$(CONFIG_RTT_AI2D_SHIFT) += test_shift

# Add directories to the build system
dirs := $(sort $(dir-y))

.PHONY: all clean

# Default build arguments
BUILD_OUT ?= k230_bin

# Default target
all:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_USAGE_AI2D)/*

ifeq ($(CONFIG_RTT_ENABLE_BUILD_AI2D_EXAMPLES),y)
	./build_app.sh
	if [ -d $(BUILD_OUT) ]; then \
		echo "Copying $(BUILD_OUT) to $(RTT_EXAMPLES_ELF_INSTALL_PATH_USAGE_AI2D)"; \
		cp -r $(BUILD_OUT)/* $(RTT_EXAMPLES_ELF_INSTALL_PATH_USAGE_AI2D); \
	else \
		echo "No build output found in $(BUILD_OUT), skipping copy."; \
	fi
endif
	@echo "Make kpu run samples done."

clean:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_USAGE_AI2D)/*
	@rm -rf out k230_bin
	@echo "Clean kpu run samples done."
