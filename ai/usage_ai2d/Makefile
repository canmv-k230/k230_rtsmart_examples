include ../../mkenv.mk

export RTT_AI_EXAMPLES_UASGE_AI2D_ELF_INSTALL_PATH := \
    $(SDK_RTSMART_SRC_DIR)/examples/elf/ai/usage_ai2d

# 直接列出所有子目录
dir-y := test_affine test_crop test_resize test_pad test_shift 

.PHONY: all clean

# Default build arguments
BUILD_DIR ?= build
BUILD_OUT ?= k230_bin

# Default target
all:
	@rm -rf $(RTT_AI_EXAMPLES_UASGE_AI2D_ELF_INSTALL_PATH)/*

ifeq ($(CONFIG_RTSMART_AI_EXAMPLES_ENABLE_USAGE_AI2D),y)
	@rm -rf $(BUILD_OUT)
	@mkdir -p $(RTT_AI_EXAMPLES_UASGE_AI2D_ELF_INSTALL_PATH)

	./build_app.sh || exit $?
	if [ -d $(BUILD_OUT) ]; then \
		echo "Copying $(BUILD_OUT) to $(RTT_AI_EXAMPLES_UASGE_AI2D_ELF_INSTALL_PATH)"; \
		cp -r $(BUILD_OUT)/* $(RTT_AI_EXAMPLES_UASGE_AI2D_ELF_INSTALL_PATH); \
	else \
		echo "No build output found in $(BUILD_OUT), skipping copy."; \
	fi
endif
	@echo "Make usage ai2d samples done."

clean:
	@rm -rf $(RTT_AI_EXAMPLES_UASGE_AI2D_ELF_INSTALL_PATH)/*
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BUILD_OUT)
	@echo "Clean usage ai2d samples done."