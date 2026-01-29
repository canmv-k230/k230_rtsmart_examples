include ../../mkenv.mk

export RTT_AI_EXAMPLES_UVC_FACE_DETECTION_ELF_INSTALL_PATH := \
    $(SDK_RTSMART_SRC_DIR)/examples/elf/ai/uvc_face_detection

dir-y := src

.PHONY: all clean

# Default build arguments
BUILD_DIR ?= build
BUILD_OUT ?= k230_bin

# Default target
all:
	@rm -rf $(RTT_AI_EXAMPLES_UVC_FACE_DETECTION_ELF_INSTALL_PATH)/*

ifeq ($(CONFIG_RTSMART_AI_EXAMPLES_ENABLE_UVC_FACE_DETECTION),y)
	@rm -rf $(BUILD_OUT)
	@mkdir -p $(RTT_AI_EXAMPLES_UVC_FACE_DETECTION_ELF_INSTALL_PATH)

	./build_app.sh || exit $?
	if [ -d $(BUILD_OUT) ]; then \
		echo "Copying $(BUILD_OUT) to $(RTT_AI_EXAMPLES_UVC_FACE_DETECTION_ELF_INSTALL_PATH)"; \
		cp -r $(BUILD_OUT)/* $(RTT_AI_EXAMPLES_UVC_FACE_DETECTION_ELF_INSTALL_PATH); \
	else \
		echo "No build output found in $(BUILD_OUT), skipping copy."; \
	fi
endif
	@echo "Make uvc+ai samples done."

clean:
	@rm -rf $(RTT_AI_EXAMPLES_UVC_FACE_DETECTION_ELF_INSTALL_PATH)/*
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BUILD_OUT)
	@echo "Clean uvc+ai samples done."