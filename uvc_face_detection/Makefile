include ../mkenv.mk

dir-y :=

dir-$(CONFIG_RTT_YOLO) += uvc_face_detection

# Add directories to the build system
dirs := $(sort $(dir-y))

.PHONY: all clean

# Default build arguments
BUILD_OUT ?= k230_bin

# Default target
all:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_UVC_FACE_DETECTION)/*

ifeq ($(CONFIG_RTT_ENABLE_BUILD_UVC_FACE_DETECTION),y)
	@rm -rf $(BUILD_OUT)
	@mkdir -p $(RTT_EXAMPLES_ELF_INSTALL_PATH_UVC_FACE_DETECTION)

	./build_app.sh || exit $?
	if [ -d $(BUILD_OUT) ]; then \
		echo "Copying $(BUILD_OUT) to $(RTT_EXAMPLES_ELF_INSTALL_PATH_UVC_FACE_DETECTION)"; \
		cp -r $(BUILD_OUT)/* $(RTT_EXAMPLES_ELF_INSTALL_PATH_UVC_FACE_DETECTION); \
	else \
		echo "No build output found in $(BUILD_OUT), skipping copy."; \
	fi
endif
	@echo "Make uvc face detection samples done."

clean:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_UVC_FACE_DETECTION)/*
	@rm -rf $(BUILD_OUT)
	@echo "Clean uvc face detection samples done."
