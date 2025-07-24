include ../mkenv.mk

dir-y :=

dir-$(CONFIG_RTT_FACE_RECOGNITION) += face_recognition

# Add directories to the build system
dirs := $(sort $(dir-y))

.PHONY: all clean

# Default build arguments
BUILD_OUT ?= k230_bin

# Default target
all:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_FACE_RECOGNITION)/*

ifeq ($(CONFIG_RTT_ENABLE_BUILD_FACE_RECOGNITION),y)
	@rm -rf $(BUILD_OUT)
	@mkdir -p $(RTT_EXAMPLES_ELF_INSTALL_PATH_FACE_RECOGNITION)

	./build_app.sh || exit $?
	if [ -d $(BUILD_OUT) ]; then \
		echo "Copying $(BUILD_OUT) to $(RTT_EXAMPLES_ELF_INSTALL_PATH_FACE_RECOGNITION)"; \
		cp -r $(BUILD_OUT)/* $(RTT_EXAMPLES_ELF_INSTALL_PATH_FACE_RECOGNITION); \
	else \
		echo "No build output found in $(BUILD_OUT), skipping copy."; \
	fi
endif
	@echo "Make face recognition samples done."

clean:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_FACE_RECOGNITION)/*
	@rm -rf $(BUILD_OUT)
	@echo "Clean face recognition samples done."
