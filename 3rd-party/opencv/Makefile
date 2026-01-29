include ../../mkenv.mk

export RTT_3RD_PARTY_OPENCV_ELF_INSTALL_PATH := \
    $(SDK_RTSMART_SRC_DIR)/examples/elf/3rd_party/opencv

dirs := $(sort $(notdir $(wildcard */)))

.PHONY: all clean

# Default build arguments
BUILD_DIR ?= build
BUILD_OUT ?= k230_bin

# Default target
all:
	@rm -rf $(RTT_3RD_PARTY_OPENCV_ELF_INSTALL_PATH)/*

ifeq ($(CONFIG_RTSMART_3RD_PARTY_ENABLE_OPENBLAS_SAMPLES),y)
	@rm -rf $(BUILD_OUT)
	@mkdir -p $(RTT_3RD_PARTY_OPENCV_ELF_INSTALL_PATH)

	./build_app.sh || exit $?
	if [ -d $(BUILD_OUT) ]; then \
		echo "Copying $(BUILD_OUT) to $(RTT_3RD_PARTY_OPENCV_ELF_INSTALL_PATH)"; \
		cp -r $(BUILD_OUT)/* $(RTT_3RD_PARTY_OPENCV_ELF_INSTALL_PATH); \
	else \
		echo "No build output found in $(BUILD_OUT), skipping copy."; \
	fi
endif
	@echo "Make opencv samples done."

clean:
	@rm -rf $(RTT_3RD_PARTY_OPENCV_ELF_INSTALL_PATH)/*
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BUILD_OUT)
	@echo "Clean opencv samples done."