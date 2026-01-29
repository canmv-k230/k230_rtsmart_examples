include ../../../mkenv.mk

export RTT_MOT_BOTSORT_TRACK_ELF_INSTALL_PATH := \
    $(SDK_RTSMART_SRC_DIR)/examples/elf/ai/multi_object_tracking/botsort_track_app

# 直接列出所有子目录
dir-y := src

.PHONY: all clean

# Default build arguments
BUILD_DIR ?= build
BUILD_OUT ?= k230_bin

# Default target
all:
	@rm -rf $(RTT_MOT_BOTSORT_TRACK_ELF_INSTALL_PATH)/*

ifeq ($(CONFIG_RTSMART_BOTSORT_TRACK_ALGORITHM),y)
	@rm -rf $(BUILD_OUT)
	@mkdir -p $(RTT_MOT_BOTSORT_TRACK_ELF_INSTALL_PATH)

	./build_app.sh || exit $?
	if [ -d $(BUILD_OUT) ]; then \
		echo "Copying $(BUILD_OUT) to $(RTT_MOT_BOTSORT_TRACK_ELF_INSTALL_PATH)"; \
		cp -r $(BUILD_OUT)/* $(RTT_MOT_BOTSORT_TRACK_ELF_INSTALL_PATH); \
	else \
		echo "No build output found in $(BUILD_OUT), skipping copy."; \
	fi
endif
	@echo "Make botsort track samples done."

clean:
	@rm -rf $(RTT_MOT_BOTSORT_TRACK_ELF_INSTALL_PATH)/*
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BUILD_OUT)
	@echo "Clean botsort track samples done."