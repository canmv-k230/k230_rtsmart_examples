include ../mkenv.mk

dir-y :=

dir-$(CONFIG_RTT_YOLO) += YOLO

# Add directories to the build system
dirs := $(sort $(dir-y))

.PHONY: all clean

# Default build arguments
BUILD_OUT ?= k230_bin

# Default target
all:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_YOLO)/*

ifeq ($(CONFIG_RTT_ENABLE_BUILD_YOLO),y)
	./build_app.sh
	if [ -d $(BUILD_OUT) ]; then \
		echo "Copying $(BUILD_OUT) to $(RTT_EXAMPLES_ELF_INSTALL_PATH_YOLO)"; \
		cp -r $(BUILD_OUT)/* $(RTT_EXAMPLES_ELF_INSTALL_PATH_YOLO); \
	else \
		echo "No build output found in $(BUILD_OUT), skipping copy."; \
	fi
endif
	@echo "Make kpu run samples done."

clean:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_YOLO)/*
	@rm -rf out k230_bin
	@echo "Clean kpu run samples done."