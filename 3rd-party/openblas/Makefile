include ../../mkenv.mk

export RTT_3RD_PARTY_OPENBLAS_ELF_INSTALL_PATH := \
    $(SDK_RTSMART_SRC_DIR)/examples/elf/3rd_party/openblas

dirs := $(sort $(notdir $(wildcard */)))

.PHONY: all clean

BUILD_DIR ?= build
BUILD_OUT ?= k230_bin

all:
	@rm -rf $(RTT_3RD_PARTY_OPENBLAS_ELF_INSTALL_PATH)/*
	@rm -rf $(BUILD_OUT)
	@mkdir -p $(RTT_3RD_PARTY_OPENBLAS_ELF_INSTALL_PATH)

	./build_app.sh || exit $?
	if [ -d $(BUILD_OUT) ]; then \
		cp -r $(BUILD_OUT)/* $(RTT_3RD_PARTY_OPENBLAS_ELF_INSTALL_PATH); \
	fi

	@echo "Make openblas samples done."

clean:
	@rm -rf $(RTT_3RD_PARTY_OPENBLAS_ELF_INSTALL_PATH)/*
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BUILD_OUT)
	@echo "Clean openblas samples done."
