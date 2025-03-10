include ../mkenv.mk

dir-y :=

dir-$(CONFIG_RTT_OPENBLAS_CBLAS_SAXPY) += openblas_cblas_saxpy
dir-$(CONFIG_RTT_OPENBLAS_CBLAS_SGEMM) += openblas_cblas_sgemm
dir-$(CONFIG_RTT_OPENBLAS_CBLAS_SGER) += openblas_cblas_sger
dir-$(CONFIG_RTT_OPENBLAS_FORTRAN_DGEMM) += openblas_fortran_dgemm
# Add directories to the build system
dirs := $(sort $(dir-y))

.PHONY: all clean

# Default build arguments
BUILD_OUT ?= k230_bin

# Default target
all:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_OPENBLAS_EXAMPLES)/*

ifeq ($(CONFIG_RTT_ENABLE_BUILD_OPENBLAS_EXAMPLES),y)
	@rm -rf $(BUILD_OUT)
	@mkdir -p $(RTT_EXAMPLES_ELF_INSTALL_PATH_OPENBLAS_EXAMPLES)

	./build_app.sh || exit $?
	if [ -d $(BUILD_OUT) ]; then \
		echo "Copying $(BUILD_OUT) to $(RTT_EXAMPLES_ELF_INSTALL_PATH_OPENBLAS_EXAMPLES)"; \
		cp -r $(BUILD_OUT)/* $(RTT_EXAMPLES_ELF_INSTALL_PATH_OPENBLAS_EXAMPLES); \
	else \
		echo "No build output found in $(BUILD_OUT), skipping copy."; \
	fi
endif
	@echo "Make openblas samples done."

clean:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_OPENBLAS_EXAMPLES)/*
	@rm -rf $(BUILD_OUT)
	@echo "Clean openblas samples done."
