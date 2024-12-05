include ../mkenv.mk

dir-y :=

# Directory assignment based on Kconfig options
dir-$(CONFIG_RTT_INTEGRATED_EXAMPLES_SMART_IPC) += smart_ipc

# Add directories to the build system
dirs := $(sort $(dir-y))

.PHONY: all clean

BUILD_OUT ?= k230_bin

all:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_INTERGRATED_POC)/*

ifeq ($(CONFIG_RTT_ENABLE_BUILD_INTEGRATED_EXAMPLES),y)
	@$(foreach dir,$(dirs), \
	$(MAKE) -C $(dir) all || exit $?;\
	if [ -d $(BUILD_OUT) ]; then \
			echo "Copying $(BUILD_OUT) to $(RTT_EXAMPLES_ELF_INSTALL_PATH_INTERGRATED_POC)"; \
			cp -r $(BUILD_OUT)/* $(RTT_EXAMPLES_ELF_INSTALL_PATH_INTERGRATED_POC); \
	else \
			echo "No build output found in $(dir)/$(BUILD_OUT), skipping copy."; \
	fi;)
endif
	@echo "Make integrated poc done."

clean:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH_INTERGRATED_POC)/*
	@$(foreach dir,$(dirs),$(MAKE) -C $(dir) clean || exit $?;)
	@rm -rf $(BUILD_OUT)
	@echo "Clean integrated poc done."
