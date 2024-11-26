include mkenv.mk

include $(SDK_SRC_ROOT_DIR)/.config

dirs := ai_poc

.PHONY: all clean

all:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH)/*

ifeq ($(CONFIG_RTT_ENABLE_BUILD_EXAMPLES),y)
	@$(foreach dir,$(dirs),make -C $(dir) all;)
endif
	@echo "Make rtsmart samples done."

clean:
	@rm -rf $(RTT_EXAMPLES_ELF_INSTALL_PATH)/*

ifeq ($(CONFIG_RTT_ENABLE_BUILD_EXAMPLES),y)
	@$(foreach dir,$(dirs),make -C $(dir) clean;)
endif
	@echo "Clean rtsmart samples done."
