include mkenv.mk

subdirs-y :=

subdirs-$(CONFIG_RTT_ENABLE_3RD_PARTY_EXAMPLES) += 3rd-party
subdirs-$(CONFIG_RTT_ENABLE_PERIPHERAL_EXAMPLES) += peripheral
subdirs-$(CONFIG_RTT_ENABLE_MPP_EXAMPLES) += mpp
subdirs-$(CONFIG_RTT_ENABLE_AI_EXAMPLES) += ai
subdirs-$(CONFIG_RTT_ENABLE_BUILD_INTEGRATED_EXAMPLES) += integrated_poc

ELF_INSTALL_DIR := $(SDK_RTSMART_SRC_DIR)/examples/elf/

.PHONY: all clean distclean

all: $(subdirs-y) | $(ELF_INSTALL_DIR)
	@$(ECHO) [BUILD] examples done.

$(ELF_INSTALL_DIR):
	@mkdir -p $@
	@echo "[INFO] Created ELF install directory: $@"

ifneq ($(strip $(subdirs-y)),)
.PHONY: $(subdirs-y)

$(subdirs-y):
	@echo "[BUILD] examples $@"
	@$(MAKE) -C $@ all || exit $?;
endif

clean:
	@if [ -n "$(subdirs-y)" ]; then \
		for dir in $(subdirs-y); do \
			echo "[CLEAN] examples $$dir"; \
			$(MAKE) -C $$dir clean|| exit $?; \
		done; \
	fi
	@rm -rf $(SDK_RTSMART_SRC_DIR)/examples/elf/

distclean: clean
	@if [ -n "$(subdirs-y)" ]; then \
		for dir in $(subdirs-y); do \
			echo "[DISTCLEAN] examples $$dir"; \
			$(MAKE) -C $$dir distclean|| exit $?; \
		done; \
	fi
