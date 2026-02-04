include ../mkenv.mk

subdirs-y :=

subdirs-$(CONFIG_RTT_INTEGRATED_EXAMPLES_SMART_IPC)	+= smart_ipc

.PHONY: all clean distclean

all: $(subdirs-y)
	@$(ECHO) [BUILD] examples integrated_poc done.

ifneq ($(strip $(subdirs-y)),)
.PHONY: $(subdirs-y)

$(subdirs-y):
	@echo "[BUILD] examples integrated_poc $@"
	@$(MAKE) -C $@ all || exit $?;
endif

clean:
	@if [ -n "$(subdirs-y)" ]; then \
		for dir in $(subdirs-y); do \
			echo "[CLEAN] examples integrated_poc $$dir"; \
			$(MAKE) -C $$dir clean || exit $?; \
		done; \
	fi

distclean:
	@if [ -n "$(subdirs-y)" ]; then \
		for dir in $(subdirs-y); do \
			echo "[DISTCLEAN] examples integrated_poc $$dir"; \
			$(MAKE) -C $$dir distclean || exit $?; \
		done; \
	fi
