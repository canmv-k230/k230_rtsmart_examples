include ../mkenv.mk

subdirs-y :=

subdirs-$(CONFIG_RTSMART_3RD_PARTY_ENABLE_FREETYPE_SAMPLES) += freetype/freetype_basic
subdirs-$(CONFIG_RTSMART_3RD_PARTY_ENABLE_LVGL_SAMPLES) += lvgl/lvgl_basic lvgl/lvgl_sensor lvgl/lvgl_hid
subdirs-$(CONFIG_RTSMART_LIBS_MINIHTTP_ENABLE_SAMPLES) += minihttp/minihttp_basic minihttp/minihttp_advanced
subdirs-$(CONFIG_RTSMART_3RD_PARTY_ENABLE_TUYALINKSDK_SAMPLES) += tuya/tuya_basic
subdirs-$(CONFIG_RTSMART_3RD_PARTY_ENABLE_OPENCV_SAMPLES) += opencv
subdirs-$(CONFIG_RTSMART_3RD_PARTY_ENABLE_OPENBLAS_SAMPLES) += openblas

.PHONY: all clean distclean

all: $(subdirs-y)
	@$(ECHO) [BUILD] examples 3rd-party done.

ifneq ($(strip $(subdirs-y)),)
.PHONY: $(subdirs-y)

$(subdirs-y):
	@echo "[BUILD] examples 3rd-party $@"
	@$(MAKE) -C $@ all || exit $?;
endif

clean:
	@if [ -n "$(subdirs-y)" ]; then \
		for dir in $(subdirs-y); do \
			echo "[CLEAN] examples 3rd-party $$dir"; \
			$(MAKE) -C $$dir clean || exit $?; \
		done; \
	fi

distclean:
	@if [ -n "$(subdirs-y)" ]; then \
		for dir in $(subdirs-y); do \
			echo "[DISTCLEAN] examples 3rd-party $$dir"; \
			$(MAKE) -C $$dir distclean || exit $?; \
		done; \
	fi
