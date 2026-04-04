include ../mkenv.mk

subdirs-y := \
	adc \
	fpioa \
	gpio \
	i2c_ssd1306 \
	netmgmt \
	ota \
	pwm \
	rotary_encoder \
	sbus \
	spi_st7789 \
	spi_w25qxx \
	timer \
	touch \
	tsensor \
	uart \
	usb_hid_kbd \
	wdt \
	fft \

.PHONY: all clean distclean

all: $(subdirs-y)
	@$(ECHO) [BUILD] examples peripheral done.

ifneq ($(strip $(subdirs-y)),)
.PHONY: $(subdirs-y)

$(subdirs-y):
	@echo "[BUILD] examples peripheral $@"
	@$(MAKE) -C $@ all || exit $?;
endif

clean:
	@if [ -n "$(subdirs-y)" ]; then \
		for dir in $(subdirs-y); do \
			echo "[CLEAN] examples peripheral $$dir"; \
			$(MAKE) -C $$dir clean || exit $?; \
		done; \
	fi

distclean:
	@if [ -n "$(subdirs-y)" ]; then \
		for dir in $(subdirs-y); do \
			echo "[DISTCLEAN] examples peripheral $$dir"; \
			$(MAKE) -C $$dir distclean || exit $?; \
		done; \
	fi
