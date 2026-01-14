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
	wdt \

.PHONY: all clean distclean

all:
	@$(foreach dir,$(subdirs-y),$(MAKE) -C $(dir) all || exit $?;)
	@echo "Make RT-Smart Peripheral Examples done."

clean:
	@$(foreach dir,$(subdirs-y),$(MAKE) -C $(dir) clean || exit $?;)
	@echo "Make RT-Smart Peripheral Examples clean done."

distclean:
	@$(foreach dir,$(subdirs-y),$(MAKE) -C $(dir) distclean || exit $?;)
	@echo "Make RT-Smart Peripheral Examples distclean done."
