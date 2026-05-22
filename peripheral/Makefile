include ../mkenv.mk

subdirs-y :=

subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_OTP_PROVISION) += otp_provision

subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_FS_SPEEDTEST) += fs_speedtest
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_ADC) += adc
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_FFT) += fft
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_FPIOA) += fpioa
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_GPIO) += gpio
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_GZIP_DECOMP) += gzip_decomp
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_I2C_SSD1306) += i2c_ssd1306
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_NETMGMT) += netmgmt
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_OTA) += ota
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_PMU) += pmu
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_PUFS) += pufs
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_PWM) += pwm
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_ROTARY_ENCODER) += rotary_encoder
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_SBUS) += sbus
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_SPI_ST7789) += spi_st7789
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_SPI_W25QXX) += spi_w25qxx
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_TIMER) += timer
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_TOUCH) += touch
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_TSENSOR) += tsensor
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_UART) += uart
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_USB_HID_KBD) += usb_hid_kbd
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_USB_HID_MOUSE) += usb_hid_mouse
subdirs-$(CONFIG_RTSMART_PERIPHERAL_ENABLE_WDT) += wdt

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
