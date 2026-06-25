include ../mkenv.mk

subdirs-y :=

subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_AUDIO)           += sample_audio
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_CIPHER)          += sample_cipher/sample_aes sample_cipher/sample_hash sample_cipher/sample_hwhash sample_cipher/sample_pufs
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_CSC)             += sample_csc
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_DEMUXER)         += sample_demuxer
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_GSDMA)           += sample_gsdma
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_LOG)             += sample_log
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_MMZ)             += sample_mmz
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_MUXER)           += sample_muxer
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_NONAI_2D)        += sample_nonai_2d
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_OGG)             += sample_ogg
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_PLAYER)          += sample_player
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_PM)              += sample_pm
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_RTSPCLIENT)      += sample_rtspclient
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_RTSPPUSHER)      += sample_rtsppusher
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_RTSPSERVER)      += sample_rtspserver
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_UVC_DEV_PICTURE) += sample_uvc_dev_picture
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_UVC_DEV_VICAP)   += sample_uvc_dev_vicap
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_UVC_HOST)        += sample_uvc_host
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_VB)              += sample_vb
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_VDEC)            += sample_vdec
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_VENC)            += sample_venc
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_VO_MIX_ORDER)    += sample_vo_mix_order
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_VO_OSD)          += sample_vo_osd
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_VO_VIDEO)        += sample_vo_video
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_VO_WBC_DUMP)     += sample_vo_wbc_dump
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_FFT_DISPLAY)     += sample_fft_display
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_VICAP)     		+= sample_vicap_sensor
subdirs-$(CONFIG_RTSMART_MPP_ENABLE_SAMPLE_WEBRTC)    		+= sample_webrtc

.PHONY: all clean distclean

all: $(subdirs-y)
	@$(ECHO) [BUILD] examples mpp done.

ifneq ($(strip $(subdirs-y)),)
.PHONY: $(subdirs-y)

$(subdirs-y):
	@echo "[BUILD] examples mpp $@"
	@$(MAKE) -C $@ all || exit $?;
endif

clean:
	@if [ -n "$(subdirs-y)" ]; then \
		for dir in $(subdirs-y); do \
			echo "[CLEAN] examples mpp $$dir"; \
			$(MAKE) -C $$dir clean || exit $?; \
		done; \
	fi

distclean:
	@if [ -n "$(subdirs-y)" ]; then \
		for dir in $(subdirs-y); do \
			echo "[DISTCLEAN] examples mpp $$dir"; \
			$(MAKE) -C $$dir distclean || exit $?; \
		done; \
	fi
