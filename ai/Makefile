include ../mkenv.mk

subdirs-y :=

subdirs-$(CONFIG_RTSMART_AI_EXAMPLES_ENABLE_AI_DEMO) += ai_demo
subdirs-$(CONFIG_RTSMART_AI_EXAMPLES_ENABLE_CLOUDPLAT_DEPLOY_CODE) += cloudplat_deploy_code
subdirs-$(CONFIG_RTSMART_AI_EXAMPLES_ENABLE_FACE_DETECTION) += face_detection
subdirs-$(CONFIG_RTSMART_AI_EXAMPLES_ENABLE_FACE_RECOGNITION) += face_recognition
subdirs-$(CONFIG_RTSMART_AI_EXAMPLES_ENABLE_MULTI_OBJECT_TRACKING) += multi_object_tracking/botsort_track_app multi_object_tracking/bytetrack_track_app multi_object_tracking/deepsort_track_app multi_object_tracking/ocsort_track_app
subdirs-$(CONFIG_RTSMART_AI_EXAMPLES_ENABLE_TRIPLE_CAMERA_AI) += triple_camera_ai
subdirs-$(CONFIG_RTSMART_AI_EXAMPLES_ENABLE_TRIPLE_CAMERA_AI_RTSP) += triple_camera_ai_rtsp
subdirs-$(CONFIG_RTSMART_AI_EXAMPLES_ENABLE_USAGE_AI2D) += usage_ai2d
subdirs-$(CONFIG_RTSMART_AI_EXAMPLES_ENABLE_USAGE_KPU) += usage_kpu
subdirs-$(CONFIG_RTSMART_AI_EXAMPLES_ENABLE_UVC_FACE_DETECTION) += uvc_face_detection
subdirs-$(CONFIG_RTSMART_AI_EXAMPLES_ENABLE_YOLO) += yolo

.PHONY: all clean distclean allclean

all: $(subdirs-y)
	@$(ECHO) [BUILD] AI examples done.

ifneq ($(strip $(subdirs-y)),)
.PHONY: $(subdirs-y)

$(subdirs-y):
	@echo "[BUILD] AI example $@"
	@$(MAKE) -C $@ all || exit $?;
endif

clean:
	@if [ -n "$(subdirs-y)" ]; then \
		for dir in $(subdirs-y); do \
			echo "[CLEAN] AI example $$dir"; \
			$(MAKE) -C $$dir clean || exit $?; \
		done; \
	fi

distclean:
	@if [ -n "$(subdirs-y)" ]; then \
		for dir in $(subdirs-y); do \
			echo "[DISTCLEAN] AI example $$dir"; \
			$(MAKE) -C $$dir distclean || exit $?; \
		done; \
	fi

allclean:
	@if [ -n "$(subdirs-y)" ]; then \
		for dir in $(subdirs-y); do \
			echo "[ALLCLEAN] Removing build and k230_bin in $$dir"; \
			rm -rf $$dir/build $$dir/k230_bin; \
		done; \
	fi
