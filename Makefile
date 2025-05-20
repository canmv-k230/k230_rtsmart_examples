include mkenv.mk

dir-y :=
dir-$(CONFIG_RTT_ENABLE_BUILD_PERIPHERAL_EXAMPLES) += peripheral
dir-$(CONFIG_RTT_ENABLE_BUILD_AI_EXAMPLES) += ai_poc
dir-$(CONFIG_RTT_ENABLE_BUILD_KPU_RUN_EXAMPLES) += kpu_run_yolov8
dir-$(CONFIG_RTT_ENABLE_BUILD_AI2D_EXAMPLES) += usage_ai2d
dir-$(CONFIG_RTT_ENABLE_BUILD_INTEGRATED_EXAMPLES) += integrated_poc
dir-$(CONFIG_RTT_ENABLE_BUILD_FACE_DETECTION) += face_detection
dir-$(CONFIG_RTT_ENABLE_BUILD_FACE_RECOGNITION) += face_recognition
dir-$(CONFIG_RTT_ENABLE_BUILD_YOLO) += YOLO
dir-$(CONFIG_RTT_ENABLE_BUILD_OPENCV_EXAMPLES) += opencv_examples
dir-$(CONFIG_RTT_ENABLE_BUILD_OPENBLAS_EXAMPLES) += openblas_examples

# Add directories to the build system
dirs := $(sort $(dir-y))

.PHONY: all clean

all:
ifeq ($(CONFIG_RTT_ENABLE_BUILD_EXAMPLES),y)
	@rm -rf elf
	@$(foreach dir,$(dirs),make -C $(dir) all || exit 1;)
endif
	@echo "Make rtsmart samples done."

clean:
	@rm -rf elf
ifeq ($(CONFIG_RTT_ENABLE_BUILD_EXAMPLES),y)
	@$(foreach dir,$(dirs),make -C $(dir) clean;)
endif
	@echo "Clean rtsmart samples done."
