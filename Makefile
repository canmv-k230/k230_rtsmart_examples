include mkenv.mk

include $(SDK_SRC_ROOT_DIR)/.config

dir-y :=

dir-$(CONFIG_RTT_ENABLE_BUILD_AI_EXAMPLES) += ai_poc
dir-$(CONFIG_RTT_ENABLE_BUILD_KPU_RUN_EXAMPLES) += kpu_run_yolov8
dir-$(CONFIG_RTT_ENABLE_BUILD_AI2D_EXAMPLES) += usage_ai2d

# Add directories to the build system
dirs := $(sort $(dir-y))

.PHONY: all clean

all:
ifeq ($(CONFIG_RTT_ENABLE_BUILD_EXAMPLES),y)
	@$(foreach dir,$(dirs),make -C $(dir) all;)
endif
	@echo "Make rtsmart samples done."

clean:
ifeq ($(CONFIG_RTT_ENABLE_BUILD_EXAMPLES),y)
	@$(foreach dir,$(dirs),make -C $(dir) clean;)
endif
	@echo "Clean rtsmart samples done."
