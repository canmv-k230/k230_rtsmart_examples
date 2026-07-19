# Include base configuration.
include ../../mkenv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/toolchain_riscv64_musl.mk

PROGRAM := xiaozhi_client
BUILD := $(SDK_RTSMART_BUILD_DIR)/examples/3rd_party/xiaozhi
BIN := $(RTT_3RD_PARTY_EXAMPLES_ELF_INSTALL_PATH)/$(PROGRAM).elf

XIAOZHI_ASSET_SOURCE := $(CURDIR)/assets
XIAOZHI_ASSET_DEST := $(RTT_3RD_PARTY_EXAMPLES_ELF_INSTALL_PATH)/xiaozhi_assets
XIAOZHI_AUDIO_LIB_DIR := $(SDK_RTSMART_SRC_DIR)/mpp/kernel/lib
XIAOZHI_KWS_SOURCE_DIR := $(SDK_RTSMART_SRC_DIR)/examples/ai/ai_demo/kws
XIAOZHI_AI_COMMON_DIR := $(SDK_RTSMART_SRC_DIR)/examples/ai/ai_demo/common_files
XIAOZHI_WAKE_MODEL_SOURCE := $(SDK_RTSMART_SRC_DIR)/libs/kmodel/ai_poc/kmodel/kws.kmodel
XIAOZHI_WAKE_MODEL_ASSET := $(XIAOZHI_ASSET_SOURCE)/kws.kmodel

C_SRCS := $(shell find src -type f -name '*.c' -print | sort)
C_OBJS := $(patsubst %.c,$(BUILD)/%.o,$(C_SRCS))
XIAOZHI_WAKE_CXX_SRCS := src/xiaozhi_wake.cpp
XIAOZHI_KWS_CXX_SRCS := $(XIAOZHI_KWS_SOURCE_DIR)/kws.cc \
	$(XIAOZHI_KWS_SOURCE_DIR)/feature_pipeline.cc \
	$(XIAOZHI_KWS_SOURCE_DIR)/fft.cc
XIAOZHI_WAKE_CXX_OBJS := $(patsubst src/%.cpp,$(BUILD)/src/%.o,$(XIAOZHI_WAKE_CXX_SRCS))
XIAOZHI_KWS_CXX_OBJS := $(patsubst $(XIAOZHI_KWS_SOURCE_DIR)/%.cc,$(BUILD)/kws/%.o,$(XIAOZHI_KWS_CXX_SRCS))
XIAOZHI_AI_CXX_OBJS := $(patsubst $(XIAOZHI_AI_COMMON_DIR)/%.cc,$(BUILD)/common_files/%.o,$(XIAOZHI_AI_COMMON_DIR)/ai_base.cc)
OBJS := $(C_OBJS) $(XIAOZHI_WAKE_CXX_OBJS) $(XIAOZHI_KWS_CXX_OBJS) \
	$(XIAOZHI_AI_CXX_OBJS)
DEPS := $(OBJS:.o=.d)
OBJ_DIRS := $(sort $(dir $(OBJS) $(BIN)))

include $(SDK_RTSMART_SRC_DIR)/libs/mk/lib3rdparty.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/librtsmart_hal.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/libmpp.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/libnncase.mk

ifeq ($(CONFIG_RTSMART_3RD_PARTY_ENABLE_LVGL),y)
include $(SDK_RTSMART_SRC_DIR)/libs/mk/liblvgl.mk
XIAOZHI_HAS_LVGL := 1
else
XIAOZHI_HAS_LVGL := 0
endif

CFLAGS := -fopenmp -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -Os
CFLAGS += -Iinclude $(LIB_CFLAGS) -I$(SDK_SRC_ROOT_DIR)/include
CXXFLAGS := -fopenmp -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -Os
CXXFLAGS += -Iinclude $(LIB_CFLAGS) -I$(SDK_SRC_ROOT_DIR)/include
CXXFLAGS += -I$(XIAOZHI_KWS_SOURCE_DIR) -I$(XIAOZHI_AI_COMMON_DIR)
CXXFLAGS += -I$(SDK_RTSMART_SRC_DIR)/libs/opencv/include/opencv4
CFLAGS += -DXIAOZHI_HAS_LVGL=$(XIAOZHI_HAS_LVGL)
CXXFLAGS += -DXIAOZHI_HAS_LVGL=$(XIAOZHI_HAS_LVGL)
CFLAGS += -include "generated/autoconf.h"
CXXFLAGS += -include "generated/autoconf.h"
CFLAGS += -Wall -Wextra -ffunction-sections -fdata-sections -std=gnu99
CXXFLAGS += -Wall -Wextra -ffunction-sections -fdata-sections -std=gnu++17

LDFLAGS := -T $(SDK_RTSMART_SRC_DIR)/libs/mk/link.lds --static -Wl,--gc-sections
LDFLAGS += $(LIB_LDFLAGS) -lwebsockets -lcjson
LDFLAGS += -L$(XIAOZHI_AUDIO_LIB_DIR) -laudio -lm

.PHONY: all clean distclean xiaozhi_resources

all: $(BIN) xiaozhi_resources
	@$(ECHO) [BUILD] examples 3rd-party $(PROGRAM) done.

xiaozhi_resources:
	@echo "[RESOURCE] Copy XiaoZhi LVGL resources"
	@$(MKDIR) -p $(XIAOZHI_ASSET_DEST)
	$(Q)cp $(XIAOZHI_WAKE_MODEL_SOURCE) $(XIAOZHI_WAKE_MODEL_ASSET)
	$(Q)rsync -aq --delete $(XIAOZHI_ASSET_SOURCE)/ $(XIAOZHI_ASSET_DEST)/

$(BIN): $(OBJS) | $(OBJ_DIRS)
	@echo "[LD] $@"
	$(Q)$(CXX) $(OBJS) $(LDFLAGS) -o $@
	$(Q)$(STRIP) $@

$(BUILD)/%.o: %.c | $(OBJ_DIRS)
	@echo "[CC] $<"
	$(Q)$(CC) $(CFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

$(BUILD)/src/%.o: src/%.cpp | $(OBJ_DIRS)
	@echo "[CXX] $<"
	$(Q)$(CXX) $(CXXFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

$(BUILD)/kws/%.o: $(XIAOZHI_KWS_SOURCE_DIR)/%.cc | $(OBJ_DIRS)
	@echo "[CXX] $<"
	$(Q)$(CXX) $(CXXFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

$(BUILD)/common_files/%.o: $(XIAOZHI_AI_COMMON_DIR)/%.cc | $(OBJ_DIRS)
	@echo "[CXX] $<"
	$(Q)$(CXX) $(CXXFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

$(OBJ_DIRS):
	@$(MKDIR) -p $@

clean:
	@rm -rf $(BUILD) $(BIN) $(XIAOZHI_ASSET_DEST)

distclean: clean

-include $(DEPS)
