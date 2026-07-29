# Include base configuration
include ../../mkenv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/toolchain_riscv64_musl.mk

DIR := $(notdir $(CURDIR))
BUILD := $(SDK_RTSMART_BUILD_DIR)/examples/mpp/$(DIR)
BIN := $(RTT_MPP_EXAMPLES_ELF_INSTALL_PATH)/$(DIR).elf
CONFIG_SRC := configs/lens_correction_640x480.json
CONFIG_INSTALL := $(RTT_MPP_EXAMPLES_ELF_INSTALL_PATH)/$(DIR)_lens_correction_640x480.json
IDENTITY_CONFIG_SRC := configs/identity_640x480.json
IDENTITY_CONFIG_INSTALL := $(RTT_MPP_EXAMPLES_ELF_INSTALL_PATH)/$(DIR)_identity_640x480.json
VSE_DMA_CONFIG_SRC := configs/vse_dma_640x480.json
VSE_DMA_CONFIG_INSTALL := $(RTT_MPP_EXAMPLES_ELF_INSTALL_PATH)/$(DIR)_vse_dma_640x480.json
INPUT_SRC := assets/input_640x480_yuv420sp.bin
INPUT_INSTALL := $(RTT_MPP_EXAMPLES_ELF_INSTALL_PATH)/input_640x480_yuv420sp.bin

C_SRCS := src/sample_dw200.c src/dewarp_map.c
CXX_SRCS := src/matrix.cpp
C_OBJS := $(patsubst %.c,$(BUILD)/%.o,$(C_SRCS))
CXX_OBJS := $(patsubst %.cpp,$(BUILD)/%.o,$(CXX_SRCS))
OBJS := $(C_OBJS) $(CXX_OBJS)
DEPS := $(OBJS:.o=.d)
OBJ_DIRS := $(sort $(dir $(OBJS) $(BIN) $(CONFIG_INSTALL) $(IDENTITY_CONFIG_INSTALL) $(VSE_DMA_CONFIG_INSTALL) $(INPUT_INSTALL)))

include $(SDK_RTSMART_SRC_DIR)/libs/mk/lib3rdparty.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/libmpp.mk

COMMON_FLAGS := -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -Os
COMMON_FLAGS += -Iinclude $(LIB_CFLAGS)
COMMON_FLAGS += -I$(SDK_SRC_ROOT_DIR)/include -include "generated/autoconf.h"
COMMON_FLAGS += -Wall -Wextra -ffunction-sections -fdata-sections
CFLAGS := $(COMMON_FLAGS) -std=gnu99
CXXFLAGS := $(COMMON_FLAGS) -std=c++20

LDFLAGS := -T $(SDK_RTSMART_SRC_DIR)/libs/mk/link.lds --static -Wl,--gc-sections
LDFLAGS += $(LIB_LDFLAGS) -L$(CJSON_INSTALL_DIR)/lib -lcjson

.PHONY: all clean distclean

all: $(BIN) $(CONFIG_INSTALL) $(IDENTITY_CONFIG_INSTALL) $(VSE_DMA_CONFIG_INSTALL) $(INPUT_INSTALL)
	@$(ECHO) [BUILD] examples mpp $(DIR) done.

$(BIN): $(OBJS) | $(OBJ_DIRS)
	@echo "[LD] $@"
	$(Q)$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
	$(Q)$(STRIP) $@

$(CONFIG_INSTALL): $(CONFIG_SRC) | $(OBJ_DIRS)
	@echo "[INSTALL] $@"
	$(Q)cp -a $< $@

$(IDENTITY_CONFIG_INSTALL): $(IDENTITY_CONFIG_SRC) | $(OBJ_DIRS)
	@echo "[INSTALL] $@"
	$(Q)cp -a $< $@

$(VSE_DMA_CONFIG_INSTALL): $(VSE_DMA_CONFIG_SRC) | $(OBJ_DIRS)
	@echo "[INSTALL] $@"
	$(Q)cp -a $< $@

$(INPUT_INSTALL): $(INPUT_SRC) | $(OBJ_DIRS)
	@echo "[INSTALL] $@"
	$(Q)cp -a $< $@

$(BUILD)/%.o: %.c | $(OBJ_DIRS)
	@echo "[CC] $@"
	$(Q)$(CC) $(CFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

$(BUILD)/%.o: %.cpp | $(OBJ_DIRS)
	@echo "[CXX] $@"
	$(Q)$(CXX) $(CXXFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

$(OBJ_DIRS):
	$(Q)mkdir -p $@

clean:
	@rm -f $(BIN) $(CONFIG_INSTALL) $(IDENTITY_CONFIG_INSTALL) $(VSE_DMA_CONFIG_INSTALL) $(INPUT_INSTALL)

distclean: clean
	@rm -rf $(BUILD)

-include $(DEPS)
