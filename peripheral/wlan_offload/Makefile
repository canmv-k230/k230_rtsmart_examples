include ../../mkenv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/toolchain_riscv64_musl.mk

DIR := $(shell basename `pwd`)
BUILD := $(SDK_RTSMART_BUILD_DIR)/examples/peripheral/$(DIR)
BIN := $(RTT_PERIPHERAL_EXAMPLES_ELF_INSTALL_PATH)/$(DIR).elf
SRC := main.c
OBJ := $(BUILD)/main.o
DEP := $(OBJ:.o=.d)

include $(SDK_RTSMART_SRC_DIR)/libs/mk/libwlan_offload.mk
WLAN_OFFLOAD_LIB := $(lib_wlan_offload_dir)/lib/libwlan_offload.a

CFLAGS := -std=gnu99 -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -Os
CFLAGS += -Wall -Wextra -Werror -ffunction-sections -fdata-sections
CFLAGS += $(LIB_CFLAGS) -I$(SDK_SRC_ROOT_DIR)/include
CFLAGS += -include "generated/autoconf.h"
LDFLAGS := -T $(SDK_RTSMART_SRC_DIR)/libs/mk/link.lds --static
LDFLAGS += -Wl,--gc-sections $(LIB_LDFLAGS)

.PHONY: all clean distclean

all: $(BIN)
	@$(ECHO) [BUILD] examples peripheral $(DIR) done.

$(OBJ): $(SRC) $(WLAN_OFFLOAD_LIB) | $(BUILD)
	@$(ECHO) [CC] $<
	$(Q)$(CC) $(CFLAGS) -MD -MP -MF $(DEP) -c $< -o $@

$(BIN): $(OBJ) $(WLAN_OFFLOAD_LIB) | $(RTT_PERIPHERAL_EXAMPLES_ELF_INSTALL_PATH)
	@$(ECHO) [LD] $@
	$(Q)$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $@
	$(Q)$(STRIP) $@

$(WLAN_OFFLOAD_LIB):
	$(Q)$(MAKE) -C $(SDK_RTSMART_SRC_DIR)/libs/wlan_offload

$(BUILD) $(RTT_PERIPHERAL_EXAMPLES_ELF_INSTALL_PATH):
	$(Q)$(MKDIR) -p $@

clean:
	$(Q)$(RM) -f $(BIN)

distclean: clean
	$(Q)$(RM) -rf $(BUILD)

-include $(DEP)
