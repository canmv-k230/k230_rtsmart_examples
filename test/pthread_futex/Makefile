include ../../mkenv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/toolchain_riscv64_musl.mk

DIR := $(shell basename `pwd`)
BUILD := $(SDK_RTSMART_BUILD_DIR)/examples/test/$(DIR)
BIN := $(SDK_RTSMART_SRC_DIR)/examples/elf/test/$(DIR).elf

C_SRCS := $(wildcard *.c)
C_OBJS := $(patsubst %.c,$(BUILD)/%.o,$(C_SRCS))
DEPS := $(C_OBJS:.o=.d)
OBJ_DIRS := $(sort $(dir $(C_OBJS) $(BIN)))

include $(SDK_RTSMART_SRC_DIR)/libs/mk/librtsmart_hal.mk

CFLAGS := -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -Os
CFLAGS += -I. $(LIB_CFLAGS) -I$(SDK_SRC_ROOT_DIR)/include -include "generated/autoconf.h"
CFLAGS += -Wall -Wextra -ffunction-sections -fdata-sections -std=gnu99

LDFLAGS := -T $(SDK_RTSMART_SRC_DIR)/libs/mk/link.lds --static -Wl,--gc-sections
LDFLAGS += $(LIB_LDFLAGS)

.PHONY: all clean distclean .FORCE

all: $(BIN)

.FORCE:

$(BIN): .FORCE $(C_OBJS) | $(OBJ_DIRS)
	@echo "[LD] $@"
	$(Q)$(CC) $(CFLAGS) $(C_OBJS) $(LDFLAGS) -o $@
	$(Q)$(STRIP) $@

$(BUILD)/%.o: %.c | $(OBJ_DIRS)
	@echo "[CC] $@"
	$(Q)$(CC) $(CFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

$(OBJ_DIRS):
	$(Q)mkdir -p $@

clean:
	$(Q)rm -f $(BIN)

distclean: clean
	$(Q)rm -rf $(BUILD)

-include $(DEPS)
