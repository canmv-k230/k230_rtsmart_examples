include ../../mkenv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/toolchain_riscv64_musl.mk

DIR := $(shell basename `pwd`)
BUILD := $(SDK_RTSMART_BUILD_DIR)/examples/peripheral/$(DIR)
BIN := $(RTT_PERIPHERAL_EXAMPLES_ELF_INSTALL_PATH)/$(DIR).elf

C_SRCS    := $(wildcard *.c)
C_OBJS    := $(patsubst %.c, $(BUILD)/%.o, $(C_SRCS))
OBJS      := $(C_OBJS)
DEPS      := $(OBJS:.o=.d)

OBJ_DIRS  := $(sort $(dir $(OBJS) $(BIN)))

include $(SDK_RTSMART_SRC_DIR)/libs/mk/librtsmart_hal.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/libmpp.mk

C_CXX_FLAGS := -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -Os
C_CXX_FLAGS += -I. $(LIB_CFLAGS) -I$(SDK_SRC_ROOT_DIR)/include -include "generated/autoconf.h"
C_CXX_FLAGS += -Wall -Wextra -ffunction-sections -fdata-sections

CFLAGS := $(C_CXX_FLAGS) -std=gnu99

LDFLAGS := -T $(SDK_RTSMART_SRC_DIR)/libs/mk/link.lds --static -Wl,--gc-sections
LDFLAGS += $(LIB_LDFLAGS)

.PHONY: all clean distclean .FORCE

all: $(BIN)
	@$(ECHO) [BUILD] examples peripheral $(DIR) done.

.FORCE:

$(BIN): $(OBJS) .FORCE | $(OBJ_DIRS)
	@echo "[LD] $@"
	$(Q)$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) -o $@
	$(Q)$(STRIP) $@

$(BUILD)/%.o: %.c | $(OBJ_DIRS)
	@echo "[CC] $@"
	$(Q)$(CC) $(CFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

$(OBJ_DIRS):
	$(Q)mkdir -p $@

clean:
	@rm -rf $(BIN)

distclean: clean
	@rm -rf $(BUILD)

-include $(DEPS)
