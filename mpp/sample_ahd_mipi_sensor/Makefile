# Include base configuration
include ../../mkenv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/toolchain_riscv64_musl.mk

DIR := $(shell basename `pwd`)
BUILD := $(SDK_RTSMART_BUILD_DIR)/examples/mpp/$(DIR)
BIN := $(RTT_MPP_EXAMPLES_ELF_INSTALL_PATH)/$(DIR).elf

C_SRCS := $(wildcard *.c)
OBJS := $(patsubst %.c,$(BUILD)/%.o,$(C_SRCS))
DEPS := $(OBJS:.o=.d)
OBJ_DIRS := $(sort $(dir $(OBJS) $(BIN)))

include $(SDK_RTSMART_SRC_DIR)/libs/mk/libmpp.mk

CFLAGS := -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -Os
CFLAGS += -I. $(LIB_CFLAGS) -I$(SDK_SRC_ROOT_DIR)/include
CFLAGS += -include "generated/autoconf.h" -std=gnu99
CFLAGS += -Wall -Wextra -ffunction-sections -fdata-sections
LDFLAGS := -T $(SDK_RTSMART_SRC_DIR)/libs/mk/link.lds --static -Wl,--gc-sections
LDFLAGS += $(LIB_LDFLAGS)

.PHONY: all clean distclean

all: $(BIN)
	@$(ECHO) [BUILD] examples mpp $(DIR) done.

$(BIN): $(OBJS) | $(OBJ_DIRS)
	@echo "[LD] $@"
	$(Q)$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@
	$(Q)$(STRIP) $@

$(BUILD)/%.o: %.c | $(OBJ_DIRS)
	@echo "[CC] $@"
	$(Q)$(CC) $(CFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

$(OBJ_DIRS):
	$(Q)mkdir -p $@

clean:
	@rm -f $(BIN)

distclean: clean
	@rm -rf $(BUILD)

-include $(DEPS)
