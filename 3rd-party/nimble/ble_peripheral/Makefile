include ../../../mkenv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/toolchain_riscv64_musl.mk

DIR := $(shell basename `pwd`)
BUILD := $(SDK_RTSMART_BUILD_DIR)/examples/3rd_party/nimble/$(DIR)
BIN := $(RTT_3RD_PARTY_EXAMPLES_ELF_INSTALL_PATH)/$(DIR).elf

C_SRCS := $(wildcard *.c)
C_OBJS := $(patsubst %.c,$(BUILD)/%.o,$(C_SRCS))
DEPS := $(C_OBJS:.o=.d)
OBJ_DIRS := $(sort $(dir $(C_OBJS) $(BIN)))

RTSMART_LINK_NIMBLE := y
include $(SDK_RTSMART_SRC_DIR)/libs/mk/lib3rdparty.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/librtsmart_hal.mk

CFLAGS := -std=gnu11 -pthread -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -Os
CFLAGS += -I. $(LIB_CFLAGS) -I$(SDK_SRC_ROOT_DIR)/include
CFLAGS += -I$(SDK_RTSMART_BUILD_DIR)/libs/3rd-party/include/nimble
CFLAGS += -include generated/autoconf.h
CFLAGS += -Wall -Wextra -ffunction-sections -fdata-sections

LDFLAGS := -T $(SDK_RTSMART_SRC_DIR)/libs/mk/link.lds --static -Wl,--gc-sections
LDFLAGS += $(LIB_LDFLAGS) -pthread

.PHONY: all clean distclean .FORCE

all: $(BIN)
	@$(ECHO) [BUILD] examples 3rd-party nimble $(DIR) done.

.FORCE:

$(BIN): $(C_OBJS) .FORCE | $(OBJ_DIRS)
	@$(ECHO) [LD] $@
	$(Q)$(CC) $(CFLAGS) $(C_OBJS) $(LDFLAGS) -o $@
	$(Q)$(STRIP) $@

$(BUILD)/%.o: %.c | $(OBJ_DIRS)
	@$(ECHO) [CC] $<
	$(Q)$(CC) $(CFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

$(OBJ_DIRS):
	$(Q)$(MKDIR) -p $@

clean:
	$(Q)$(RM) $(BIN)

distclean: clean
	$(Q)$(RM) -rf $(BUILD)

-include $(DEPS)
