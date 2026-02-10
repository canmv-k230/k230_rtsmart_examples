# Include base configuration
include ../../../mkenv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/toolchain_riscv64_musl.mk

# Build configuration
DIR := $(shell basename `pwd`)
BUILD := $(SDK_RTSMART_BUILD_DIR)/examples/3rd_party/$(DIR)
BIN := $(RTT_3RD_PARTY_EXAMPLES_ELF_INSTALL_PATH)/tuya/$(DIR).elf

# Source files handling
C_SRCS    := $(wildcard *.c)
C_OBJS    := $(patsubst %.c, $(BUILD)/%.o, $(C_SRCS))

OBJS      := $(C_OBJS)
DEPS      := $(OBJS:.o=.d)

# Directory creation logic - Captured and sorted
OBJ_DIRS  := $(sort $(dir $(OBJS) $(BIN)))

# Library configuration
include $(SDK_RTSMART_SRC_DIR)/libs/mk/lib3rdparty.mk

# Compiler flags
CFLAGS := -fopenmp -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -Os
CFLAGS += -I. $(LIB_CFLAGS) -I$(lib3rd_party_inc_dir)/tuya -I$(SDK_SRC_ROOT_DIR)/include -include "generated/autoconf.h"
CFLAGS += -Wall -Wextra -ffunction-sections -fdata-sections
CFLAGS += -std=gnu99

# Tuya explicit libs (keep for clarity/ordering)
TUYA_LIBS := -ltuya_iot_core -lmbedtls -lmbedx509 -lmbedcrypto -lmbedtls_port

# Linker flags
LDFLAGS := -T $(SDK_RTSMART_SRC_DIR)/libs/mk/link.lds --static -Wl,--gc-sections
LDFLAGS += $(LIB_LDFLAGS)
LDFLAGS += $(TUYA_LIBS)

# Phony targets
.PHONY: all clean distclean

all: $(BIN)
	@$(ECHO) [BUILD] examples 3rd-party $(DIR) done.

# Rule to link the final ELF - Order-only dependency on directory
$(BIN): $(OBJS) | $(OBJ_DIRS)
	@echo "[LD] $@"
	$(Q)$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@
	$(Q)$(STRIP) $@

# Rule for C objects - Order-only dependency on directory
$(BUILD)/%.o: %.c | $(OBJ_DIRS)
	@echo "[CC] $@"
	$(Q)$(CC) $(CFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

# Actual directory creation rule
$(OBJ_DIRS):
	$(Q)mkdir -p $@

clean:
	@rm -f $(BIN)

distclean: clean
	@rm -rf $(BUILD)

-include $(DEPS)
