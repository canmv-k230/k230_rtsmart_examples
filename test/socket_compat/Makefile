export MKENV_INCLUDED_RTSMART_EXAMPLE=1
include ../../mkenv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/toolchain_riscv64_musl.mk

DIR := $(shell basename `pwd`)
BUILD := $(SDK_RTSMART_BUILD_DIR)/examples/test/$(DIR)
K230_BIN := $(SDK_RTSMART_SRC_DIR)/examples/elf/test/$(DIR).elf
PC_BIN := $(BUILD)/socket_compat_peer

include $(SDK_RTSMART_SRC_DIR)/libs/mk/librtsmart_hal.mk

COMMON_FLAGS := -I. -Wall -Wextra -ffunction-sections -fdata-sections
K230_CFLAGS := $(COMMON_FLAGS) -std=gnu99 -march=rv64imafdcv -mabi=lp64d
K230_CFLAGS += -mcmodel=medany -Os $(LIB_CFLAGS)
K230_CFLAGS += -I$(SDK_SRC_ROOT_DIR)/include -include "generated/autoconf.h"
K230_LDFLAGS := -T $(SDK_RTSMART_SRC_DIR)/libs/mk/link.lds --static
K230_LDFLAGS += -Wl,--gc-sections $(LIB_LDFLAGS)
HOST_CC ?= cc
HOST_CFLAGS ?= -O2 -g
HOST_CFLAGS += $(COMMON_FLAGS) -std=gnu99

.PHONY: all k230 pc clean distclean

all: k230 pc

k230: $(K230_BIN)

pc: $(PC_BIN)

$(K230_BIN): socket_compat_test.c socket_test_protocol.h $(wildcard $(lib_rtsmart_hal_lib_dir)/*.a) | $(dir $(K230_BIN)) $(BUILD)
	@echo "[CC] $@"
	$(Q)$(CC) $(K230_CFLAGS) socket_compat_test.c $(K230_LDFLAGS) -o $@
	$(Q)$(STRIP) $@

$(PC_BIN): socket_compat_peer.c socket_test_protocol.h | $(BUILD)
	@echo "[HOSTCC] $@"
	$(Q)$(HOST_CC) $(HOST_CFLAGS) socket_compat_peer.c -o $@

$(dir $(K230_BIN)) $(BUILD):
	$(Q)mkdir -p $@

clean:
	$(Q)$(RM) $(K230_BIN) $(PC_BIN)

distclean: clean
	$(Q)$(RM) -r $(BUILD)
