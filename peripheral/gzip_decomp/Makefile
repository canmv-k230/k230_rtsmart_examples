# Include base configuration
include ../../mkenv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/toolchain_riscv64_musl.mk

# Build configuration
DIR := $(shell basename `pwd`)
BUILD := $(SDK_RTSMART_BUILD_DIR)/examples/peripheral/$(DIR)
BIN := $(RTT_PERIPHERAL_EXAMPLES_ELF_INSTALL_PATH)/$(DIR).elf
FIXTURE_BUILD := $(BUILD)/fixtures
FIXTURE_INSTALL := $(RTT_PERIPHERAL_EXAMPLES_ELF_INSTALL_PATH)/$(DIR)_data
K230_GZIP := $(SDK_SRC_ROOT_DIR)/tools/image_tools/k230_priv_gzip.py
FIXTURE_GZ := \
	$(FIXTURE_BUILD)/empty.k230.gz \
	$(FIXTURE_BUILD)/small.k230.gz \
	$(FIXTURE_BUILD)/multi.k230.gz \
	$(FIXTURE_BUILD)/max.k230.gz

# Source files handling
C_SRCS    := $(wildcard *.c)
CXX_SRCS  := $(wildcard *.cpp)

# Consistent variable naming
C_OBJS    := $(patsubst %.c, $(BUILD)/%.o, $(C_SRCS))
CXX_OBJS  := $(patsubst %.cpp, $(BUILD)/%.o, $(CXX_SRCS))

OBJS      := $(C_OBJS) $(CXX_OBJS)
DEPS      := $(OBJS:.o=.d)

# Directory creation logic - Captured and sorted
OBJ_DIRS  := $(sort $(dir $(OBJS) $(BIN)))

# Library configuration
include $(SDK_RTSMART_SRC_DIR)/libs/mk/librtsmart_hal.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/libmpp.mk

# Compiler flags
C_CXX_FLAGS := -fopenmp -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -Os
C_CXX_FLAGS += -I. $(LIB_CFLAGS) -I$(SDK_SRC_ROOT_DIR)/include -include "generated/autoconf.h"
C_CXX_FLAGS += -Wall -Wextra -ffunction-sections -fdata-sections

CFLAGS 		:= $(C_CXX_FLAGS) -std=gnu99
CXX_FLAGS 	:= $(C_CXX_FLAGS) -std=c++20 

# Linker flags
LDFLAGS := -T $(SDK_RTSMART_SRC_DIR)/libs/mk/link.lds --static -Wl,--gc-sections
LDFLAGS += $(LIB_LDFLAGS)

# Phony targets
.PHONY: all clean distclean fixtures .FORCE

all: $(BIN) fixtures
	@$(ECHO) [BUILD] examples peripheral $(DIR) done.

fixtures: $(FIXTURE_GZ) | $(FIXTURE_INSTALL)
	$(Q)cp $(FIXTURE_GZ) $(FIXTURE_INSTALL)/

$(FIXTURE_BUILD)/empty.bin: | $(FIXTURE_BUILD)
	$(Q)truncate -s 0 $@

$(FIXTURE_BUILD)/small.bin: | $(FIXTURE_BUILD)
	$(Q)printf 'K230 gzip decompression small fixture\n' > $@

$(FIXTURE_BUILD)/multi.bin: | $(FIXTURE_BUILD)
	$(Q)truncate -s 393339 $@

$(FIXTURE_BUILD)/max.bin: | $(FIXTURE_BUILD)
	$(Q)truncate -s 16777216 $@

$(FIXTURE_BUILD)/%.k230.gz: $(FIXTURE_BUILD)/%.bin
	$(Q)python3 $(K230_GZIP) compress $< -o $@ -k -f -l 9

.FORCE:

# Rule to link the final ELF - Order-only dependency on directory
$(BIN): $(OBJS) .FORCE | $(OBJ_DIRS)
	@echo "[LD] $@"
	$(Q)$(CXX) $(CFLAGS) $(OBJS) $(LDFLAGS) -o $@
	$(Q)$(STRIP) $@

# Rule for C objects - Order-only dependency on directory
$(BUILD)/%.o: %.c | $(OBJ_DIRS)
	@echo "[CC] $@"
	$(Q)$(CC) $(CFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

# Rule for C++ objects - Order-only dependency on directory
$(BUILD)/%.o: %.cpp | $(OBJ_DIRS)
	@echo "[CXX] $@"
	$(Q)$(CXX) $(CXX_FLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

# Actual directory creation rule
$(OBJ_DIRS) $(FIXTURE_BUILD) $(FIXTURE_INSTALL):
	$(Q)mkdir -p $@

clean:
	@rm -rf $(BIN)
	@rm -rf $(FIXTURE_INSTALL)

distclean: clean
	@rm -rf $(BUILD)

-include $(DEPS)
