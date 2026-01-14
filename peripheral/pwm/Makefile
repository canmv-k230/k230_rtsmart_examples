# Include base configuration
include ../../mkenv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/toolchain_riscv64_musl.mk

# Build configuration
CURRECT_DIR_NAME := $(shell basename `pwd`)
BUILD := $(SDK_RTSMART_BUILD_DIR)/examples/peripheral/$(CURRECT_DIR_NAME)
BIN := $(RTT_EXAMPLES_ELF_INSTALL_PATH_PERIPHERAL_EXAMPLES)/$(CURRECT_DIR_NAME).elf

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

# Compiler flags
CFLAGS := -std=gnu99 -fopenmp -march=rv64imafdcv -mabi=lp64d -mcmodel=medany
CFLAGS += -I. $(LIB_CFLAGS) -I$(SDK_SRC_ROOT_DIR)/include -include "generated/autoconf.h"
CFLAGS += -Wall -Wextra -ffunction-sections -fdata-sections

CXX_FLAGS := -I. $(LIB_CFLAGS) -I$(SDK_SRC_ROOT_DIR)/include -include "generated/autoconf.h"
CXX_FLAGS += -Wall -Wextra -ffunction-sections -fdata-sections

# Linker flags
LDFLAGS := -T $(SDK_RTSMART_SRC_DIR)/libs/mk/link.lds --static -Wl,--gc-sections
LDFLAGS += $(LIB_LDFLAGS)

# Phony targets
.PHONY: all clean distclean

all: $(BIN)
	@echo "[SUCCESS] Built RTSmart HAL sample $(CURRECT_DIR_NAME)"

# Rule to link the final ELF - Order-only dependency on directory
$(BIN): $(OBJS) | $(OBJ_DIRS)
	@echo "[LD] $@"
	$(Q)$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

# Rule for C objects - Order-only dependency on directory
$(BUILD)/%.o: %.c | $(OBJ_DIRS)
	@echo "[CC] $@"
	$(Q)$(CC) $(CFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

# Rule for C++ objects - Order-only dependency on directory
$(BUILD)/%.o: %.cpp | $(OBJ_DIRS)
	@echo "[CXX] $@"
	$(Q)$(CXX) $(CXX_FLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

# Actual directory creation rule
$(OBJ_DIRS):
	$(Q)mkdir -p $@

clean:
	@echo "Cleaning RTSmart HAL sample $(CURRECT_DIR_NAME) build artifacts..."
	@rm -rf $(BUILD)

distclean: clean
	@echo "Cleaning RTSmart HAL sample $(CURRECT_DIR_NAME) installed binaries..."
	@rm -f $(BIN)

-include $(DEPS)
