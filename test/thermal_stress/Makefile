# Include base configuration
include ../../mkenv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/toolchain_riscv64_musl.mk

# Build configuration
DIR 	:= $(shell basename `pwd`)
BUILD 	:= $(SDK_RTSMART_BUILD_DIR)/examples/test/$(DIR)
INSTALL := $(SDK_RTSMART_SRC_DIR)/examples/elf/test/$(DIR)
BIN 	:= $(INSTALL)/$(DIR).elf

CXX_SRCS  := $(wildcard *.cpp)
CXX_OBJS  := $(patsubst %.cpp, $(BUILD)/%.o, $(CXX_SRCS))
OBJS      := $(CXX_OBJS)
DEPS      := $(OBJS:.o=.d)

# The KPU load model is installed next to the program.
ASSETS_SRC		:= $(SDK_RTSMART_SRC_DIR)/libs/kmodel/ai_poc/kmodel/yolov8n_640.kmodel
ASSETS_INSTALL 	:= $(addprefix $(INSTALL)/, $(notdir $(ASSETS_SRC)))

OBJ_DIRS  := $(sort $(dir $(OBJS) $(BIN) $(ASSETS_INSTALL)))

# Library configuration
include $(SDK_RTSMART_SRC_DIR)/libs/mk/libmpp.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/libnncase.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/librtsmart_hal.mk

# Compiler flags
C_CXX_FLAGS := -fopenmp -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -Os
C_CXX_FLAGS += -I. $(LIB_CFLAGS) -I$(SDK_SRC_ROOT_DIR)/include -include "generated/autoconf.h"
C_CXX_FLAGS += -Wall -Wextra -ffunction-sections -fdata-sections
CXX_FLAGS 	:= $(C_CXX_FLAGS) -std=c++20

# Linker flags
LDFLAGS := -T $(SDK_RTSMART_SRC_DIR)/libs/mk/link.lds --static -Wl,--gc-sections
LDFLAGS += $(LIB_LDFLAGS)
LDFLAGS += -lpthread

.PHONY: all clean distclean

all: $(BIN) $(ASSETS_INSTALL)
	@$(ECHO) [BUILD] examples test $(DIR) done.

$(BIN): $(OBJS) | $(OBJ_DIRS)
	@echo "[LD] $@"
	$(Q)$(CXX) $(CXX_FLAGS) $^ $(LDFLAGS) -o $@
	$(Q)$(STRIP) $@

$(ASSETS_INSTALL): $(ASSETS_SRC) | $(OBJ_DIRS)
	@echo "[INSTALL] $@"
	$(Q)cp -a $< $@

$(BUILD)/%.o: %.cpp | $(OBJ_DIRS)
	@echo "[CXX] $@"
	$(Q)$(CXX) $(CXX_FLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

$(OBJ_DIRS):
	$(Q)mkdir -p $@

clean:
	@rm -rf $(INSTALL)

distclean: clean
	@rm -rf $(BUILD)

-include $(DEPS)
