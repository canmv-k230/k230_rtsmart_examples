# Include base configuration
include ../../mkenv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/toolchain_riscv64_musl.mk

# Build configuration
DIR 	:= $(shell basename `pwd`)
BUILD 	:= $(SDK_RTSMART_BUILD_DIR)/examples/integrated_poc/$(DIR)
INSTALL := $(RTT_EXAMPLES_ELF_INSTALL_PATH_INTERGRATED_POC)/$(DIR)
BIN 	:= $(INSTALL)/$(DIR).elf

# Source files handling
C_SRCS    := $(wildcard src/*.c) $(wildcard ocsort/*.c)
CXX_SRCS  := $(wildcard src/*.cc) $(wildcard src/*.cpp) \
             $(wildcard botsort/*.cc) $(wildcard botsort/*.cpp) \
			 $(wildcard src/video_stream_pipeline/*.cc) $(wildcard src/video_stream_pipeline/video_source/*.cc)

# Consistent variable naming
C_OBJS    := $(addprefix $(BUILD)/, $(addsuffix .o, $(basename $(notdir $(C_SRCS)))))
CXX_OBJS  := $(addprefix $(BUILD)/, $(addsuffix .o, $(basename $(notdir $(CXX_SRCS)))))

OBJS      := $(C_OBJS) $(CXX_OBJS)
DEPS      := $(OBJS:.o=.d)

# Assets
ASSETS_SRC		:= $(wildcard utils/*)
ASSETS_INSTALL 	:= $(addprefix $(INSTALL)/, $(notdir $(ASSETS_SRC)))

# Directory creation logic - Captured and sorted
OBJ_DIRS  := $(sort $(dir $(OBJS) $(BIN) $(ASSETS_INSTALL)))

# Library configuration
include $(SDK_RTSMART_SRC_DIR)/libs/mk/libmpp.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/libopencv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/libnncase.mk

# Compiler flags
C_CXX_FLAGS := -fopenmp -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -Os
C_CXX_FLAGS += -I. -I./src/video_stream_pipeline/video_source -I./src/video_stream_pipeline/include $(LIB_CFLAGS) -I$(SDK_SRC_ROOT_DIR)/include -include "generated/autoconf.h"
C_CXX_FLAGS += -Wall -Wextra -ffunction-sections -fdata-sections
C_CXX_FLAGS	+= -Ibotsort -I../../ai/multi_object_tracking/libs/Eigen32

CFLAGS 		:= $(C_CXX_FLAGS) -std=gnu99
CXX_FLAGS 	:= $(C_CXX_FLAGS) -std=c++20

# Linker flags
LDFLAGS := -T $(SDK_RTSMART_SRC_DIR)/libs/mk/link.lds --static -Wl,--gc-sections
LDFLAGS += $(LIB_LDFLAGS)

# Phony targets
.PHONY: all clean distclean

all: $(BIN) $(ASSETS_INSTALL)
	@$(ECHO) [BUILD] examples integrated_poc $(DIR) done.

# Rule to link the final ELF - Order-only dependency on directory
$(BIN): $(OBJS) | $(OBJ_DIRS)
	@echo "[LD] $@"
	$(Q)$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@
	$(Q)$(STRIP) $@

$(ASSETS_INSTALL): | $(OBJ_DIRS)
	@echo "[INSTALL] $@"
	$(Q)cp -a $(filter %/$(notdir $@), $(ASSETS_SRC)) $@

# Rule for C objects - Order-only dependency on directory
vpath %.c src botsort src/video_stream_pipeline src/video_stream_pipeline/video_source
$(BUILD)/%.o: %.c | $(OBJ_DIRS)
	@echo "[CC] $@"
	$(Q)$(CC) $(CFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

# Rule for C++ objects - Order-only dependency on directory
vpath %.cpp src botsort src/video_stream_pipeline src/video_stream_pipeline/video_source
$(BUILD)/%.o: %.cpp | $(OBJ_DIRS)
	@echo "[CXX] $@"
	$(Q)$(CXX) $(CXX_FLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

vpath %.cc src botsort src/video_stream_pipeline src/video_stream_pipeline/video_source
$(BUILD)/%.o: %.cc | $(OBJ_DIRS)
	@echo "[CXX] $@"
	$(Q)$(CXX) $(CXX_FLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

# Actual directory creation rule
$(OBJ_DIRS):
	$(Q)mkdir -p $@

clean:
	@rm -rf $(INSTALL)

distclean: clean
	@rm -rf $(BUILD)

-include $(DEPS)
