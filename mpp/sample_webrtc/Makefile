# Include base configuration
include ../../mkenv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/toolchain_riscv64_musl.mk

# Build configuration
DIR := $(shell basename `pwd`)
BUILD := $(SDK_RTSMART_BUILD_DIR)/examples/mpp/$(DIR)
BIN := $(RTT_MPP_EXAMPLES_ELF_INSTALL_PATH)/$(DIR).elf

# Source files handling
C_SRCS    := $(wildcard *.c)
CXX_SRCS  := $(wildcard *.cpp)

# Consistent variable naming
C_OBJS    := $(patsubst %.c, $(BUILD)/%.o, $(C_SRCS))
CXX_OBJS  := $(patsubst %.cpp, $(BUILD)/%.o, $(CXX_SRCS))

OBJS      := $(C_OBJS) $(CXX_OBJS)
DEPS      := $(OBJS:.o=.d)

# Directory creation logic
OBJ_DIRS  := $(sort $(dir $(OBJS) $(BIN)))

# Library configuration
include $(SDK_RTSMART_SRC_DIR)/libs/mk/libmpp.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/librtsmart_hal.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/lib3rdparty.mk

# libpeer library paths
LIBPEER_SRC := $(SDK_RTSMART_SRC_DIR)/libs/3rd-party/libpeer/src
LIBPEER_LIB_DIR := $(SDK_RTSMART_BUILD_DIR)/libs/3rd-party/lib
LIBPEER_3RD_PARTY_LIB_DIR := $(SDK_RTSMART_SRC_DIR)/libs/3rd-party/libpeer/3rd-party/lib

# libpeer static libraries (direct path to avoid SDK global mbedtls conflict)
LIBPEER_LIBS := $(LIBPEER_LIB_DIR)/libpeer.a
LIBPEER_LIBS += $(LIBPEER_LIB_DIR)/libcjson.a
LIBPEER_LIBS += $(LIBPEER_3RD_PARTY_LIB_DIR)/libmbedcrypto.a
LIBPEER_LIBS += $(LIBPEER_3RD_PARTY_LIB_DIR)/libmbedx509.a
LIBPEER_LIBS += $(LIBPEER_3RD_PARTY_LIB_DIR)/libmbedtls.a
LIBPEER_LIBS += $(LIBPEER_3RD_PARTY_LIB_DIR)/libsrtp2.a
LIBPEER_LIBS += $(LIBPEER_3RD_PARTY_LIB_DIR)/libusrsctp.a

# Compiler flags
C_CXX_FLAGS := -fopenmp -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -Os
C_CXX_FLAGS += -I. $(LIB_CFLAGS) -I$(SDK_SRC_ROOT_DIR)/include -include "generated/autoconf.h"
C_CXX_FLAGS += -I$(LIBPEER_SRC)
C_CXX_FLAGS += -Wall -Wextra -ffunction-sections -fdata-sections

CFLAGS		:= $(C_CXX_FLAGS) -std=gnu99
CXX_FLAGS	:= $(C_CXX_FLAGS) -std=c++20

# Linker flags
# libpeer libs (with newer mbedtls) MUST come before SDK LIB_LDFLAGS (older mbedtls)
# so the linker resolves mbedtls symbols from libpeer's version first.
LDFLAGS := -T $(SDK_RTSMART_SRC_DIR)/libs/mk/link.lds --static -Wl,--gc-sections
LDFLAGS += -Wl,--start-group $(LIBPEER_LIBS) -Wl,--end-group
LDFLAGS += $(LIB_LDFLAGS)
LDFLAGS += -lpthread

# Phony targets
.PHONY: all clean distclean

all: $(BIN)
	@$(ECHO) [BUILD] examples mpp $(DIR) done.

# Rule to link the final ELF
$(BIN): $(OBJS) | $(OBJ_DIRS)
	@echo "[LD] $@"
	$(Q)$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@
	$(Q)$(STRIP) $@

# Rule for C objects
$(BUILD)/%.o: %.c | $(OBJ_DIRS)
	@echo "[CC] $@"
	$(Q)$(CC) $(CFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

# Rule for C++ objects
$(BUILD)/%.o: %.cpp | $(OBJ_DIRS)
	@echo "[CXX] $@"
	$(Q)$(CXX) $(CXX_FLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

# Actual directory creation rule
$(OBJ_DIRS):
	$(Q)mkdir -p $@

clean:
	@rm -rf $(BIN)

distclean: clean
	@rm -rf $(BUILD)

-include $(DEPS)
