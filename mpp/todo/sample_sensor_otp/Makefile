include ../mkenv.mk

include $(MPP_SRC_DIR)/userapps/sample/mpp.mk
include $(MPP_SRC_DIR)/userapps/sample/rt-smart.mk

DIR=$(shell basename `pwd`)
LOCAL_SRC_DIR = $(shell pwd)
BIN = $(MPP_USERAPP_SMAPLE_ELF_DIR)/$(DIR).elf
BUILD := $(SDK_RTSMART_BUILD_DIR)/mpp/userapps/sample/$(DIR)
LIBPATH = $(MPP_LIB_PATH)
LIBS = $(MPP_LIBS)

LOCAL_CFLAGS = -I$(LOCAL_SRC_DIR) \
			   -I$(LOCAL_SRC_DIR)/../sample_vo

SRCS = $(wildcard *.c)
SRCS += ../sample_vo/vo_test_case.c
SRCS += ../sample_vo/vo_bind_test.c

COBJS 	:= $(patsubst %, $(BUILD)/%, $(SRCS:.c=.o))
CDEPS 	:= $(patsubst %, $(BUILD)/%, $(SRCS:.c=.o.d))

CPP_SRCS = $(wildcard *.cpp)
CPP_OBJS 	:= $(patsubst %, $(BUILD)/%, $(CPP_SRCS:.cpp=.o))
CPP_DEPS 	:= $(patsubst %, $(BUILD)/%, $(CPP_SRCS:.cpp=.o.d))

OBJS := $(COBJS) $(CPP_OBJS)
DEPS := $(CDEPS) $(CPP_DEPS)

all: $(BIN)
	@echo "Make sample $(DIR) done."

$(COBJS): $(BUILD)/%.o : %.c
	@echo [CC] $@
	$(Q)$(CC) $(CC_CFLAGS) $(LOCAL_CFLAGS) $(BSP_CFLGAS) $(RTSMART_CFLAGS) $(MPP_USER_CFLGAS) -MD -MP -MF $@.d -c $< -o $@

$(BIN): $(OBJS)
	@echo [LD] $@
	$(Q)$(CC) -o $(BIN) $(LINKFLAG) -Wl,--whole-archive -Wl,--no-whole-archive -n --static $(OBJS) -L$(LIBPATH) -Wl,--start-group $(LIBS) -Wl,--end-group

clean:
	$(Q)rm -rf $(BIN) $(OBJS) $(DEPS)

.PHONY: all clean

# $(sort $(var)) removes duplicates
#
# The net effect of this, is it causes the objects to depend on the
# object directories (but only for existence), and the object directories
# will be created if they don't exist.
OBJ_DIRS = $(sort $(dir $(OBJS)))
$(OBJS): | $(OBJ_DIRS)
$(OBJ_DIRS):
	$(Q)mkdir -p $@

-include $(DEPS)
