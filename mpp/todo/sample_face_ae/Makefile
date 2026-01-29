include ../mkenv.mk

include $(MPP_SRC_DIR)/userapps/sample/mpp.mk
include $(MPP_SRC_DIR)/userapps/sample/rt-smart.mk

DIR=$(shell basename `pwd`)
LOCAL_SRC_DIR = $(shell pwd)
BIN = $(MPP_USERAPP_SMAPLE_ELF_DIR)/face_ae/$(DIR).elf
BUILD := $(SDK_RTSMART_BUILD_DIR)/mpp/userapps/sample/$(DIR)
LIBPATH = $(MPP_LIB_PATH)
LIBS = $(MPP_LIBS)
AI_LIBPATH=$(MPP_SRC_DIR)/../libs/nncase
KMODEL_LIBPATH=$(MPP_SRC_DIR)/userapps/sample/kmodel

LOCAL_CFLAGS = -I$(LOCAL_SRC_DIR) -I$(LOCAL_SRC_DIR)/../sample_vo

WERROR_FLAGS = -Wno-register -Wno-multichar -Wno-pessimizing-move -Wno-deprecated-declarations -Wno-unused-result -Wno-unused-variable -Wno-format -Wno-return-type -Wno-sign-compare -Wno-unused-label

RISCV64_NN_INCLUDE = \
			-I$(AI_LIBPATH)/riscv64 \
			-I$(AI_LIBPATH)/riscv64/nncase/include \
			-I$(AI_LIBPATH)/riscv64/nncase/include/nncase \
			-I$(AI_LIBPATH)/riscv64/nncase/include/nncase/runtime \
			-I$(AI_LIBPATH)/riscv64/rvvlib/include \

RISCV64_NN_LIBS = \
			-L$(AI_LIBPATH)/riscv64/rvvlib/ -lrvv \
			-L$(AI_LIBPATH)/riscv64/nncase/lib/ -lNncase.Runtime.Native -lnncase.rt_modules.k230 -lfunctional_k230

CXX_FLAGS_AI = -std=c++20 \
			$(WERROR_FLAGS) \
			$(RISCV64_NN_INCLUDE) \
			$(RISCV64_NN_LIBS) \

SRCS = $(wildcard *.c)
COBJS 	:= $(patsubst %, $(BUILD)/%, $(SRCS:.c=.o))
CDEPS 	:= $(patsubst %, $(BUILD)/%, $(SRCS:.c=.o.d))

CPP_SRCS = $(wildcard *.cc)
CPP_OBJS 	:= $(patsubst %, $(BUILD)/%, $(CPP_SRCS:.cc=.o))
CPP_DEPS 	:= $(patsubst %, $(BUILD)/%, $(CPP_SRCS:.cc=.o.d))

OBJS := $(COBJS) $(CPP_OBJS)
DEPS := $(CDEPS) $(CPP_DEPS)

all: $(BIN)
	@echo "Make sample $(DIR) done."

$(COBJS): $(BUILD)/%.o : %.c
	@echo [CC] $@
	$(Q)$(CC) $(CC_CFLAGS) $(LOCAL_CFLAGS) $(BSP_CFLGAS) $(RTSMART_CFLAGS) $(MPP_USER_CFLGAS) -MD -MP -MF $@.d -c $< -o $@

$(CPP_OBJS): $(BUILD)/%.o : %.cc
	@echo [CXX] $@
	$(Q)$(CPP) $(CC_CFLAGS) $(CXX_FLAGS_AI) $(LOCAL_CFLAGS) $(BSP_CFLGAS) $(RTSMART_CFLAGS) $(MPP_USER_CFLGAS) -MD -MP -MF $@.d -c $< -o $@

$(BIN): $(OBJS)
	@echo [LD] $@
	$(Q)mkdir -p $(MPP_USERAPP_SMAPLE_ELF_DIR)/face_ae/
	$(Q)$(CPP) -o $(BIN)  $(LINKFLAG) -Wl,--whole-archive -Wl,--no-whole-archive -n --static $^ -L$(LIBPATH) $(CXX_FLAGS_AI) -Wl,--start-group $(LIBS) -Wl,--end-group
	$(Q)cp $(KMODEL_LIBPATH)/mobile_retinaface/test.kmodel $(MPP_USERAPP_SMAPLE_ELF_DIR)/face_ae/

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
