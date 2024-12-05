include ../../mkenv.mk

include $(MPP_SRC_DIR)/middleware/mpp.mk
include $(MPP_SRC_DIR)/middleware/rt-smart.mk

include ../toolchain.mk

CURRECT_DIR_NAME=$(shell basename `pwd`)
LOCAL_SRC_DIR = $(shell pwd)
BIN = $(LOCAL_SRC_DIR)/$(CURRECT_DIR_NAME).elf

LIBPATH = $(MPP_LIB_PATH)
LIBPATH += -L$(MPP_SRC_DIR)/middleware/lib
LIBPATH += -L$(NNCASE_SRC_DIR)/riscv64/rvvlib -L$(NNCASE_SRC_DIR)/riscv64/nncase/lib
LIBPATH += -L$(OPENCV_SRC_DIR)/lib -L$(OPENCV_SRC_DIR)/3rdparty/lib -L$(OPENCV_SRC_DIR)/lib/opencv4/3rdparty

LIVE555_LIBS = -lrtsp_server -lliveMedia -lgroupsock -lBasicUsageEnvironment -lUsageEnvironment
NNCASE_LIBS = -lrvv -lNncase.Runtime.Native -lnncase.rt_modules.k230 -lfunctional_k230
OPENCV_LIBS = -lstdc++ -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lzlib -llibjpeg-turbo -llibopenjp2 -llibpng -llibtiff -llibwebp -lcsi_cv -latomic
LIBS = $(MPP_LIBS)
LIBS += $(LIVE555_LIBS)
LIBS += $(NNCASE_LIBS)
LIBS += $(OPENCV_LIBS)

RTSP_SERVER = $(MPP_SRC_DIR)/middleware/src/rtsp_server
INCPATH = -I$(LOCAL_SRC_DIR) -I$(LOCAL_SRC_DIR)/face_detection -I$(RTSP_SERVER)/include -I$(NNCASE_SRC_DIR)/riscv64/rvvlib/include -I$(NNCASE_SRC_DIR)/riscv64 -I$(NNCASE_SRC_DIR)/riscv64/nncase/include -I$(NNCASE_SRC_DIR)/riscv64/nncase/include/nncase/runtime -I$(OPENCV_SRC_DIR)/include/opencv4

CC_CFLAGS_EX=-mcmodel=medany -march=rv64imafdcv -mabi=lp64d -Wall -O0 -g -gdwarf-2 -n --static $(KCFLAGS)
LIVE555_COMPILE_OPTS = -DSOCKLEN_T=socklen_t -DNO_SSTREAM=1 -D_LARGEFILE_SOURCE=1 -D_FILE_OFFSET_BITS=64 -DNO_OPENSSL=1
WERROR_FLAGS = -Wno-register -Wno-multichar -Wno-pessimizing-move -Wno-deprecated-declarations -Wno-unused-result -Wno-unused-variable -Wno-format -Wno-return-type -Wno-sign-compare -Wno-unused-label
CXX_FLAGS = $(WERROR_FLAGS) \
			$(LIVE555_COMPILE_OPTS) \

LINKFLAG = -T ../link.lds

SRCS = $(wildcard $(LOCAL_SRC_DIR)/*.c) $(wildcard $(LOCAL_SRC_DIR)/face_detection/*.c)
OBJS = $(patsubst %.c,%.o,$(SRCS))
CPP_SRCS = $(wildcard $(LOCAL_SRC_DIR)/*.cpp) $(wildcard $(LOCAL_SRC_DIR)/face_detection/*.cpp)
CPP_OBJS = $(patsubst %.cpp,%.o,$(CPP_SRCS))

BIN_DIR = ../k230_bin
all:$(BIN)
	@echo "${PWD}/Makefile all"
	@mkdir -p $(BIN_DIR)
	@cp $(BIN) $(BIN_DIR)
	@cp -a ${SDK_RTSMART_SRC_DIR}/libs/kmodel/ai_poc/kmodel/face_detection_320.kmodel $(BIN_DIR)
	echo "$$$$$$$$$$$$$$$$$$$$list cpp objs"
	echo $(CPP_OBJS)
	@rm -rf $(BIN) $(OBJS) $(CPP_OBJS)

$(OBJS): %.o : %.c
	@echo CC $@
	@$(CC) $(CC_CFLAGS_EX) $(INCPATH) $(BSP_CFLGAS) $(RTSMART_CFLAGS) $(MPP_USER_CFLGAS) -c $< -o $@

$(CPP_OBJS): %.o : %.cpp
	@echo CXX $@
	@$(CXX) $(CC_CFLAGS_EX) $(CXX_FLAGS) $(INCPATH) $(BSP_CFLGAS) $(RTSMART_CFLAGS) $(MPP_USER_CFLGAS) -c $< -o $@

$(BIN): $(OBJS) $(CPP_OBJS)
	@echo LD $@
	$(CXX) -o $(BIN) $(LINKFLAG) -n --static $^ -L$(LIBPATH) -Wl,--start-group $(LIBS) -Wl,--end-group

clean:
	echo "${PWD}/Makefile clean"
	rm -rf $(BIN) $(OBJS) $(CPP_OBJS)

.PHONY: all clean
