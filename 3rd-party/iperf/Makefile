# Build the upstream iperf repository supplied by the SDK repo manifest.
include ../../mkenv.mk
include $(SDK_RTSMART_SRC_DIR)/libs/mk/toolchain_riscv64_musl.mk

PROGRAM := iperf3
IPERF_SOURCE := $(CURDIR)/iperf
BUILD := $(SDK_RTSMART_BUILD_DIR)/examples/3rd_party/iperf
CONFIG_STAMP := $(BUILD)/.configured
UPSTREAM_BIN := $(BUILD)/src/$(PROGRAM)
BIN := $(RTT_3RD_PARTY_EXAMPLES_ELF_INSTALL_PATH)/$(PROGRAM).elf
RTSMART_PATCH := $(CURDIR)/patches/0001-rtsmart-compat.patch

IPERF_CFLAGS := -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -Os
IPERF_CFLAGS += -ffunction-sections -fdata-sections
IPERF_CFLAGS += -DRTSMART
IPERF_LDFLAGS := -T $(SDK_RTSMART_SRC_DIR)/libs/mk/link.lds --static
IPERF_LDFLAGS += -Wl,--gc-sections
IPERF_CONFIGURE_FLAGS := --host=$(CROSS_COMPILE_PREFIX:%-=%)
IPERF_CONFIGURE_FLAGS += --build=$(shell $(IPERF_SOURCE)/config/config.guess 2>/dev/null)
IPERF_CONFIGURE_FLAGS += --disable-shared --enable-static --enable-static-bin
IPERF_CONFIGURE_FLAGS += --without-openssl --without-sctp

.PHONY: all clean distclean FORCE

all: $(BIN)
	@$(ECHO) [BUILD] examples 3rd-party $(PROGRAM) done.

$(BIN): $(CONFIG_STAMP) FORCE | $(RTT_3RD_PARTY_EXAMPLES_ELF_INSTALL_PATH)
	@echo "[MAKE] upstream $(PROGRAM)"
	$(Q)$(MAKE) -C $(BUILD)/src $(PROGRAM)
	@echo "[CP] $(BIN)"
	$(Q)cp $(UPSTREAM_BIN) $(BIN)
	$(Q)$(STRIP) $(BIN)

$(CONFIG_STAMP): Makefile $(RTSMART_PATCH) $(IPERF_SOURCE)/configure \
	$(IPERF_SOURCE)/src/iperf_api.c $(IPERF_SOURCE)/src/tcp_info.c | $(BUILD)
	@cd "$(BUILD)" && "$(IPERF_SOURCE)/configure" \
		$(IPERF_CONFIGURE_FLAGS) \
		CC="$(CC)" AR="$(AR)" RANLIB="$(CROSS_COMPILE)ranlib" \
		STRIP="$(STRIP)" CFLAGS="$(IPERF_CFLAGS)" \
		LDFLAGS="$(IPERF_LDFLAGS)"
	@cp "$(IPERF_SOURCE)/src/iperf_api.c" "$(BUILD)/src/iperf_api.c"
	@cp "$(IPERF_SOURCE)/src/tcp_info.c" "$(BUILD)/src/tcp_info.c"
	@patch --silent -d "$(BUILD)" -p1 < "$(RTSMART_PATCH)"
	@touch $@

$(BUILD):
	@$(MKDIR) -p $@

$(IPERF_SOURCE)/configure:
	@echo "error: upstream iperf source is missing from $(IPERF_SOURCE)"
	@false

$(RTT_3RD_PARTY_EXAMPLES_ELF_INSTALL_PATH):
	@$(MKDIR) -p $@

FORCE:

clean:
	@rm -rf $(BUILD) $(BIN)

distclean: clean
