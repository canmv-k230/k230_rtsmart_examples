ifneq ($(MKENV_INCLUDED_RTSMART_EXAMPLE),1)
include ../mkenv.mk
endif

out_dir := $(SDK_RTSMART_BUILD_DIR)/examples/peripheral
elf_install_dir := $(RTT_EXAMPLES_ELF_INSTALL_PATH_PERIPHERAL_EXAMPLES)/

.PHONY: all clean
all:
	mkdir -p $(out_dir);
	mkdir -p $(elf_install_dir);
	cmake -B $(out_dir) -DCMAKE_INSTALL_PREFIX=$(elf_install_dir) . || exit $?;
	make -C $(out_dir) install || exit $?;

clean:
	@make -C $(out_dir) clean || true
	@rm -rf $(out_dir) || true
