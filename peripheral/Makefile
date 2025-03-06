ifneq ($(MKENV_INCLUDED_RTSMART_EXAMPLE),1)
include ../mkenv.mk
endif

out_dir := $(SDK_RTSMART_BUILD_DIR)/examples/peripheral
elf_install_dir := $(SDK_BUILD_IMAGES_DIR)/sdcard/app/examples/peripheral

.PHONY: all clean
all:
	mkdir -p $(out_dir);
	mkdir -p $(elf_install_dir);
	cmake -B $(out_dir) -DCMAKE_INSTALL_PREFIX=$(elf_install_dir) . ;
	make -C $(out_dir) install;

clean:
	@make -C $(out_dir) clean;
	@rm -rf $(out_dir);
