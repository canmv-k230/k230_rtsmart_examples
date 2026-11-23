# See README.md for the SDK layout, dependencies, and standalone checkout usage.
HAL ?= ../../../../libs/rtsmart_hal
PUFS := $(HAL)/drivers/pufs
SRCS := test_pufs_host.c ../test_hmac_long_key.c
HAL_SRCS := drv_pufs.c drv_pufs_hash.c drv_pufs_cipher.c drv_pufs_asym.c

.PHONY: check
check:
	@test -f "$(PUFS)/drv_pufs.h" || { \
	    echo "PUFS HAL not found at $(PUFS); set HAL=/path/to/rtsmart_hal (see README.md)." >&2; exit 1; }
	@set -e; bin=$$(mktemp); trap 'rm -f "$$bin"' EXIT; \
	$(CC) $(CFLAGS) -std=c99 -Wall -Wextra -Werror -I"$(PUFS)" -I"$(HAL)/utils" \
	    $(SRCS) $(foreach src,$(HAL_SRCS),"$(PUFS)/$(src)") \
	    -pthread -Wl,--wrap=open,--wrap=close,--wrap=ioctl -lcrypto -o "$$bin"; \
	"$$bin"
