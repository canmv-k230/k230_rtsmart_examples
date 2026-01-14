#include "k230_ota.h"
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define OTA_DEFAULT_IMAGE "/data/ota_test.kdimg"

#define READ_CHUNK_SIZE (64 * 1024) // 64KB chunks

int main(int argc, char* argv[])
{
    const char* image_path = OTA_DEFAULT_IMAGE;
    int         fd_img     = -1;
    k230_ota_t* ota_ctx    = NULL;
    int         ret        = -1;
    uint8_t*    buf        = NULL;
    ssize_t     rd;

    if (argc >= 2 && argv && argv[1])
        image_path = argv[1];

    printf("[ota_test] Starting OTA with img=%s\n", image_path);

    // 1. Open the source image file
    fd_img = open(image_path, O_RDONLY, 0);
    if (fd_img < 0) {
        perror("[ota_test] Failed to open source image");
        goto _exit;
    }

    // 2. Create the OTA instance (User doesn't see the struct definition)
    ota_ctx = k230_ota_create();
    if (!ota_ctx) {
        fprintf(stderr, "[ota_test] Failed to create OTA session\n");
        goto _exit;
    }

    // 3. Allocate a buffer for streaming the data
    buf = (uint8_t*)malloc(READ_CHUNK_SIZE);
    if (!buf) {
        fprintf(stderr, "[ota_test] Memory allocation failed\n");
        goto _exit;
    }

    // 4. Read from file and update via the OTA instance
    printf("[ota_test] Writing data...\n");
    while ((rd = read(fd_img, buf, READ_CHUNK_SIZE)) > 0) {
        if (k230_ota_update(ota_ctx, buf, rd) < 0) {
            fprintf(stderr, "[ota_test] OTA update failed\n");
            goto _exit;
        }
    }

    if (rd < 0) {
        perror("[ota_test] Error reading image file");
        goto _exit;
    }

    printf("[ota_test] OTA update successful!\n");
    ret = 0;

_exit:
    // 5. Clean up - k230_ota_destroy handles closing the device and freeing the struct
    if (ota_ctx)
        k230_ota_destroy(ota_ctx);

    if (fd_img >= 0)
        close(fd_img);

    if (buf)
        free(buf);

    printf("[ota_test] Exit, ret=%d\n", ret);
    return ret;
}
