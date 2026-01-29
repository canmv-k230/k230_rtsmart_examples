/* Copyright (c) 2023, Canaan Bright Sight Co., Ltd
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "sample_pufs.h"

int key_inout_test(void)
{
    int ret;
    int fd;
    pufs_key_io_t ctl;

    printf("##########################################################\n");
    printf("KEY_INOUT TEST DEMO\n");
    printf("##########################################################\n");

    fd = open("/dev/pufs", O_RDWR);
    if (fd < 0) {
        printf("open /dev/pufs err!\n");
        return -1;
    }

    for (int i = KS_SK128_0; i <= KS_SK128_7; i++) {
        uint8_t key[16];
        memset(key, 0x11 * i, sizeof(key));
        ctl.mode = KM_IMPORT_PT;
        ctl.keytype = KT_SSKEY;
        ctl.keyslot = i;
        ctl.keyaddr = (void*)key;
        ctl.keybits = 128;
        ret = ioctl(fd, PUFS_KEY_INOUT, &ctl);
        if (ret) {
            printf("ioctl PUFS_KEY_INOUT err!\n");
            break;
        }
    }

    if (ret == 0)
        printf("Success!\n");
    else
        printf("Fail!\n");

    for (int i = KS_SK128_0; i <= KS_SK128_7; i++) {
        ctl.mode = KM_CLEAR;
        ctl.keytype = KT_SSKEY;
        ctl.keyslot = i;
        ctl.keybits = 128;
        ioctl(fd, PUFS_KEY_INOUT, &ctl);
    }

    close(fd);

    return ret;
}
