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

static const struct hmac_test_pattern {
    uint8_t hash;
    uint32_t keybits;
    const void* key;
    uint32_t msglen;
    const void* msg;
    const void* md;
} hmac_tp[] = {
    {
        HASH_SHA_224,
        32,
        "Jefe",
        28,
        "what do ya want for nothing?",
        "\xa3\x0e\x01\x09\x8b\xc6\xdb\xbf\x45\x69\x0f\x3a\x7e\x9e\x6d\x0f\x8b\xbe\xa2\xa3\x9e\x61\x48\x00\x8f\xd0\x5e\x44",
    },
    {
        HASH_SHA_256,
        32,
        "Jefe",
        28,
        "what do ya want for nothing?",
        "\x5b\xdc\xc1\x46\xbf\x60\x75\x4e\x6a\x04\x24\x26\x08\x95\x75\xc7\x5a\x00\x3f\x08\x9d\x27\x39\x83\x9d\xec\x58\xb9\x64\xec\x38\x43",
    },
    {
        HASH_SHA_384,
        32,
        "Jefe",
        28,
        "what do ya want for nothing?",
        "\xaf\x45\xd2\xe3\x76\x48\x40\x31\x61\x7f\x78\xd2\xb5\x8a\x6b\x1b\x9c\x7e\xf4\x64\xf5\xa0\x1b\x47\xe4\x2e\xc3\x73\x63\x22\x44\x5e\x8e\x22\x40\xca\x5e\x69\xe2\xc7\x8b\x32\x39\xec\xfa\xb2\x16\x49",
    },
    {
        HASH_SHA_512,
        32,
        "Jefe",
        28,
        "what do ya want for nothing?",
        "\x16\x4b\x7a\x7b\xfc\xf8\x19\xe2\xe3\x95\xfb\xe7\x3b\x56\xe0\xa3\x87\xbd\x64\x22\x2e\x83\x1f\xd6\x10\x27\x0c\xd7\xea\x25\x05\x54\x97\x58\xbf\x75\xc0\x5a\x99\x4a\x6d\x03\x4f\x65\xf8\xf0\xe6\xfd\xca\xea\xb1\xa3\x4d\x4a\x6b\x4b\x63\x6e\x07\x0a\x38\xbc\xe7\x37",
    },
};

static const struct cmac_test_pattern {
    uint8_t cipher;
    uint32_t keybits;
    const void* key;
    uint32_t msglen;
    const void* msg;
    const void* md;
} cmac_tp[] = {
    { SK_AES,
        128, "\x2B\x7E\x15\x16\x28\xAE\xD2\xA6\xAB\xF7\x15\x88\x09\xCF\x4F\x3C",
        0, NULL,
        "\xBB\x1D\x69\x29\xE9\x59\x37\x28\x7F\xA3\x7D\x12\x9B\x75\x67\x46" },
    { SK_AES,
        128, "\x2B\x7E\x15\x16\x28\xAE\xD2\xA6\xAB\xF7\x15\x88\x09\xCF\x4F\x3C",
        64, "\x6B\xC1\xBE\xE2\x2E\x40\x9F\x96\xE9\x3D\x7E\x11\x73\x93\x17\x2A\xAE\x2D\x8A\x57\x1E\x03\xAC\x9C\x9E\xB7\x6F\xAC\x45\xAF\x8E\x51\x30\xC8\x1C\x46\xA3\x5C\xE4\x11\xE5\xFB\xC1\x19\x1A\x0A\x52\xEF\xF6\x9F\x24\x45\xDF\x4F\x9B\x17\xAD\x2B\x41\x7B\xE6\x6C\x37\x10",
        "\x51\xF0\xBE\xBF\x7E\x3B\x9D\x92\xFC\x49\x74\x17\x79\x36\x3C\xFE" },
    { SK_AES,
        192, "\x8E\x73\xB0\xF7\xDA\x0E\x64\x52\xC8\x10\xF3\x2B\x80\x90\x79\xE5\x62\xF8\xEA\xD2\x52\x2C\x6B\x7B",
        0, NULL,
        "\xD1\x7D\xDF\x46\xAD\xAA\xCD\xE5\x31\xCA\xC4\x83\xDE\x7A\x93\x67" },
    { SK_AES,
        192, "\x8E\x73\xB0\xF7\xDA\x0E\x64\x52\xC8\x10\xF3\x2B\x80\x90\x79\xE5\x62\xF8\xEA\xD2\x52\x2C\x6B\x7B",
        64, "\x6B\xC1\xBE\xE2\x2E\x40\x9F\x96\xE9\x3D\x7E\x11\x73\x93\x17\x2A\xAE\x2D\x8A\x57\x1E\x03\xAC\x9C\x9E\xB7\x6F\xAC\x45\xAF\x8E\x51\x30\xC8\x1C\x46\xA3\x5C\xE4\x11\xE5\xFB\xC1\x19\x1A\x0A\x52\xEF\xF6\x9F\x24\x45\xDF\x4F\x9B\x17\xAD\x2B\x41\x7B\xE6\x6C\x37\x10",
        "\xA1\xD5\xDF\x0E\xED\x79\x0F\x79\x4D\x77\x58\x96\x59\xF3\x9A\x11" },
    { SK_AES,
        256, "\x60\x3D\xEB\x10\x15\xCA\x71\xBE\x2B\x73\xAE\xF0\x85\x7D\x77\x81\x1F\x35\x2C\x07\x3B\x61\x08\xD7\x2D\x98\x10\xA3\x09\x14\xDF\xF4",
        0, NULL,
        "\x02\x89\x62\xF6\x1B\x7B\xF8\x9E\xFC\x6B\x55\x1F\x46\x67\xD9\x83" },
    { SK_AES,
        256, "\x60\x3D\xEB\x10\x15\xCA\x71\xBE\x2B\x73\xAE\xF0\x85\x7D\x77\x81\x1F\x35\x2C\x07\x3B\x61\x08\xD7\x2D\x98\x10\xA3\x09\x14\xDF\xF4",
        64, "\x6B\xC1\xBE\xE2\x2E\x40\x9F\x96\xE9\x3D\x7E\x11\x73\x93\x17\x2A\xAE\x2D\x8A\x57\x1E\x03\xAC\x9C\x9E\xB7\x6F\xAC\x45\xAF\x8E\x51\x30\xC8\x1C\x46\xA3\x5C\xE4\x11\xE5\xFB\xC1\x19\x1A\x0A\x52\xEF\xF6\x9F\x24\x45\xDF\x4F\x9B\x17\xAD\x2B\x41\x7B\xE6\x6C\x37\x10",
        "\xE1\x99\x21\x90\x54\x9F\x6E\xD5\x69\x6A\x2C\x05\x6C\x31\x54\x10" },
};

int hmac_test(void)
{
    int ret;
    int fd;
    int ntests = sizeof(hmac_tp) / sizeof(hmac_tp[0]);
    uint8_t out[64];
    uint32_t outlen;
    pufs_mac_init_t init;
    pufs_mac_update_t update;
    pufs_mac_final_t final;

    printf("##########################################################\n");
    printf("HMAC TEST DEMO\n");
    printf("##########################################################\n");

    fd = open("/dev/pufs", O_RDWR);
    if (fd < 0) {
        printf("open /dev/pufs err!\n");
        return -1;
    }

    for (int i = 0; i < ntests; i++) {
        printf("case %d, msglen = %d\n", i, hmac_tp[i].msglen);
        memset(out, 0, sizeof(out));
        // init
        init.cipher = MAC_HMAC;
        init.mode = hmac_tp[i].hash;
        init.keytype = KT_SWKEY;
        init.keyaddr = (void*)hmac_tp[i].key;
        init.keybits = hmac_tp[i].keybits;
        ret = ioctl(fd, PUFS_MAC_INIT, &init);
        if (ret) {
            printf("ioctl PUFS_MAC_INIT err!\n");
            break;
        }
        // update
        update.msg = (void*)hmac_tp[i].msg;
        update.msglen = hmac_tp[i].msglen;
        ret = ioctl(fd, PUFS_MAC_UPDATE, &update);
        if (ret) {
            printf("ioctl PUFS_MAC_UPDATE err!\n");
            break;
        }
        // finish
        final.dgst = out;
        final.dlen = &outlen;
        ret = ioctl(fd, PUFS_MAC_FINAL, &final);
        if (ret) {
            printf("ioctl PUFS_MAC_FINAL err!\n");
            break;
        }

        printf("hashlen = %d\n", outlen);
        ret = memcmp(out, hmac_tp[i].md, outlen);
        if (ret) {
            printf("compare err!\n");
            break;
        }
    }

    if (ret == 0)
        printf("Success!\n");
    else
        printf("Fail!\n");

    close(fd);

    return ret;
}

int cmac_test(void)
{
    int ret;
    int fd;
    int ntests = sizeof(cmac_tp) / sizeof(cmac_tp[0]);
    uint8_t out[64];
    uint32_t outlen;
    pufs_mac_init_t init;
    pufs_mac_update_t update;
    pufs_mac_final_t final;

    printf("##########################################################\n");
    printf("CMAC TEST DEMO\n");
    printf("##########################################################\n");

    fd = open("/dev/pufs", O_RDWR);
    if (fd < 0) {
        printf("open /dev/pufs err!\n");
        return -1;
    }

    for (int i = 0; i < ntests; i++) {
        printf("case %d, msglen = %d\n", i, cmac_tp[i].msglen);
        memset(out, 0, sizeof(out));
        // init
        init.cipher = MAC_CMAC;
        init.mode = cmac_tp[i].cipher;
        init.keytype = KT_SWKEY;
        init.keyaddr = (void*)cmac_tp[i].key;
        init.keybits = cmac_tp[i].keybits;
        ret = ioctl(fd, PUFS_MAC_INIT, &init);
        if (ret) {
            printf("ioctl PUFS_MAC_INIT err!\n");
            break;
        }
        // update
        update.msg = (void*)cmac_tp[i].msg;
        update.msglen = cmac_tp[i].msglen;
        ret = ioctl(fd, PUFS_MAC_UPDATE, &update);
        if (ret) {
            printf("ioctl PUFS_MAC_UPDATE err!\n");
            break;
        }
        // finish
        final.dgst = out;
        final.dlen = &outlen;
        ret = ioctl(fd, PUFS_MAC_FINAL, &final);
        if (ret) {
            printf("ioctl PUFS_MAC_FINAL err!\n");
            break;
        }

        printf("hashlen = %d\n", outlen);
        ret = memcmp(out, cmac_tp[i].md, outlen);
        if (ret) {
            printf("compare err!\n");
            break;
        }
    }

    if (ret == 0)
        printf("Success!\n");
    else
        printf("Fail!\n");

    close(fd);

    return ret;
}
