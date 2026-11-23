/* SPDX-License-Identifier: BSD-2-Clause */

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "drv_pufs.h"

/* RFC 4231 sections 4.7 and 4.8. Both use 131 bytes of 0xaa as the key. */
static const uint8_t tc6_msg[] = "Test Using Larger Than Block-Size Key - Hash Key First";
static const uint8_t tc7_msg[] = "This is a test using a larger than block-size key and a larger "
                                "than block-size data. The key needs to be hashed before being "
                                "used by the HMAC algorithm.";
static const uint8_t tc6_sha256[] = {
    0x60, 0xe4, 0x31, 0x59, 0x1e, 0xe0, 0xb6, 0x7f,
    0x0d, 0x8a, 0x26, 0xaa, 0xcb, 0xf5, 0xb7, 0x7f,
    0x8e, 0x0b, 0xc6, 0x21, 0x37, 0x28, 0xc5, 0x14,
    0x05, 0x46, 0x04, 0x0f, 0x0e, 0xe3, 0x7f, 0x54,
};
static const uint8_t tc7_sha512[] = {
    0xe3, 0x7b, 0x6a, 0x77, 0x5d, 0xc8, 0x7d, 0xba,
    0xa4, 0xdf, 0xa9, 0xf9, 0x6e, 0x5e, 0x3f, 0xfd,
    0xde, 0xbd, 0x71, 0xf8, 0x86, 0x72, 0x89, 0x86,
    0x5d, 0xf5, 0xa3, 0x2d, 0x20, 0xcd, 0xc9, 0x44,
    0xb6, 0x02, 0x2c, 0xac, 0x3c, 0x49, 0x82, 0xb1,
    0x0d, 0x5e, 0xeb, 0x55, 0xc3, 0xe4, 0xde, 0x15,
    0x13, 0x46, 0x76, 0xfb, 0x6d, 0xe0, 0x44, 0x60,
    0x65, 0xc9, 0x74, 0x40, 0xfa, 0x8c, 0x6a, 0x58,
};

int test_hmac_long_keys(void)
{
    static const struct {
        pufs_hashtype_t hash;
        const uint8_t *msg, *mac;
        uint32_t msglen, maclen;
        unsigned testcase;
    } cases[] = {
        { HASH_SHA_256, tc6_msg, tc6_sha256, sizeof(tc6_msg) - 1, sizeof(tc6_sha256), 6 },
        { HASH_SHA_512, tc7_msg, tc7_sha512, sizeof(tc7_msg) - 1, sizeof(tc7_sha512), 7 },
    };
    static const pufs_hashtype_t wide_hashes[] = {
        HASH_SHA_384, HASH_SHA_512, HASH_SHA_512_224, HASH_SHA_512_256,
    };
    drv_pufs_inst dev;
    drv_pufs_hash_inst hmac;
    uint8_t key[131], mac[PUFS_DGST_MAX_LEN];
    unsigned failures = 0;

    if (drv_pufs_open(&dev) != 0) {
        puts("[FAIL] open /dev/pufs for long-key HMAC");
        return -1;
    }
    memset(key, 0xaa, sizeof(key));
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        uint32_t len = 0;
        int ret = drv_pufs_hmac_init(&hmac, &dev, cases[i].hash, KT_SWKEY, key, sizeof(key) * 8);
        if (ret == 0)
            ret = drv_pufs_hmac_update(&hmac, cases[i].msg, cases[i].msglen);
        if (ret == 0)
            ret = drv_pufs_hmac_final(&hmac, mac, &len);
        bool ok = ret == 0 && len == cases[i].maclen && memcmp(mac, cases[i].mac, len) == 0;

        printf("[%s] RFC 4231 test case %u\n", ok ? "PASS" : "FAIL", cases[i].testcase);
        failures += !ok;
    }
    for (size_t i = 0; i < sizeof(wide_hashes) / sizeof(wide_hashes[0]); i++) {
        bool ok = drv_pufs_hmac_init(&hmac, &dev, wide_hashes[i], KT_SWKEY, key, 64 * 8) == 0;

        for (uint32_t len = 65; len <= 128; len++) {
            errno = 0;
            int ret = drv_pufs_hmac_init(&hmac, &dev, wide_hashes[i], KT_SWKEY, key, len * 8);
            ok &= ret == -1 && errno == ENOTSUP;
        }
        ok &= drv_pufs_hmac_init(&hmac, &dev, wide_hashes[i], KT_SWKEY, key, 129 * 8) == 0;
        printf("[%s] HMAC hash %d rejects 65-128 byte keys only\n", ok ? "PASS" : "FAIL", wide_hashes[i]);
        failures += !ok;
    }
    drv_pufs_close(&dev);
    return failures ? -1 : 0;
}
