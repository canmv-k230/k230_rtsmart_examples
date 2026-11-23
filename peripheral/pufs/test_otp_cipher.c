/* SPDX-License-Identifier: BSD-2-Clause */

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "drv_pufs.h"

#define FIXTURE_SLOT OTPKEY_10
#define OTP_KEY_ADDR(slot) (((slot) - OTPKEY_0) * 32u)

/* Public test key provisioned in OTPKEY_10 on the k230-2 lab board. */
static const uint8_t fixture[] = {
    0x2f, 0xcc, 0xb8, 0x27, 0x7d, 0xfb, 0xbb, 0xb1,
    0x5b, 0xd2, 0x6d, 0xdc, 0xdd, 0xa2, 0x2d, 0x59,
    0x10, 0x15, 0x74, 0xf7, 0xbc, 0x58, 0xee, 0x30,
    0xc2, 0xc7, 0x64, 0xc4, 0x26, 0x08, 0xb1, 0x03,
};

/* AES-256-XTS ciphertexts from OpenSSL, with the fixture as key 1 or key 2. */
static const uint8_t expected_xts[2][64] = {
    {
        0x8b, 0x62, 0x25, 0x24, 0x2b, 0xf5, 0x89, 0x2e,
        0xcb, 0xae, 0x3f, 0xe9, 0x90, 0xc1, 0x8a, 0x51,
        0x01, 0x6f, 0x2f, 0x12, 0x8c, 0xee, 0xb6, 0x0e,
        0xe3, 0xf0, 0xbb, 0x8c, 0xcd, 0x0b, 0xc3, 0xdc,
        0x06, 0xc5, 0x9a, 0x56, 0x68, 0x19, 0x66, 0xba,
        0x24, 0x8b, 0x3f, 0x51, 0x8c, 0xbf, 0xa8, 0xe5,
        0x5e, 0xac, 0x9a, 0x80, 0xf4, 0x17, 0xba, 0x8e,
        0xf4, 0x31, 0xe3, 0xe3, 0x11, 0x8c, 0x2a, 0xa1,
    },
    {
        0x2f, 0xa0, 0x57, 0xf6, 0x92, 0x4b, 0x9d, 0x26,
        0xa1, 0x2c, 0x9f, 0x05, 0xa4, 0xd8, 0xfb, 0xf3,
        0x31, 0x2d, 0x15, 0x48, 0xa1, 0xe2, 0xb9, 0x93,
        0x3f, 0x66, 0x3f, 0x40, 0xb6, 0x92, 0xe0, 0x12,
        0x13, 0x98, 0x60, 0x8b, 0x54, 0xf4, 0xb0, 0xd6,
        0x2a, 0xd4, 0xf7, 0xc3, 0x2e, 0x87, 0xa1, 0xd6,
        0x69, 0x0c, 0x68, 0x69, 0x01, 0xf4, 0x2a, 0xb5,
        0x9b, 0xde, 0x35, 0x9a, 0xf3, 0xe3, 0x5e, 0xb4,
    },
};

struct cipher_case {
    const char *name;
    pufs_skcipher_mode_t mode;
    bool gcm_compound;
    unsigned xts_slot; /* Which of the two XTS keys uses the fixture. */
};

static bool is_aead(const struct cipher_case *tc)
{
    return tc->mode == MODE_GCM || tc->mode == MODE_CCM;
}

static int crypt(drv_pufs_inst *dev, const struct cipher_case *tc, bool otp,
                 bool encrypt, const uint8_t *in, uint8_t *out, uint32_t len,
                 uint8_t tag[16])
{
    const uint8_t other_key[32] = {0xa5, 0x19};
    const uint8_t *key = otp ? PUFS_CIPHER_SLOT(FIXTURE_SLOT) : fixture;
    pufs_keytype_t kt = otp ? KT_OTPKEY : KT_SWKEY;
    uint8_t iv[16] = {0x42}, aad[20] = {0x81};
    uint32_t n = 0, tail = 0;
    uint32_t taglen = is_aead(tc) ? 16 : 0;
    drv_pufs_cipher_inst c;
    int ret;

    if (tc->mode == MODE_XTS) {
        bool first = tc->xts_slot == 1;
        ret = drv_pufs_cipher_xts_init(&c, dev, SK_AES, encrypt,
                                      first ? kt : KT_SWKEY, first ? key : other_key,
                                      first ? KT_SWKEY : kt, first ? other_key : key,
                                      sizeof(fixture) * 8, iv, sizeof(iv));
    } else if (tc->mode == MODE_CCM) {
        ret = drv_pufs_cipher_ccm_init(&c, dev, SK_AES, encrypt, kt, key, 256,
                                      iv, 12, sizeof(aad), len, taglen);
    } else if (tc->gcm_compound) {
        ret = drv_pufs_cipher_init_update(&c, dev, SK_AES, MODE_GCM, encrypt,
                                         kt, key, 256, iv, 12, NULL, NULL,
                                         aad, sizeof(aad));
    } else {
        const uint8_t *mode_iv = tc->mode == MODE_ECB ? NULL : iv;
        uint32_t ivlen = tc->mode == MODE_ECB ? 0 : (is_aead(tc) ? 12 : sizeof(iv));
        ret = drv_pufs_cipher_init(&c, dev, SK_AES, tc->mode, encrypt,
                                   kt, key, 256, mode_iv, ivlen);
    }

    if (!ret && taglen && !tc->gcm_compound)
        ret = drv_pufs_cipher_update(&c, NULL, NULL, aad, sizeof(aad));
    if (!ret)
        ret = drv_pufs_cipher_update(&c, out, &n, in, len);
    if (!ret && n <= len)
        ret = drv_pufs_cipher_final(&c, out + n, &tail, taglen ? tag : NULL, taglen);
    return ret ? ret : (n + tail == len ? 0 : -1);
}

int test_otp_cipher_fixture(void)
{
    const struct cipher_case cases[] = {
        {"ECB", MODE_ECB, false, 0}, {"CBC", MODE_CBC, false, 0},
        {"CFB", MODE_CFB, false, 0}, {"OFB", MODE_OFB, false, 0},
        {"CTR", MODE_CTR, false, 0}, {"GCM", MODE_GCM, false, 0},
        {"GCM compound init", MODE_GCM, true, 0}, {"CCM", MODE_CCM, false, 0},
        {"XTS data key", MODE_XTS, false, 1}, {"XTS tweak key", MODE_XTS, false, 2},
    };
    drv_pufs_inst dev;
    uint8_t actual[sizeof(fixture)], plain[64];
    unsigned passed = 0, failed = 0;

    if (drv_pufs_open(&dev) != 0)
        return 1;
    if (drv_pufs_otp_read(&dev, OTP_KEY_ADDR(FIXTURE_SLOT), actual, sizeof(actual)) != 0 ||
        memcmp(actual, fixture, sizeof(fixture)) != 0) {
        printf("OTP slot 10 does not contain the public lab fixture\n");
        drv_pufs_close(&dev);
        return 1;
    }
    for (size_t i = 0; i < sizeof(plain); i++)
        plain[i] = (uint8_t)(i + 0x10);

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const struct cipher_case *tc = &cases[i];
        uint32_t len = is_aead(tc) ? 48 : sizeof(plain);
        uint8_t sw[80], otp[80], recovered[80], sw_tag[16] = {0}, otp_tag[16] = {0};
        int s = crypt(&dev, tc, false, true, plain, sw, len, sw_tag);
        int o = crypt(&dev, tc, true, true, plain, otp, len, otp_tag);
        int d = o ? o : crypt(&dev, tc, true, false, otp, recovered, len, otp_tag);
        bool match = s == 0 && o == 0 && d == 0 &&
                     memcmp(sw, otp, len) == 0 &&
                     memcmp(sw_tag, otp_tag, sizeof(sw_tag)) == 0 &&
                     memcmp(plain, recovered, len) == 0;
        if (tc->xts_slot != 0)
            match = match && memcmp(sw, expected_xts[tc->xts_slot - 1], len) == 0;
        printf("[%s] %s: sw=%d otp=%d decrypt=%d\n", match ? "PASS" : "FAIL", tc->name, s, o, d);
        passed += match;
        failed += !match;
    }
    drv_pufs_close(&dev);
    printf("OTP_CIPHER_SUMMARY: %u passed, %u failed\n", passed, failed);
    return failed != 0;
}
