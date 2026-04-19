/* Copyright (c) 2023, Canaan Bright Sight Co., Ltd
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <stdio.h>
#include <string.h>

#include "mbedtls/aes.h"
#include "mbedtls/bignum.h"
#include "mbedtls/ccm.h"
#include "mbedtls/cipher.h"
#include "mbedtls/cmac.h"
#include "mbedtls/ecp.h"
#include "mbedtls/ecdsa.h"
#include "mbedtls/ecdh.h"
#include "mbedtls/gcm.h"
#include "mbedtls/md.h"
#include "mbedtls/sha256.h"
#include "mbedtls/sha512.h"

static int test_passed;
static int test_failed;
static int test_skipped;

#ifndef MBEDTLS_PUFS_AES_HW_THRESHOLD
#define MBEDTLS_PUFS_AES_HW_THRESHOLD 256u
#endif

#define AES_HW_CBC_TEST_LEN \
    ((((size_t) MBEDTLS_PUFS_AES_HW_THRESHOLD) + 15u) / 16u * 16u)
#define AES_HW_CTR_TEST_LEN ((size_t) MBEDTLS_PUFS_AES_HW_THRESHOLD + 13u)

static const unsigned char aes128_key[16];

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("[FAIL] %s\n", msg); \
        test_failed++; \
        return -1; \
    } \
    printf("[PASS] %s\n", msg); \
    test_passed++; \
} while (0)

#define TEST_RET(ret, msg) do { \
    if ((ret) != 0) { \
        printf("[FAIL] %s (ret=%d)\n", msg, (ret)); \
        test_failed++; \
        return -1; \
    } \
    printf("[PASS] %s\n", msg); \
    test_passed++; \
} while (0)

#define TEST_EXPECT_RET(ret, expected, msg) do { \
    if ((ret) != (expected)) { \
        printf("[FAIL] %s (ret=%d, expected=%d)\n", msg, (ret), (expected)); \
        test_failed++; \
        return -1; \
    } \
    printf("[PASS] %s\n", msg); \
    test_passed++; \
} while (0)

static int buffer_is_zero(const unsigned char *buffer, size_t length)
{
    size_t index;

    for (index = 0; index < length; index++) {
        if (buffer[index] != 0)
            return 0;
    }

    return 1;
}

static int test_rng(void *context, unsigned char *output, size_t output_len)
{
    size_t index;

    (void) context;

    for (index = 0; index < output_len; index++)
        output[index] = (unsigned char) (0x5a + (index & 0x0f));

    return 0;
}

static void fill_test_pattern(unsigned char *buffer, size_t length,
                              unsigned char seed)
{
    size_t index;

    for (index = 0; index < length; index++)
        buffer[index] = (unsigned char) (seed + index * 13u);
}

static int aes_cbc_encrypt_chunked(const unsigned char *input,
                                   size_t length,
                                   const unsigned char iv_in[16],
                                   unsigned char *output)
{
    mbedtls_aes_context ctx;
    unsigned char iv[16];
    size_t offset = 0;
    int ret;

    mbedtls_aes_init(&ctx);
    ret = mbedtls_aes_setkey_enc(&ctx, aes128_key, 128);
    if (ret != 0)
        goto cleanup;

    memcpy(iv, iv_in, sizeof(iv));
    while (offset < length) {
        ret = mbedtls_aes_crypt_cbc(&ctx, MBEDTLS_AES_ENCRYPT, 16,
                                    iv, input + offset, output + offset);
        if (ret != 0)
            goto cleanup;
        offset += 16;
    }

cleanup:
    mbedtls_aes_free(&ctx);
    return ret;
}

static int aes_ctr_crypt_chunked(const unsigned char *input,
                                 size_t length,
                                 const unsigned char nonce_in[16],
                                 unsigned char *output,
                                 unsigned char nonce_out[16],
                                 unsigned char stream_block[16],
                                 size_t *nc_off)
{
    mbedtls_aes_context ctx;
    size_t offset = 0;
    int ret;

    mbedtls_aes_init(&ctx);
    ret = mbedtls_aes_setkey_enc(&ctx, aes128_key, 128);
    if (ret != 0)
        goto cleanup;

    memcpy(nonce_out, nonce_in, 16);
    memset(stream_block, 0, 16);
    *nc_off = 0;

    while (offset < length) {
        size_t step = length - offset;

        if (step > 19)
            step = 19;

        ret = mbedtls_aes_crypt_ctr(&ctx, step, nc_off,
                                    nonce_out, stream_block,
                                    input + offset, output + offset);
        if (ret != 0)
            goto cleanup;
        offset += step;
    }

cleanup:
    mbedtls_aes_free(&ctx);
    return ret;
}

static const unsigned char sha_msg[] = { 'a', 'b', 'c' };

static const unsigned char sha224_expected[28] = {
    0x23, 0x09, 0x7d, 0x22, 0x34, 0x05, 0xd8, 0x22,
    0x86, 0x42, 0xa4, 0x77, 0xbd, 0xa2, 0x55, 0xb3,
    0x2a, 0xad, 0xbc, 0xe4, 0xbd, 0xa0, 0xb3, 0xf7,
    0xe3, 0x6c, 0x9d, 0xa7,
};

static const unsigned char sha256_expected[32] = {
    0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
    0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
    0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
    0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
};

static const unsigned char sha384_expected[48] = {
    0xcb, 0x00, 0x75, 0x3f, 0x45, 0xa3, 0x5e, 0x8b,
    0xb5, 0xa0, 0x3d, 0x69, 0x9a, 0xc6, 0x50, 0x07,
    0x27, 0x2c, 0x32, 0xab, 0x0e, 0xde, 0xd1, 0x63,
    0x1a, 0x8b, 0x60, 0x5a, 0x43, 0xff, 0x5b, 0xed,
    0x80, 0x86, 0x07, 0x2b, 0xa1, 0xe7, 0xcc, 0x23,
    0x58, 0xba, 0xec, 0xa1, 0x34, 0xc8, 0x25, 0xa7,
};

static const unsigned char sha512_expected[64] = {
    0xdd, 0xaf, 0x35, 0xa1, 0x93, 0x61, 0x7a, 0xba,
    0xcc, 0x41, 0x73, 0x49, 0xae, 0x20, 0x41, 0x31,
    0x12, 0xe6, 0xfa, 0x4e, 0x89, 0xa9, 0x7e, 0xa2,
    0x0a, 0x9e, 0xee, 0xe6, 0x4b, 0x55, 0xd3, 0x9a,
    0x21, 0x92, 0x99, 0x2a, 0x27, 0x4f, 0xc1, 0xa8,
    0x36, 0xba, 0x3c, 0x23, 0xa3, 0xfe, 0xeb, 0xbd,
    0x45, 0x4d, 0x44, 0x23, 0x64, 0x3c, 0xe8, 0x0e,
    0x2a, 0x9a, 0xc9, 0x4f, 0xa5, 0x4c, 0xa4, 0x9f,
};

static const unsigned char aes128_key[16] = {
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
};

static const unsigned char aes_ecb_expected[16] = {
    0x3a, 0xd7, 0x7b, 0xb4, 0x0d, 0x7a, 0x36, 0x60,
    0xa8, 0x9e, 0xca, 0xf3, 0x24, 0x66, 0xef, 0x97,
};

static const unsigned char aes_cbc_iv[16] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
};

static const unsigned char aes_ctr_nonce_counter[16] = {
    0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
    0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
};

static const unsigned char aes_sp38a_plaintext[64] = {
    0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
    0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
    0xae, 0x2d, 0x8a, 0x57, 0x1e, 0x03, 0xac, 0x9c,
    0x9e, 0xb7, 0x6f, 0xac, 0x45, 0xaf, 0x8e, 0x51,
    0x30, 0xc8, 0x1c, 0x46, 0xa3, 0x5c, 0xe4, 0x11,
    0xe5, 0xfb, 0xc1, 0x19, 0x1a, 0x0a, 0x52, 0xef,
    0xf6, 0x9f, 0x24, 0x45, 0xdf, 0x4f, 0x9b, 0x17,
    0xad, 0x2b, 0x41, 0x7b, 0xe6, 0x6c, 0x37, 0x10,
};

static const unsigned char aes_cbc_expected[64] = {
    0x76, 0x49, 0xab, 0xac, 0x81, 0x19, 0xb2, 0x46,
    0xce, 0xe9, 0x8e, 0x9b, 0x12, 0xe9, 0x19, 0x7d,
    0x50, 0x86, 0xcb, 0x9b, 0x50, 0x72, 0x19, 0xee,
    0x95, 0xdb, 0x11, 0x3a, 0x91, 0x76, 0x78, 0xb2,
    0x73, 0xbe, 0xd6, 0xb8, 0xe3, 0xc1, 0x74, 0x3b,
    0x71, 0x16, 0xe6, 0x9e, 0x22, 0x22, 0x95, 0x16,
    0x3f, 0xf1, 0xca, 0xa1, 0x68, 0x1f, 0xac, 0x09,
    0x12, 0x0e, 0xca, 0x30, 0x75, 0x86, 0xe1, 0xa7,
};

static const unsigned char aes_ctr_expected[64] = {
    0x87, 0x4d, 0x61, 0x91, 0xb6, 0x20, 0xe3, 0x26,
    0x1b, 0xef, 0x68, 0x64, 0x99, 0x0d, 0xb6, 0xce,
    0x98, 0x06, 0xf6, 0x6b, 0x79, 0x70, 0xfd, 0xff,
    0x86, 0x17, 0x18, 0x7b, 0xb9, 0xff, 0xfd, 0xff,
    0x5a, 0xe4, 0xdf, 0x3e, 0xdb, 0xd5, 0xd3, 0x5e,
    0x5b, 0x4f, 0x09, 0x02, 0x0d, 0xb0, 0x3e, 0xab,
    0x1e, 0x03, 0x1d, 0xda, 0x2f, 0xbe, 0x03, 0xd1,
    0x79, 0x21, 0x70, 0xa0, 0xf3, 0x00, 0x9c, 0xee,
};

static const unsigned char cmac_msg16[16] = {
    0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
    0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
};

static const unsigned char cmac_expected[16] = {
    0x07, 0x0a, 0x16, 0xb4, 0x6b, 0x4d, 0x41, 0x44,
    0xf7, 0x9b, 0xdd, 0x9d, 0xd0, 0x4a, 0x28, 0x7c,
};

static const unsigned char gcm_iv[12] = {
    0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
};

static const unsigned char gcm_aad[20] = {
    0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9,
    0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf, 0xb0, 0xb1, 0xb2, 0xb3,
};

static const unsigned char gcm_plaintext[32] = {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
    0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
};

static const unsigned char ccm_nonce[7] = {
    0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab,
};

static const unsigned char ccm_aad[12] = {
    0x31, 0x32, 0x33, 0x34, 0x35, 0x36,
    0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c,
};

static const unsigned char ccm_plaintext[24] = {
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
};

static const char ecdsa_test_d[] = "1";
static const char ecdsa_test_qx[] =
    "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296";
static const char ecdsa_test_qy[] =
    "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5";

static int test_sha2(void)
{
    unsigned char digest224[28];
    unsigned char digest256[32];
    unsigned char digest384[48];
    unsigned char digest512[64];
    int ret;

    printf("\n[Test] SHA-2\n");

    ret = mbedtls_sha256(sha_msg, sizeof(sha_msg), digest224, 1);
    TEST_RET(ret, "mbedtls_sha224");
    TEST_ASSERT(memcmp(digest224, sha224_expected, sizeof(digest224)) == 0,
                "SHA-224 known-answer vector");

    ret = mbedtls_sha256(sha_msg, sizeof(sha_msg), digest256, 0);
    TEST_RET(ret, "mbedtls_sha256");
    TEST_ASSERT(memcmp(digest256, sha256_expected, sizeof(digest256)) == 0,
                "SHA-256 known-answer vector");

    ret = mbedtls_sha512(sha_msg, sizeof(sha_msg), digest384, 1);
    TEST_RET(ret, "mbedtls_sha384");
    TEST_ASSERT(memcmp(digest384, sha384_expected, sizeof(digest384)) == 0,
                "SHA-384 known-answer vector");

    ret = mbedtls_sha512(sha_msg, sizeof(sha_msg), digest512, 0);
    TEST_RET(ret, "mbedtls_sha512");
    TEST_ASSERT(memcmp(digest512, sha512_expected, sizeof(digest512)) == 0,
                "SHA-512 known-answer vector");

    return 0;
}

static int test_aes(void)
{
    mbedtls_aes_context ctx;
    unsigned char output[64];
    unsigned char decrypted[64];
    unsigned char iv[16];
    unsigned char nonce_counter[16];
    unsigned char stream_block[16];
    unsigned char hw_cbc_output[AES_HW_CBC_TEST_LEN];
    unsigned char sw_cbc_output[AES_HW_CBC_TEST_LEN];
    unsigned char hw_cbc_plain[AES_HW_CBC_TEST_LEN];
    unsigned char large_cbc_plain[AES_HW_CBC_TEST_LEN];
    unsigned char hw_ctr_output[AES_HW_CTR_TEST_LEN];
    unsigned char sw_ctr_output[AES_HW_CTR_TEST_LEN];
    unsigned char hw_ctr_plain[AES_HW_CTR_TEST_LEN];
    unsigned char large_ctr_plain[AES_HW_CTR_TEST_LEN];
    unsigned char hw_nonce_after[16];
    unsigned char sw_nonce_after[16];
    unsigned char hw_stream_after[16];
    unsigned char sw_stream_after[16];
    size_t nc_off;
    size_t hw_nc_off;
    size_t sw_nc_off;
    int ret;

    printf("\n[Test] AES\n");

    mbedtls_aes_init(&ctx);

    ret = mbedtls_aes_setkey_enc(&ctx, aes128_key, 128);
    TEST_RET(ret, "mbedtls_aes_setkey_enc");

    ret = mbedtls_aes_crypt_ecb(&ctx, MBEDTLS_AES_ENCRYPT,
                                cmac_msg16, output);
    TEST_RET(ret, "mbedtls_aes_crypt_ecb(encrypt)");
    TEST_ASSERT(memcmp(output, aes_ecb_expected, sizeof(aes_ecb_expected)) == 0,
                "AES-128-ECB known-answer vector");

    memcpy(iv, aes_cbc_iv, sizeof(iv));
    ret = mbedtls_aes_crypt_cbc(&ctx, MBEDTLS_AES_ENCRYPT,
                                sizeof(aes_sp38a_plaintext),
                                iv, aes_sp38a_plaintext, output);
    TEST_RET(ret, "mbedtls_aes_crypt_cbc(encrypt)");
    TEST_ASSERT(memcmp(output, aes_cbc_expected, sizeof(aes_cbc_expected)) == 0,
                "AES-128-CBC known-answer vector");

    memcpy(nonce_counter, aes_ctr_nonce_counter, sizeof(nonce_counter));
    memset(stream_block, 0, sizeof(stream_block));
    nc_off = 0;
    ret = mbedtls_aes_crypt_ctr(&ctx, sizeof(aes_sp38a_plaintext),
                                &nc_off, nonce_counter, stream_block,
                                aes_sp38a_plaintext, output);
    TEST_RET(ret, "mbedtls_aes_crypt_ctr");
    TEST_ASSERT(memcmp(output, aes_ctr_expected, sizeof(aes_ctr_expected)) == 0,
                "AES-128-CTR known-answer vector");

    fill_test_pattern(large_cbc_plain, sizeof(large_cbc_plain), 0x31);
    ret = aes_cbc_encrypt_chunked(large_cbc_plain, sizeof(large_cbc_plain),
                                  aes_cbc_iv, sw_cbc_output);
    TEST_RET(ret, "AES-CBC chunked software reference");

    memcpy(iv, aes_cbc_iv, sizeof(iv));
    ret = mbedtls_aes_crypt_cbc(&ctx, MBEDTLS_AES_ENCRYPT,
                                sizeof(large_cbc_plain),
                                iv, large_cbc_plain, hw_cbc_output);
    TEST_RET(ret, "AES-CBC bulk encrypt");
    TEST_ASSERT(memcmp(hw_cbc_output, sw_cbc_output, sizeof(hw_cbc_output)) == 0,
                "AES-CBC bulk path matches software reference");

    ret = mbedtls_aes_setkey_dec(&ctx, aes128_key, 128);
    TEST_RET(ret, "mbedtls_aes_setkey_dec");

    ret = mbedtls_aes_crypt_ecb(&ctx, MBEDTLS_AES_DECRYPT,
                                aes_ecb_expected, output);
    TEST_RET(ret, "mbedtls_aes_crypt_ecb(decrypt)");
    TEST_ASSERT(memcmp(output, cmac_msg16, sizeof(cmac_msg16)) == 0,
                "AES-128-ECB decrypt round-trip");

    memcpy(iv, aes_cbc_iv, sizeof(iv));
    ret = mbedtls_aes_crypt_cbc(&ctx, MBEDTLS_AES_DECRYPT,
                                sizeof(aes_cbc_expected),
                                iv, aes_cbc_expected, decrypted);
    TEST_RET(ret, "mbedtls_aes_crypt_cbc(decrypt)");
    TEST_ASSERT(memcmp(decrypted, aes_sp38a_plaintext, sizeof(decrypted)) == 0,
                "AES-128-CBC decrypt round-trip");

    memcpy(iv, aes_cbc_iv, sizeof(iv));
    ret = mbedtls_aes_crypt_cbc(&ctx, MBEDTLS_AES_DECRYPT,
                                sizeof(hw_cbc_output),
                                iv, hw_cbc_output, hw_cbc_plain);
    TEST_RET(ret, "AES-CBC bulk decrypt");
    TEST_ASSERT(memcmp(hw_cbc_plain, large_cbc_plain, sizeof(hw_cbc_plain)) == 0,
                "AES-CBC bulk decrypt round-trip");

    ret = mbedtls_aes_setkey_enc(&ctx, aes128_key, 128);
    TEST_RET(ret, "mbedtls_aes_setkey_enc(reload)");

    fill_test_pattern(large_ctr_plain, sizeof(large_ctr_plain), 0x57);
    ret = aes_ctr_crypt_chunked(large_ctr_plain, sizeof(large_ctr_plain),
                                aes_ctr_nonce_counter, sw_ctr_output,
                                sw_nonce_after, sw_stream_after, &sw_nc_off);
    TEST_RET(ret, "AES-CTR chunked software reference");

    memcpy(hw_nonce_after, aes_ctr_nonce_counter, sizeof(hw_nonce_after));
    memset(hw_stream_after, 0, sizeof(hw_stream_after));
    hw_nc_off = 0;
    ret = mbedtls_aes_crypt_ctr(&ctx, sizeof(large_ctr_plain), &hw_nc_off,
                                hw_nonce_after, hw_stream_after,
                                large_ctr_plain, hw_ctr_output);
    TEST_RET(ret, "AES-CTR bulk encrypt");
    TEST_ASSERT(memcmp(hw_ctr_output, sw_ctr_output, sizeof(hw_ctr_output)) == 0,
                "AES-CTR bulk path matches software reference");
    TEST_ASSERT(memcmp(hw_nonce_after, sw_nonce_after, sizeof(hw_nonce_after)) == 0,
                "AES-CTR updates nonce counter consistently");
    TEST_ASSERT(memcmp(hw_stream_after, sw_stream_after, sizeof(hw_stream_after)) == 0,
                "AES-CTR updates stream block consistently");
    TEST_ASSERT(hw_nc_off == sw_nc_off,
                "AES-CTR updates nc_off consistently");

    memcpy(nonce_counter, aes_ctr_nonce_counter, sizeof(nonce_counter));
    memset(stream_block, 0, sizeof(stream_block));
    nc_off = 0;
    ret = mbedtls_aes_crypt_ctr(&ctx, sizeof(hw_ctr_output), &nc_off,
                                nonce_counter, stream_block,
                                hw_ctr_output, hw_ctr_plain);
    TEST_RET(ret, "AES-CTR bulk decrypt");
    TEST_ASSERT(memcmp(hw_ctr_plain, large_ctr_plain, sizeof(hw_ctr_plain)) == 0,
                "AES-CTR bulk decrypt round-trip");

    mbedtls_aes_free(&ctx);
    return 0;
}

static int test_gcm(void)
{
    mbedtls_gcm_context ctx;
    unsigned char ciphertext[sizeof(gcm_plaintext)];
    unsigned char decrypted[sizeof(gcm_plaintext)];
    unsigned char bad_output[sizeof(gcm_plaintext)];
    unsigned char tag[16];
    unsigned char bad_tag[16];
    int ret;

    printf("\n[Test] AES-GCM\n");

    mbedtls_gcm_init(&ctx);

    ret = mbedtls_gcm_setkey(&ctx, MBEDTLS_CIPHER_ID_AES, aes128_key, 128);
    TEST_RET(ret, "mbedtls_gcm_setkey");

    ret = mbedtls_gcm_crypt_and_tag(&ctx, MBEDTLS_GCM_ENCRYPT,
                                    sizeof(gcm_plaintext),
                                    gcm_iv, sizeof(gcm_iv),
                                    gcm_aad, sizeof(gcm_aad),
                                    gcm_plaintext, ciphertext,
                                    sizeof(tag), tag);
    TEST_RET(ret, "mbedtls_gcm_crypt_and_tag");

    ret = mbedtls_gcm_auth_decrypt(&ctx, sizeof(gcm_plaintext),
                                   gcm_iv, sizeof(gcm_iv),
                                   gcm_aad, sizeof(gcm_aad),
                                   tag, sizeof(tag),
                                   ciphertext, decrypted);
    TEST_RET(ret, "mbedtls_gcm_auth_decrypt(valid tag)");
    TEST_ASSERT(memcmp(decrypted, gcm_plaintext, sizeof(decrypted)) == 0,
                "AES-GCM round-trip");

    memcpy(bad_tag, tag, sizeof(bad_tag));
    bad_tag[0] ^= 0x01;
    memset(bad_output, 0xa5, sizeof(bad_output));

    ret = mbedtls_gcm_auth_decrypt(&ctx, sizeof(gcm_plaintext),
                                   gcm_iv, sizeof(gcm_iv),
                                   gcm_aad, sizeof(gcm_aad),
                                   bad_tag, sizeof(bad_tag),
                                   ciphertext, bad_output);
    TEST_EXPECT_RET(ret, MBEDTLS_ERR_GCM_AUTH_FAILED,
                    "AES-GCM rejects bad tag");
    TEST_ASSERT(buffer_is_zero(bad_output, sizeof(bad_output)),
                "AES-GCM clears output on auth failure");

    mbedtls_gcm_free(&ctx);
    return 0;
}

static int test_ccm(void)
{
    mbedtls_ccm_context ctx;
    unsigned char ciphertext[sizeof(ccm_plaintext)];
    unsigned char decrypted[sizeof(ccm_plaintext)];
    unsigned char bad_output[sizeof(ccm_plaintext)];
    unsigned char tag[8];
    unsigned char bad_tag[8];
    int ret;

    printf("\n[Test] AES-CCM\n");

    mbedtls_ccm_init(&ctx);

    ret = mbedtls_ccm_setkey(&ctx, MBEDTLS_CIPHER_ID_AES, aes128_key, 128);
    TEST_RET(ret, "mbedtls_ccm_setkey");

    ret = mbedtls_ccm_encrypt_and_tag(&ctx, sizeof(ccm_plaintext),
                                      ccm_nonce, sizeof(ccm_nonce),
                                      ccm_aad, sizeof(ccm_aad),
                                      ccm_plaintext, ciphertext,
                                      tag, sizeof(tag));
    TEST_RET(ret, "mbedtls_ccm_encrypt_and_tag");

    ret = mbedtls_ccm_auth_decrypt(&ctx, sizeof(ccm_plaintext),
                                   ccm_nonce, sizeof(ccm_nonce),
                                   ccm_aad, sizeof(ccm_aad),
                                   ciphertext, decrypted,
                                   tag, sizeof(tag));
    TEST_RET(ret, "mbedtls_ccm_auth_decrypt(valid tag)");
    TEST_ASSERT(memcmp(decrypted, ccm_plaintext, sizeof(decrypted)) == 0,
                "AES-CCM round-trip");

    memcpy(bad_tag, tag, sizeof(bad_tag));
    bad_tag[0] ^= 0x01;
    memset(bad_output, 0x5a, sizeof(bad_output));

    ret = mbedtls_ccm_auth_decrypt(&ctx, sizeof(ccm_plaintext),
                                   ccm_nonce, sizeof(ccm_nonce),
                                   ccm_aad, sizeof(ccm_aad),
                                   ciphertext, bad_output,
                                   bad_tag, sizeof(bad_tag));
    TEST_EXPECT_RET(ret, MBEDTLS_ERR_CCM_AUTH_FAILED,
                    "AES-CCM rejects bad tag");
    TEST_ASSERT(buffer_is_zero(bad_output, sizeof(bad_output)),
                "AES-CCM clears output on auth failure");

    mbedtls_ccm_free(&ctx);
    return 0;
}

static int test_cmac(void)
{
    const mbedtls_cipher_info_t *cipher_info;
    unsigned char output[16];
    int ret;

    printf("\n[Test] AES-CMAC\n");

    cipher_info = mbedtls_cipher_info_from_type(MBEDTLS_CIPHER_AES_128_ECB);
    TEST_ASSERT(cipher_info != NULL, "Resolve AES-128-ECB cipher info");

    ret = mbedtls_cipher_cmac(cipher_info,
                              aes128_key, 128,
                              cmac_msg16, sizeof(cmac_msg16),
                              output);
    TEST_RET(ret, "mbedtls_cipher_cmac");
    TEST_ASSERT(memcmp(output, cmac_expected, sizeof(output)) == 0,
                "AES-CMAC RFC 4493 vector");

    return 0;
}

static int test_ecdsa_verify(void)
{
    mbedtls_ecp_group group;
    mbedtls_ecp_point public_key;
    mbedtls_mpi private_key;
    mbedtls_mpi sig_r;
    mbedtls_mpi sig_s;
    unsigned char hash[32];
    unsigned char bad_hash[32];
    int ret;

    printf("\n[Test] ECDSA Verify\n");

    mbedtls_ecp_group_init(&group);
    mbedtls_ecp_point_init(&public_key);
    mbedtls_mpi_init(&private_key);
    mbedtls_mpi_init(&sig_r);
    mbedtls_mpi_init(&sig_s);

    ret = mbedtls_sha256(sha_msg, sizeof(sha_msg), hash, 0);
    TEST_RET(ret, "Hash message for ECDSA");

    ret = mbedtls_ecp_group_load(&group, MBEDTLS_ECP_DP_SECP256R1);
    TEST_RET(ret, "Load secp256r1 group");

    ret = mbedtls_ecp_point_read_string(&public_key, 16,
                                        ecdsa_test_qx, ecdsa_test_qy);
    TEST_RET(ret, "Load secp256r1 public key");

    ret = mbedtls_mpi_read_string(&private_key, 16, ecdsa_test_d);
    TEST_RET(ret, "Load secp256r1 private key");

    ret = mbedtls_ecdsa_sign_det_ext(&group,
                                     &sig_r, &sig_s,
                                     &private_key,
                                     hash, sizeof(hash),
                                     MBEDTLS_MD_SHA256,
                                     test_rng, NULL);
    TEST_RET(ret, "Create deterministic ECDSA signature");

    ret = mbedtls_ecdsa_verify(&group,
                               hash, sizeof(hash),
                               &public_key,
                               &sig_r, &sig_s);
    TEST_RET(ret, "mbedtls_ecdsa_verify(valid signature)");

    memcpy(bad_hash, hash, sizeof(bad_hash));
    bad_hash[0] ^= 0x01;

    ret = mbedtls_ecdsa_verify(&group,
                               bad_hash, sizeof(bad_hash),
                               &public_key,
                               &sig_r, &sig_s);
    TEST_ASSERT(ret == MBEDTLS_ERR_ECP_VERIFY_FAILED,
                "ECDSA verify rejects bad hash");

    mbedtls_mpi_free(&sig_s);
    mbedtls_mpi_free(&sig_r);
    mbedtls_mpi_free(&private_key);
    mbedtls_ecp_point_free(&public_key);
    mbedtls_ecp_group_free(&group);
    return 0;
}

static int test_sha2_streaming(void)
{
    mbedtls_sha256_context ctx256;
    mbedtls_sha512_context ctx512;
    unsigned char digest256[32];
    unsigned char digest512[64];
    int ret;

    printf("\n[Test] SHA-2 Streaming (byte-at-a-time)\n");

    /* SHA-256 streaming */
    mbedtls_sha256_init(&ctx256);
    ret = mbedtls_sha256_starts(&ctx256, 0);
    TEST_RET(ret, "sha256_starts");

    for (size_t i = 0; i < sizeof(sha_msg); i++) {
        ret = mbedtls_sha256_update(&ctx256, &sha_msg[i], 1);
        if (ret != 0) break;
    }
    TEST_RET(ret, "sha256_update byte-at-a-time");

    ret = mbedtls_sha256_finish(&ctx256, digest256);
    TEST_RET(ret, "sha256_finish");
    TEST_ASSERT(memcmp(digest256, sha256_expected, 32) == 0,
                "SHA-256 streaming matches one-shot");
    mbedtls_sha256_free(&ctx256);

    /* SHA-512 streaming */
    mbedtls_sha512_init(&ctx512);
    ret = mbedtls_sha512_starts(&ctx512, 0);
    TEST_RET(ret, "sha512_starts");

    for (size_t i = 0; i < sizeof(sha_msg); i++) {
        ret = mbedtls_sha512_update(&ctx512, &sha_msg[i], 1);
        if (ret != 0) break;
    }
    TEST_RET(ret, "sha512_update byte-at-a-time");

    ret = mbedtls_sha512_finish(&ctx512, digest512);
    TEST_RET(ret, "sha512_finish");
    TEST_ASSERT(memcmp(digest512, sha512_expected, 64) == 0,
                "SHA-512 streaming matches one-shot");
    mbedtls_sha512_free(&ctx512);

    return 0;
}

static int test_sha2_empty(void)
{
    unsigned char digest256[32];
    unsigned char digest512[64];
    int ret;

    static const unsigned char sha256_empty[32] = {
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
        0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
        0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
        0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
    };

    static const unsigned char sha512_empty[64] = {
        0xcf, 0x83, 0xe1, 0x35, 0x7e, 0xef, 0xb8, 0xbd,
        0xf1, 0x54, 0x28, 0x50, 0xd6, 0x6d, 0x80, 0x07,
        0xd6, 0x20, 0xe4, 0x05, 0x0b, 0x57, 0x15, 0xdc,
        0x83, 0xf4, 0xa9, 0x21, 0xd3, 0x6c, 0xe9, 0xce,
        0x47, 0xd0, 0xd1, 0x3c, 0x5d, 0x85, 0xf2, 0xb0,
        0xff, 0x83, 0x18, 0xd2, 0x87, 0x7e, 0xec, 0x2f,
        0x63, 0xb9, 0x31, 0xbd, 0x47, 0x41, 0x7a, 0x81,
        0xa5, 0x38, 0x32, 0x7a, 0xf9, 0x27, 0xda, 0x3e,
    };

    printf("\n[Test] SHA-2 Empty Message\n");

    ret = mbedtls_sha256(NULL, 0, digest256, 0);
    TEST_RET(ret, "SHA-256 empty");
    TEST_ASSERT(memcmp(digest256, sha256_empty, 32) == 0,
                "SHA-256(\"\") known-answer vector");

    ret = mbedtls_sha512(NULL, 0, digest512, 0);
    TEST_RET(ret, "SHA-512 empty");
    TEST_ASSERT(memcmp(digest512, sha512_empty, 64) == 0,
                "SHA-512(\"\") known-answer vector");

    return 0;
}

static int test_sha2_large(void)
{
    mbedtls_sha256_context ctx;
    unsigned char digest1[32], digest2[32];
    unsigned char buf[4096];
    int ret;

    printf("\n[Test] SHA-256 Large Data (4KB)\n");

    fill_test_pattern(buf, sizeof(buf), 0x77);

    ret = mbedtls_sha256(buf, sizeof(buf), digest1, 0);
    TEST_RET(ret, "SHA-256 one-shot 4KB");

    mbedtls_sha256_init(&ctx);
    ret = mbedtls_sha256_starts(&ctx, 0);
    TEST_RET(ret, "sha256_starts");

    for (size_t off = 0; off < sizeof(buf); off += 64) {
        ret = mbedtls_sha256_update(&ctx, buf + off, 64);
        if (ret != 0) break;
    }
    TEST_RET(ret, "sha256_update 64-byte chunks");

    ret = mbedtls_sha256_finish(&ctx, digest2);
    TEST_RET(ret, "sha256_finish");
    TEST_ASSERT(memcmp(digest1, digest2, 32) == 0,
                "SHA-256 4KB: one-shot == streaming");
    mbedtls_sha256_free(&ctx);

    return 0;
}

static int test_aes256(void)
{
    mbedtls_aes_context ctx;
    unsigned char output[16], decrypted[16];
    unsigned char cbc_ct[64], cbc_dec[64];
    unsigned char iv[16];
    int ret;

    static const unsigned char aes256_key_tv[32] = {
        0x60, 0x3d, 0xeb, 0x10, 0x15, 0xca, 0x71, 0xbe,
        0x2b, 0x73, 0xae, 0xf0, 0x85, 0x7d, 0x77, 0x81,
        0x1f, 0x35, 0x2c, 0x07, 0x3b, 0x61, 0x08, 0xd7,
        0x2d, 0x98, 0x10, 0xa3, 0x09, 0x14, 0xdf, 0xf4,
    };

    static const unsigned char aes256_ecb_exp[16] = {
        0xf3, 0xee, 0xd1, 0xbd, 0xb5, 0xd2, 0xa0, 0x3c,
        0x06, 0x4b, 0x5a, 0x7e, 0x3d, 0xb1, 0x81, 0xf8,
    };

    printf("\n[Test] AES-256\n");

    mbedtls_aes_init(&ctx);

    /* AES-256-ECB encrypt */
    ret = mbedtls_aes_setkey_enc(&ctx, aes256_key_tv, 256);
    TEST_RET(ret, "AES-256 setkey_enc");

    ret = mbedtls_aes_crypt_ecb(&ctx, MBEDTLS_AES_ENCRYPT,
                                cmac_msg16, output);
    TEST_RET(ret, "AES-256-ECB encrypt");
    TEST_ASSERT(memcmp(output, aes256_ecb_exp, 16) == 0,
                "AES-256-ECB known-answer vector");

    /* AES-256-ECB decrypt */
    ret = mbedtls_aes_setkey_dec(&ctx, aes256_key_tv, 256);
    TEST_RET(ret, "AES-256 setkey_dec");

    ret = mbedtls_aes_crypt_ecb(&ctx, MBEDTLS_AES_DECRYPT,
                                aes256_ecb_exp, decrypted);
    TEST_RET(ret, "AES-256-ECB decrypt");
    TEST_ASSERT(memcmp(decrypted, cmac_msg16, 16) == 0,
                "AES-256-ECB round-trip");

    /* AES-256-CBC encrypt + decrypt round-trip */
    ret = mbedtls_aes_setkey_enc(&ctx, aes256_key_tv, 256);
    TEST_RET(ret, "AES-256 setkey_enc(CBC)");

    memcpy(iv, aes_cbc_iv, 16);
    ret = mbedtls_aes_crypt_cbc(&ctx, MBEDTLS_AES_ENCRYPT,
                                sizeof(aes_sp38a_plaintext),
                                iv, aes_sp38a_plaintext, cbc_ct);
    TEST_RET(ret, "AES-256-CBC encrypt");

    ret = mbedtls_aes_setkey_dec(&ctx, aes256_key_tv, 256);
    TEST_RET(ret, "AES-256 setkey_dec(CBC)");

    memcpy(iv, aes_cbc_iv, 16);
    ret = mbedtls_aes_crypt_cbc(&ctx, MBEDTLS_AES_DECRYPT,
                                sizeof(aes_sp38a_plaintext),
                                iv, cbc_ct, cbc_dec);
    TEST_RET(ret, "AES-256-CBC decrypt");
    TEST_ASSERT(memcmp(cbc_dec, aes_sp38a_plaintext, sizeof(aes_sp38a_plaintext)) == 0,
                "AES-256-CBC round-trip");

    mbedtls_aes_free(&ctx);
    return 0;
}

static int test_aes_threshold(void)
{
    mbedtls_aes_context ctx;
    int ret;

    size_t sizes[] = {
        16,
        MBEDTLS_PUFS_AES_HW_THRESHOLD - 16,
        MBEDTLS_PUFS_AES_HW_THRESHOLD,
        MBEDTLS_PUFS_AES_HW_THRESHOLD + 16,
        MBEDTLS_PUFS_AES_HW_THRESHOLD * 2,
    };
    unsigned char iv[16];
    unsigned char *plain = NULL;
    unsigned char *cipher_buf = NULL;
    unsigned char *dec = NULL;

    printf("\n[Test] AES CBC Threshold Boundary\n");

    mbedtls_aes_init(&ctx);

    for (size_t t = 0; t < sizeof(sizes)/sizeof(sizes[0]); t++) {
        size_t sz = (sizes[t] + 15) / 16 * 16;

        plain = (unsigned char *)malloc(sz);
        cipher_buf = (unsigned char *)malloc(sz);
        dec = (unsigned char *)malloc(sz);
        TEST_ASSERT(plain && cipher_buf && dec, "Alloc buffers");

        fill_test_pattern(plain, sz, (unsigned char)(t * 31 + 7));

        ret = mbedtls_aes_setkey_enc(&ctx, aes128_key, 128);
        TEST_RET(ret, "setkey_enc");

        memcpy(iv, aes_cbc_iv, 16);
        ret = mbedtls_aes_crypt_cbc(&ctx, MBEDTLS_AES_ENCRYPT, sz,
                                    iv, plain, cipher_buf);
        TEST_RET(ret, "CBC encrypt");

        ret = mbedtls_aes_setkey_dec(&ctx, aes128_key, 128);
        TEST_RET(ret, "setkey_dec");

        memcpy(iv, aes_cbc_iv, 16);
        ret = mbedtls_aes_crypt_cbc(&ctx, MBEDTLS_AES_DECRYPT, sz,
                                    iv, cipher_buf, dec);
        TEST_RET(ret, "CBC decrypt");

        {
            char msg[80];
            snprintf(msg, sizeof(msg), "CBC round-trip OK for %zu bytes", sz);
            TEST_ASSERT(memcmp(dec, plain, sz) == 0, msg);
        }

        free(plain); plain = NULL;
        free(cipher_buf); cipher_buf = NULL;
        free(dec); dec = NULL;
    }

    mbedtls_aes_free(&ctx);
    return 0;
}

static int test_gcm_streaming(void)
{
    mbedtls_gcm_context ctx;
    unsigned char ciphertext[sizeof(gcm_plaintext)];
    unsigned char ct_stream[sizeof(gcm_plaintext)];
    unsigned char tag_oneshot[16], tag_stream[16];
    size_t olen;
    int ret;

    printf("\n[Test] AES-GCM Streaming\n");

    mbedtls_gcm_init(&ctx);
    ret = mbedtls_gcm_setkey(&ctx, MBEDTLS_CIPHER_ID_AES, aes128_key, 128);
    TEST_RET(ret, "GCM setkey");

    /* One-shot reference */
    ret = mbedtls_gcm_crypt_and_tag(&ctx, MBEDTLS_GCM_ENCRYPT,
                                    sizeof(gcm_plaintext),
                                    gcm_iv, sizeof(gcm_iv),
                                    gcm_aad, sizeof(gcm_aad),
                                    gcm_plaintext, ciphertext,
                                    sizeof(tag_oneshot), tag_oneshot);
    TEST_RET(ret, "GCM one-shot encrypt");

    /* Streaming encrypt */
    ret = mbedtls_gcm_starts(&ctx, MBEDTLS_GCM_ENCRYPT,
                             gcm_iv, sizeof(gcm_iv));
    TEST_RET(ret, "GCM starts");

    ret = mbedtls_gcm_update_ad(&ctx, gcm_aad, sizeof(gcm_aad));
    TEST_RET(ret, "GCM update_ad");

    size_t total = 0;
    for (size_t off = 0; off < sizeof(gcm_plaintext); off += 8) {
        size_t step = sizeof(gcm_plaintext) - off;
        if (step > 8) step = 8;
        olen = 0;
        ret = mbedtls_gcm_update(&ctx, gcm_plaintext + off, step,
                                 ct_stream + total,
                                 sizeof(ct_stream) - total, &olen);
        if (ret != 0) break;
        total += olen;
    }
    TEST_RET(ret, "GCM streaming updates");

    olen = 0;
    ret = mbedtls_gcm_finish(&ctx, ct_stream + total,
                             sizeof(ct_stream) - total, &olen,
                             tag_stream, sizeof(tag_stream));
    TEST_RET(ret, "GCM finish");
    total += olen;

    TEST_ASSERT(total == sizeof(gcm_plaintext),
                "GCM streaming output length matches");
    TEST_ASSERT(memcmp(ct_stream, ciphertext, sizeof(ciphertext)) == 0,
                "GCM streaming ciphertext matches one-shot");
    TEST_ASSERT(memcmp(tag_stream, tag_oneshot, sizeof(tag_oneshot)) == 0,
                "GCM streaming tag matches one-shot");

    mbedtls_gcm_free(&ctx);
    return 0;
}

static int test_gcm_aad_only(void)
{
    mbedtls_gcm_context ctx;
    unsigned char tag[16];
    int ret;

    printf("\n[Test] AES-GCM AAD Only (no plaintext)\n");

    mbedtls_gcm_init(&ctx);
    ret = mbedtls_gcm_setkey(&ctx, MBEDTLS_CIPHER_ID_AES, aes128_key, 128);
    TEST_RET(ret, "GCM setkey");

    ret = mbedtls_gcm_crypt_and_tag(&ctx, MBEDTLS_GCM_ENCRYPT,
                                    0,
                                    gcm_iv, sizeof(gcm_iv),
                                    gcm_aad, sizeof(gcm_aad),
                                    NULL, NULL,
                                    sizeof(tag), tag);
    TEST_RET(ret, "GCM encrypt (AAD only)");

    ret = mbedtls_gcm_auth_decrypt(&ctx, 0,
                                   gcm_iv, sizeof(gcm_iv),
                                   gcm_aad, sizeof(gcm_aad),
                                   tag, sizeof(tag),
                                   NULL, NULL);
    TEST_RET(ret, "GCM auth_decrypt (AAD only, valid tag)");

    tag[0] ^= 0x01;
    ret = mbedtls_gcm_auth_decrypt(&ctx, 0,
                                   gcm_iv, sizeof(gcm_iv),
                                   gcm_aad, sizeof(gcm_aad),
                                   tag, sizeof(tag),
                                   NULL, NULL);
    TEST_EXPECT_RET(ret, MBEDTLS_ERR_GCM_AUTH_FAILED,
                    "GCM AAD-only rejects tampered tag");

    mbedtls_gcm_free(&ctx);
    return 0;
}

static int test_cmac_streaming(void)
{
    mbedtls_cipher_context_t ctx;
    const mbedtls_cipher_info_t *cipher_info;
    unsigned char output_oneshot[16], output_stream[16];
    int ret;

    printf("\n[Test] AES-CMAC Streaming\n");

    cipher_info = mbedtls_cipher_info_from_type(MBEDTLS_CIPHER_AES_128_ECB);
    TEST_ASSERT(cipher_info != NULL, "Resolve cipher info");

    /* One-shot reference */
    ret = mbedtls_cipher_cmac(cipher_info, aes128_key, 128,
                              cmac_msg16, sizeof(cmac_msg16),
                              output_oneshot);
    TEST_RET(ret, "CMAC one-shot");

    /* Streaming: 4-byte chunks */
    mbedtls_cipher_init(&ctx);
    ret = mbedtls_cipher_setup(&ctx, cipher_info);
    TEST_RET(ret, "cipher_setup");

    ret = mbedtls_cipher_cmac_starts(&ctx, aes128_key, 128);
    TEST_RET(ret, "cmac_starts");

    for (size_t off = 0; off < sizeof(cmac_msg16); off += 4) {
        ret = mbedtls_cipher_cmac_update(&ctx, cmac_msg16 + off, 4);
        if (ret != 0) break;
    }
    TEST_RET(ret, "cmac_update streaming");

    ret = mbedtls_cipher_cmac_finish(&ctx, output_stream);
    TEST_RET(ret, "cmac_finish");

    TEST_ASSERT(memcmp(output_oneshot, output_stream, 16) == 0,
                "CMAC streaming matches one-shot");
    TEST_ASSERT(memcmp(output_stream, cmac_expected, 16) == 0,
                "CMAC streaming matches known vector");

    mbedtls_cipher_free(&ctx);
    return 0;
}

static int test_cmac_empty(void)
{
    const mbedtls_cipher_info_t *cipher_info;
    unsigned char output[16];
    int ret;

    /* RFC 4493 TC1: AES-CMAC with zero-length message */
    static const unsigned char cmac_empty_expected[16] = {
        0xbb, 0x1d, 0x69, 0x29, 0xe9, 0x59, 0x37, 0x28,
        0x7f, 0xa3, 0x7d, 0x12, 0x9b, 0x75, 0x67, 0x46,
    };

    printf("\n[Test] AES-CMAC Empty Message\n");

    cipher_info = mbedtls_cipher_info_from_type(MBEDTLS_CIPHER_AES_128_ECB);
    TEST_ASSERT(cipher_info != NULL, "Resolve cipher info");

    ret = mbedtls_cipher_cmac(cipher_info, aes128_key, 128,
                              NULL, 0, output);
    TEST_RET(ret, "CMAC empty message");
    TEST_ASSERT(memcmp(output, cmac_empty_expected, 16) == 0,
                "AES-CMAC empty matches RFC 4493 TC1");

    return 0;
}

static int rng_counter;

static int test_rng_stateful(void *context, unsigned char *output,
                             size_t output_len)
{
    (void)context;
    for (size_t i = 0; i < output_len; i++)
        output[i] = (unsigned char)(rng_counter++ & 0xff);
    return 0;
}

static int test_ecdh(void)
{
    mbedtls_ecp_group grp;
    mbedtls_ecp_point Q1, Q2;
    mbedtls_mpi d1, d2, z1, z2;
    int ret;

    printf("\n[Test] ECDH Key Agreement\n");

    mbedtls_ecp_group_init(&grp);
    mbedtls_ecp_point_init(&Q1);
    mbedtls_ecp_point_init(&Q2);
    mbedtls_mpi_init(&d1);
    mbedtls_mpi_init(&d2);
    mbedtls_mpi_init(&z1);
    mbedtls_mpi_init(&z2);

    ret = mbedtls_ecp_group_load(&grp, MBEDTLS_ECP_DP_SECP256R1);
    TEST_RET(ret, "Load secp256r1 group");

    /*
     * Generate real software private keys (not HW sentinels).
     * mbedtls_ecdh_gen_public ALT stores d in HW and returns a sentinel,
     * which breaks the d1*Q2 == d2*Q1 cross-check.
     */
    rng_counter = 0x10;
    ret = mbedtls_ecp_gen_privkey(&grp, &d1, test_rng_stateful, NULL);
    TEST_RET(ret, "Generate private key 1");
    ret = mbedtls_ecp_mul(&grp, &Q1, &d1, &grp.G, test_rng_stateful, NULL);
    TEST_RET(ret, "Compute public key 1");

    ret = mbedtls_ecp_gen_privkey(&grp, &d2, test_rng_stateful, NULL);
    TEST_RET(ret, "Generate private key 2");
    ret = mbedtls_ecp_mul(&grp, &Q2, &d2, &grp.G, test_rng_stateful, NULL);
    TEST_RET(ret, "Compute public key 2");

    /* Shared secrets must match: d1*Q2 == d2*Q1 */
    ret = mbedtls_ecdh_compute_shared(&grp, &z1, &Q2, &d1,
                                      test_rng_stateful, NULL);
    TEST_RET(ret, "Compute shared secret (d1*Q2)");

    ret = mbedtls_ecdh_compute_shared(&grp, &z2, &Q1, &d2,
                                      test_rng_stateful, NULL);
    TEST_RET(ret, "Compute shared secret (d2*Q1)");

    TEST_ASSERT(mbedtls_mpi_cmp_mpi(&z1, &z2) == 0,
                "ECDH shared secrets match (d1*Q2 == d2*Q1)");

    mbedtls_mpi_free(&z2);
    mbedtls_mpi_free(&z1);
    mbedtls_mpi_free(&d2);
    mbedtls_mpi_free(&d1);
    mbedtls_ecp_point_free(&Q2);
    mbedtls_ecp_point_free(&Q1);
    mbedtls_ecp_group_free(&grp);
    return 0;
}

static int test_ecdsa_curves(void)
{
    int ret;

    struct {
        mbedtls_ecp_group_id grp_id;
        const char *name;
        mbedtls_md_type_t md_type;
        size_t hlen;
    } curves[] = {
        { MBEDTLS_ECP_DP_SECP256R1, "secp256r1", MBEDTLS_MD_SHA256, 32 },
        { MBEDTLS_ECP_DP_SECP384R1, "secp384r1", MBEDTLS_MD_SHA384, 48 },
    };

    printf("\n[Test] ECDSA Multiple Curves\n");

    rng_counter = 0x42;

    for (size_t i = 0; i < sizeof(curves)/sizeof(curves[0]); i++) {
        mbedtls_ecp_group grp;
        mbedtls_ecp_point Q;
        mbedtls_mpi d, r, s;
        unsigned char hash[64];

        mbedtls_ecp_group_init(&grp);
        mbedtls_ecp_point_init(&Q);
        mbedtls_mpi_init(&d);
        mbedtls_mpi_init(&r);
        mbedtls_mpi_init(&s);

        ret = mbedtls_ecp_group_load(&grp, curves[i].grp_id);
        if (ret != 0) {
            printf("[SKIP] %s not supported\n", curves[i].name);
            test_skipped++;
            goto next;
        }

        /*
         * Generate real software private key (not HW sentinel).
         * mbedtls_ecdh_gen_public ALT returns sentinel d that can't be
         * used with mbedtls_ecdsa_sign.
         */
        ret = mbedtls_ecp_gen_privkey(&grp, &d, test_rng_stateful, NULL);
        if (ret != 0) {
            printf("[FAIL] %s privkey gen (ret=%d)\n", curves[i].name, ret);
            test_failed++;
            goto next;
        }

        ret = mbedtls_ecp_mul(&grp, &Q, &d, &grp.G,
                              test_rng_stateful, NULL);
        if (ret != 0) {
            printf("[FAIL] %s pubkey compute (ret=%d)\n", curves[i].name, ret);
            test_failed++;
            goto next;
        }

        if (curves[i].hlen == 32)
            mbedtls_sha256(sha_msg, sizeof(sha_msg), hash, 0);
        else
            mbedtls_sha512(sha_msg, sizeof(sha_msg), hash, 1);

        ret = mbedtls_ecdsa_sign_det_ext(&grp, &r, &s, &d,
                                         hash, curves[i].hlen,
                                         curves[i].md_type,
                                         test_rng_stateful, NULL);
        if (ret != 0) {
            printf("[FAIL] %s sign (ret=%d)\n", curves[i].name, ret);
            test_failed++;
            goto next;
        }

        ret = mbedtls_ecdsa_verify(&grp, hash, curves[i].hlen,
                                   &Q, &r, &s);
        {
            char msg[80];
            snprintf(msg, sizeof(msg),
                     "ECDSA sign+verify (%s)", curves[i].name);
            TEST_RET(ret, msg);
        }

next:
        mbedtls_mpi_free(&s);
        mbedtls_mpi_free(&r);
        mbedtls_mpi_free(&d);
        mbedtls_ecp_point_free(&Q);
        mbedtls_ecp_group_free(&grp);
    }

    return 0;
}

int main(void)
{
    int ret = 0;

    printf("mbedTLS PUFS ALT verify\n");

    if (test_sha2() != 0)
        ret = 1;
    if (test_sha2_streaming() != 0)
        ret = 1;
    if (test_sha2_empty() != 0)
        ret = 1;
    if (test_sha2_large() != 0)
        ret = 1;
    if (test_aes() != 0)
        ret = 1;
    if (test_aes256() != 0)
        ret = 1;
    if (test_aes_threshold() != 0)
        ret = 1;
    if (test_gcm() != 0)
        ret = 1;
    if (test_gcm_streaming() != 0)
        ret = 1;
    if (test_gcm_aad_only() != 0)
        ret = 1;
    if (test_ccm() != 0)
        ret = 1;
    if (test_cmac() != 0)
        ret = 1;
    if (test_cmac_streaming() != 0)
        ret = 1;
    if (test_cmac_empty() != 0)
        ret = 1;
    if (test_ecdsa_verify() != 0)
        ret = 1;
    if (test_ecdsa_curves() != 0)
        ret = 1;
    if (test_ecdh() != 0)
        ret = 1;

    printf("\nSummary: %d passed, %d failed, %d skipped\n",
           test_passed, test_failed, test_skipped);

    return ret;
}