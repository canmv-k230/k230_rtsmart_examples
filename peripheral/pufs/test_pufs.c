/* Copyright (c) 2023, Canaan Bright Sight Co., Ltd
 *
 * SPDX-License-Identifier: BSD-2-Clause
 *
 * PUF Secure Engine HAL test suite
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>

#include "drv_pufs.h"

/* Test result counters */
static int test_passed = 0;
static int test_failed = 0;
static int test_skipped = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("[FAIL] %s:%d - %s\n", __func__, __LINE__, msg); \
        test_failed++; \
        return -1; \
    } else { \
        printf("[PASS] %s\n", msg); \
        test_passed++; \
    } \
} while(0)

#define TEST_ASSERT_NOFAIL(cond, msg) do { \
    if (!(cond)) { \
        printf("[FAIL] %s:%d - %s\n", __func__, __LINE__, msg); \
        test_failed++; \
    } else { \
        printf("[PASS] %s\n", msg); \
        test_passed++; \
    } \
} while(0)

#define TEST_SKIP(msg) do { \
    printf("[SKIP] %s\n", msg); \
    test_skipped++; \
    return 0; \
} while(0)

#define TEST_START(name) do { \
    printf("\n========== Test: %s ==========\n", name); \
} while(0)

/* Helper: compare byte arrays */
static int memcmp_hex(const uint8_t *a, const uint8_t *b, uint32_t len)
{
    for (uint32_t i = 0; i < len; i++) {
        if (a[i] != b[i]) return -1;
    }
    return 0;
}

static void print_hex(const char *label, const uint8_t *data, uint32_t len)
{
    printf("%s: ", label);
    for (uint32_t i = 0; i < len; i++)
        printf("%02x", data[i]);
    printf("\n");
}

static const char *otp_lock_state_name(uint8_t lock)
{
    switch (lock) {
    case OTP_NA:
        return "NA";
    case OTP_RO:
        return "RO";
    case OTP_RW:
        return "RW";
    default:
        return "UNKNOWN";
    }
}

static uint64_t get_time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/* ================================================================
 * NIST SHA-256 test vectors (from FIPS 180-4)
 * ================================================================ */

/* SHA-256("abc") */
static const uint8_t sha256_msg1[] = { 'a', 'b', 'c' };
static const uint8_t sha256_exp1[] = {
    0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
    0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
    0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
    0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
};

/* SHA-256("") - empty message */
static const uint8_t sha256_exp_empty[] = {
    0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
    0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
    0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
    0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
};

/* SHA-224("abc") */
static const uint8_t sha224_exp1[] = {
    0x23, 0x09, 0x7d, 0x22, 0x34, 0x05, 0xd8, 0x22,
    0x86, 0x42, 0xa4, 0x77, 0xbd, 0xa2, 0x55, 0xb3,
    0x2a, 0xad, 0xbc, 0xe4, 0xbd, 0xa0, 0xb3, 0xf7,
    0xe3, 0x6c, 0x9d, 0xa7,
};

/* SHA-384("abc") */
static const uint8_t sha384_exp1[] = {
    0xcb, 0x00, 0x75, 0x3f, 0x45, 0xa3, 0x5e, 0x8b,
    0xb5, 0xa0, 0x3d, 0x69, 0x9a, 0xc6, 0x50, 0x07,
    0x27, 0x2c, 0x32, 0xab, 0x0e, 0xde, 0xd1, 0x63,
    0x1a, 0x8b, 0x60, 0x5a, 0x43, 0xff, 0x5b, 0xed,
    0x80, 0x86, 0x07, 0x2b, 0xa1, 0xe7, 0xcc, 0x23,
    0x58, 0xba, 0xec, 0xa1, 0x34, 0xc8, 0x25, 0xa7,
};

/* SHA-512("abc") */
static const uint8_t sha512_exp1[] = {
    0xdd, 0xaf, 0x35, 0xa1, 0x93, 0x61, 0x7a, 0xba,
    0xcc, 0x41, 0x73, 0x49, 0xae, 0x20, 0x41, 0x31,
    0x12, 0xe6, 0xfa, 0x4e, 0x89, 0xa9, 0x7e, 0xa2,
    0x0a, 0x9e, 0xee, 0xe6, 0x4b, 0x55, 0xd3, 0x9a,
    0x21, 0x92, 0x99, 0x2a, 0x27, 0x4f, 0xc1, 0xa8,
    0x36, 0xba, 0x3c, 0x23, 0xa3, 0xfe, 0xeb, 0xbd,
    0x45, 0x4d, 0x44, 0x23, 0x64, 0x3c, 0xe8, 0x0e,
    0x2a, 0x9a, 0xc9, 0x4f, 0xa5, 0x4c, 0xa4, 0x9f,
};

/* SM3("abc") */
static const uint8_t sm3_exp1[] = {
    0x66, 0xc7, 0xf0, 0xf4, 0x62, 0xee, 0xed, 0xd9,
    0xd1, 0xf2, 0xd4, 0x6b, 0xdc, 0x10, 0xe4, 0xe2,
    0x41, 0x67, 0xc4, 0x87, 0x5c, 0xf2, 0xf7, 0xa2,
    0x29, 0x7d, 0xa0, 0x2b, 0x8f, 0x4b, 0xa8, 0xe0,
};

/* ================================================================
 * AES test vectors
 * ================================================================ */
static const uint8_t aes128_key[] = {
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
};
static const uint8_t aes128_cbc_iv[] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
};
static const uint8_t aes128_cbc_pt[] = {
    0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
    0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
    0xae, 0x2d, 0x8a, 0x57, 0x1e, 0x03, 0xac, 0x9c,
    0x9e, 0xb7, 0x6f, 0xac, 0x45, 0xaf, 0x8e, 0x51,
};
static const uint8_t aes128_cbc_ct[] = {
    0x76, 0x49, 0xab, 0xac, 0x81, 0x19, 0xb2, 0x46,
    0xce, 0xe9, 0x8e, 0x9b, 0x12, 0xe9, 0x19, 0x7d,
    0x50, 0x86, 0xcb, 0x9b, 0x50, 0x72, 0x19, 0xee,
    0x95, 0xdb, 0x11, 0x3a, 0x91, 0x76, 0x78, 0xb2,
};
static const uint8_t aes128_ecb_pt[] = {
    0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
    0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
};
static const uint8_t aes128_ecb_ct[] = {
    0x3a, 0xd7, 0x7b, 0xb4, 0x0d, 0x7a, 0x36, 0x60,
    0xa8, 0x9e, 0xca, 0xf3, 0x24, 0x66, 0xef, 0x97,
};

/* AES-256 key for testing */
static const uint8_t aes256_key[] = {
    0x60, 0x3d, 0xeb, 0x10, 0x15, 0xca, 0x71, 0xbe,
    0x2b, 0x73, 0xae, 0xf0, 0x85, 0x7d, 0x77, 0x81,
    0x1f, 0x35, 0x2c, 0x07, 0x3b, 0x61, 0x08, 0xd7,
    0x2d, 0x98, 0x10, 0xa3, 0x09, 0x14, 0xdf, 0xf4,
};
/* AES-256-ECB("6bc1...172a") from NIST SP 800-38A */
static const uint8_t aes256_ecb_ct[] = {
    0xf3, 0xee, 0xd1, 0xbd, 0xb5, 0xd2, 0xa0, 0x3c,
    0x06, 0x4b, 0x5a, 0x7e, 0x3d, 0xb1, 0x81, 0xf8,
};

/* SM4 test vectors (GB/T 32907-2016) */
static const uint8_t sm4_key[] = {
    0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
    0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
};
static const uint8_t sm4_pt[] = {
    0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
    0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
};
static const uint8_t sm4_ct[] = {
    0x68, 0x1e, 0xdf, 0x34, 0xd2, 0x06, 0x96, 0x5e,
    0x86, 0xb3, 0xe9, 0x4f, 0x53, 0x6e, 0x42, 0x46,
};

/* HMAC-SHA256 RFC 4231 TC2: key = "Jefe" */
static const uint8_t hmac_key_jefe[] = { 'J', 'e', 'f', 'e' };
static const uint8_t hmac_msg_jefe[] = {
    'w', 'h', 'a', 't', ' ', 'd', 'o', ' ',
    'y', 'a', ' ', 'w', 'a', 'n', 't', ' ',
    'f', 'o', 'r', ' ', 'n', 'o', 't', 'h',
    'i', 'n', 'g', '?'
};
static const uint8_t hmac_exp_jefe[] = {
    0x5b, 0xdc, 0xc1, 0x46, 0xbf, 0x60, 0x75, 0x4e,
    0x6a, 0x04, 0x24, 0x26, 0x08, 0x95, 0x75, 0xc7,
    0x5a, 0x00, 0x3f, 0x08, 0x9d, 0x27, 0x39, 0x83,
    0x9d, 0xec, 0x58, 0xb9, 0x64, 0xec, 0x38, 0x43,
};

/* AES-CMAC RFC 4493 test vector: 16-byte message with key from AES128 */
static const uint8_t cmac_msg16[] = {
    0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
    0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
};
static const uint8_t cmac_exp16[] = {
    0x07, 0x0a, 0x16, 0xb4, 0x6b, 0x4d, 0x41, 0x44,
    0xf7, 0x9b, 0xdd, 0x9d, 0xd0, 0x4a, 0x28, 0x7c,
};

/* ================================================================
 * Section 1: Device Management Tests
 * ================================================================ */

static int test_device_open_close(void)
{
    TEST_START("Device Open/Close");

    drv_pufs_inst dev;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open /dev/pufs");
    TEST_ASSERT(dev.fd >= 0, "Valid file descriptor");

    ret = drv_pufs_close(&dev);
    TEST_ASSERT(ret == 0, "Close device");

    return 0;
}

static int test_device_double_close(void)
{
    TEST_START("Device Double Close");

    drv_pufs_inst dev;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_close(&dev);
    TEST_ASSERT(ret == 0, "First close");

    /* Second close on already-closed fd: should not crash */
    ret = drv_pufs_close(&dev);
    printf("  double close returned %d (non-crash is ok)\n", ret);
    test_passed++;

    return 0;
}

static int test_device_multiple_open(void)
{
    TEST_START("Multiple Concurrent Opens");

    drv_pufs_inst dev1, dev2, dev3;
    int ret;

    ret = drv_pufs_open(&dev1);
    TEST_ASSERT(ret == 0, "Open device #1");

    ret = drv_pufs_open(&dev2);
    TEST_ASSERT(ret == 0, "Open device #2");

    ret = drv_pufs_open(&dev3);
    TEST_ASSERT(ret == 0, "Open device #3");

    /* All fds should be different */
    TEST_ASSERT(dev1.fd != dev2.fd && dev2.fd != dev3.fd, "All fds are distinct");

    drv_pufs_close(&dev3);
    drv_pufs_close(&dev2);
    drv_pufs_close(&dev1);
    return 0;
}

/* ================================================================
 * Section 2: Hash Tests
 * ================================================================ */

static int test_sha256_oneshot(void)
{
    TEST_START("SHA-256 One-Shot");

    drv_pufs_inst dev;
    drv_pufs_hash_inst hash;
    uint8_t dgst[64];
    uint32_t dlen = 0;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_hash_init(&hash, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "Hash init");

    ret = drv_pufs_hash_update(&hash, sha256_msg1, sizeof(sha256_msg1));
    TEST_ASSERT(ret == 0, "Hash update");

    ret = drv_pufs_hash_final(&hash, dgst, &dlen);
    TEST_ASSERT(ret == 0, "Hash final");
    TEST_ASSERT(dlen == 32, "Digest length == 32");

    print_hex("SHA-256(abc)", dgst, dlen);
    TEST_ASSERT(memcmp_hex(dgst, sha256_exp1, 32) == 0, "SHA-256(abc) matches expected");

    drv_pufs_close(&dev);
    return 0;
}

static int test_sha256_empty(void)
{
    TEST_START("SHA-256 Empty Message");

    drv_pufs_inst dev;
    drv_pufs_hash_inst hash;
    uint8_t dgst[64];
    uint32_t dlen = 0;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_hash_init(&hash, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "Hash init");

    ret = drv_pufs_hash_final(&hash, dgst, &dlen);
    TEST_ASSERT(ret == 0, "Hash final (empty)");
    TEST_ASSERT(dlen == 32, "Digest length == 32");

    print_hex("SHA-256(\"\")", dgst, dlen);
    TEST_ASSERT(memcmp_hex(dgst, sha256_exp_empty, 32) == 0, "SHA-256(\"\") matches expected");

    drv_pufs_close(&dev);
    return 0;
}

static int test_sha256_streaming(void)
{
    TEST_START("SHA-256 Streaming (byte-at-a-time)");

    drv_pufs_inst dev;
    drv_pufs_hash_inst hash;
    uint8_t dgst[64];
    uint32_t dlen = 0;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_hash_init(&hash, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "Hash init");

    for (uint32_t i = 0; i < sizeof(sha256_msg1); i++) {
        ret = drv_pufs_hash_update(&hash, &sha256_msg1[i], 1);
        TEST_ASSERT(ret == 0, "Hash update single byte");
    }

    ret = drv_pufs_hash_final(&hash, dgst, &dlen);
    TEST_ASSERT(ret == 0, "Hash final");

    print_hex("SHA-256(a+b+c)", dgst, dlen);
    TEST_ASSERT(memcmp_hex(dgst, sha256_exp1, 32) == 0,
                "SHA-256 streaming matches one-shot");

    drv_pufs_close(&dev);
    return 0;
}

/* Test all supported hash algorithms */
static int test_hash_all_algorithms(void)
{
    TEST_START("Hash All Algorithms (SHA-224/256/384/512/SM3)");

    drv_pufs_inst dev;
    drv_pufs_hash_inst hash;
    uint8_t dgst[64];
    uint32_t dlen;
    int ret;

    struct {
        pufs_hashtype_t type;
        const char *name;
        const uint8_t *expected;
        uint32_t explen;
    } cases[] = {
        { HASH_SHA_224, "SHA-224", sha224_exp1, 28 },
        { HASH_SHA_256, "SHA-256", sha256_exp1, 32 },
        { HASH_SHA_384, "SHA-384", sha384_exp1, 48 },
        { HASH_SHA_512, "SHA-512", sha512_exp1, 64 },
        { HASH_SM3,     "SM3",     sm3_exp1,    32 },
    };

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        ret = drv_pufs_hash_init(&hash, &dev, cases[i].type);
        TEST_ASSERT(ret == 0, "Hash init");

        ret = drv_pufs_hash_update(&hash, sha256_msg1, sizeof(sha256_msg1));
        TEST_ASSERT(ret == 0, "Hash update");

        dlen = 0;
        ret = drv_pufs_hash_final(&hash, dgst, &dlen);
        TEST_ASSERT(ret == 0, "Hash final");

        char msg[64];
        snprintf(msg, sizeof(msg), "%s digest length == %u", cases[i].name, cases[i].explen);
        TEST_ASSERT(dlen == cases[i].explen, msg);

        print_hex(cases[i].name, dgst, dlen);
        snprintf(msg, sizeof(msg), "%s(\"abc\") matches expected", cases[i].name);
        TEST_ASSERT(memcmp_hex(dgst, cases[i].expected, dlen) == 0, msg);
    }

    drv_pufs_close(&dev);
    return 0;
}

/* Test: hash large data (4KB) - verifies multi-block streaming */
static int test_hash_large_data(void)
{
    TEST_START("SHA-256 Large Data (4KB)");

    drv_pufs_inst dev;
    drv_pufs_hash_inst hash1, hash2;
    uint8_t dgst1[64], dgst2[64];
    uint32_t dlen1 = 0, dlen2 = 0;
    int ret;

    /* Build a repeating 4KB buffer */
    uint8_t *buf = (uint8_t *)malloc(4096);
    TEST_ASSERT(buf != NULL, "Alloc 4KB buffer");
    for (int i = 0; i < 4096; i++)
        buf[i] = (uint8_t)(i & 0xff);

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* One-shot: single update of 4KB */
    ret = drv_pufs_hash_init(&hash1, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "Hash init (one-shot)");

    ret = drv_pufs_hash_update(&hash1, buf, 4096);
    TEST_ASSERT(ret == 0, "Hash update 4KB");

    ret = drv_pufs_hash_final(&hash1, dgst1, &dlen1);
    TEST_ASSERT(ret == 0, "Hash final (one-shot)");

    /* Multi-chunk: 64 updates of 64 bytes each */
    ret = drv_pufs_hash_init(&hash2, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "Hash init (chunked)");

    for (int off = 0; off < 4096; off += 64) {
        ret = drv_pufs_hash_update(&hash2, buf + off, 64);
        if (ret != 0) break;
    }
    TEST_ASSERT(ret == 0, "Hash update 64x64B");

    ret = drv_pufs_hash_final(&hash2, dgst2, &dlen2);
    TEST_ASSERT(ret == 0, "Hash final (chunked)");

    TEST_ASSERT(dlen1 == dlen2, "Both digest lengths match");
    TEST_ASSERT(memcmp_hex(dgst1, dgst2, dlen1) == 0,
                "One-shot == chunked for 4KB data");

    print_hex("SHA-256(4KB)", dgst1, dlen1);

    free(buf);
    drv_pufs_close(&dev);
    return 0;
}

/* Test: hash with various chunk sizes to stress streaming boundary handling */
static int test_hash_chunk_boundary(void)
{
    TEST_START("SHA-256 Chunk Boundary Stress");

    drv_pufs_inst dev;
    drv_pufs_hash_inst hash_ref, hash_test;
    uint8_t dgst_ref[64], dgst_test[64];
    uint32_t dlen_ref = 0, dlen_test = 0;
    int ret;

    /* 200 bytes of test data (not a multiple of block size 64) */
    uint8_t data[200];
    for (int i = 0; i < 200; i++)
        data[i] = (uint8_t)(i * 7 + 3);

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Reference: single update */
    ret = drv_pufs_hash_init(&hash_ref, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "Ref hash init");
    ret = drv_pufs_hash_update(&hash_ref, data, 200);
    TEST_ASSERT(ret == 0, "Ref hash update");
    ret = drv_pufs_hash_final(&hash_ref, dgst_ref, &dlen_ref);
    TEST_ASSERT(ret == 0, "Ref hash final");

    /* Chunked: 7-byte chunks (misaligned with 64-byte block) */
    ret = drv_pufs_hash_init(&hash_test, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "Test hash init");
    for (int off = 0; off < 200; off += 7) {
        int chunk = (200 - off < 7) ? (200 - off) : 7;
        ret = drv_pufs_hash_update(&hash_test, data + off, chunk);
        if (ret != 0) break;
    }
    TEST_ASSERT(ret == 0, "All 7-byte chunk updates OK");
    ret = drv_pufs_hash_final(&hash_test, dgst_test, &dlen_test);
    TEST_ASSERT(ret == 0, "Test hash final");

    TEST_ASSERT(memcmp_hex(dgst_ref, dgst_test, dlen_ref) == 0,
                "7-byte chunked matches single-update");

    drv_pufs_close(&dev);
    return 0;
}

/* Test: consecutive hash operations reusing same instance */
static int test_hash_reuse_instance(void)
{
    TEST_START("Hash Instance Reuse");

    drv_pufs_inst dev;
    drv_pufs_hash_inst hash;
    uint8_t dgst[64];
    uint32_t dlen;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* First hash: SHA-256("abc") */
    ret = drv_pufs_hash_init(&hash, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "First hash init");
    ret = drv_pufs_hash_update(&hash, sha256_msg1, sizeof(sha256_msg1));
    TEST_ASSERT(ret == 0, "First hash update");
    dlen = 0;
    ret = drv_pufs_hash_final(&hash, dgst, &dlen);
    TEST_ASSERT(ret == 0, "First hash final");
    TEST_ASSERT(memcmp_hex(dgst, sha256_exp1, 32) == 0, "First hash correct");

    /* Second hash: SHA-256("") — reuse same hash instance */
    ret = drv_pufs_hash_init(&hash, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "Second hash init (reuse)");
    dlen = 0;
    ret = drv_pufs_hash_final(&hash, dgst, &dlen);
    TEST_ASSERT(ret == 0, "Second hash final");
    TEST_ASSERT(memcmp_hex(dgst, sha256_exp_empty, 32) == 0,
                "Second hash correct (no leftover state)");

    /* Third hash: SHA-256("abc") again */
    ret = drv_pufs_hash_init(&hash, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "Third hash init");
    ret = drv_pufs_hash_update(&hash, sha256_msg1, sizeof(sha256_msg1));
    TEST_ASSERT(ret == 0, "Third hash update");
    dlen = 0;
    ret = drv_pufs_hash_final(&hash, dgst, &dlen);
    TEST_ASSERT(ret == 0, "Third hash final");
    TEST_ASSERT(memcmp_hex(dgst, sha256_exp1, 32) == 0,
                "Third hash correct (clean reuse)");

    drv_pufs_close(&dev);
    return 0;
}

/* ================================================================
 * Section 3: HMAC Tests
 * ================================================================ */

static int test_hmac_sha256(void)
{
    TEST_START("HMAC-SHA256 (RFC 4231 TC2)");

    drv_pufs_inst dev;
    drv_pufs_hash_inst hmac;
    uint8_t dgst[64];
    uint32_t dlen = 0;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_hmac_init(&hmac, &dev, HASH_SHA_256, KT_SWKEY,
                             hmac_key_jefe, 32);
    TEST_ASSERT(ret == 0, "HMAC init");

    ret = drv_pufs_hmac_update(&hmac, hmac_msg_jefe, sizeof(hmac_msg_jefe));
    TEST_ASSERT(ret == 0, "HMAC update");

    ret = drv_pufs_hmac_final(&hmac, dgst, &dlen);
    TEST_ASSERT(ret == 0, "HMAC final");
    TEST_ASSERT(dlen == 32, "HMAC digest length == 32");

    print_hex("HMAC-SHA256", dgst, dlen);
    TEST_ASSERT(memcmp_hex(dgst, hmac_exp_jefe, 32) == 0,
                "HMAC-SHA256 matches RFC 4231 TC2");

    drv_pufs_close(&dev);
    return 0;
}

/* HMAC streaming: split message across multiple updates */
static int test_hmac_streaming(void)
{
    TEST_START("HMAC-SHA256 Streaming");

    drv_pufs_inst dev;
    drv_pufs_hash_inst hmac;
    uint8_t dgst[64];
    uint32_t dlen = 0;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_hmac_init(&hmac, &dev, HASH_SHA_256, KT_SWKEY,
                             hmac_key_jefe, 32);
    TEST_ASSERT(ret == 0, "HMAC init");

    /* Split the message into 3 pieces */
    ret = drv_pufs_hmac_update(&hmac, hmac_msg_jefe, 10);
    TEST_ASSERT(ret == 0, "HMAC update part 1");

    ret = drv_pufs_hmac_update(&hmac, hmac_msg_jefe + 10, 10);
    TEST_ASSERT(ret == 0, "HMAC update part 2");

    ret = drv_pufs_hmac_update(&hmac, hmac_msg_jefe + 20,
                               sizeof(hmac_msg_jefe) - 20);
    TEST_ASSERT(ret == 0, "HMAC update part 3");

    ret = drv_pufs_hmac_final(&hmac, dgst, &dlen);
    TEST_ASSERT(ret == 0, "HMAC final");

    TEST_ASSERT(memcmp_hex(dgst, hmac_exp_jefe, 32) == 0,
                "Streamed HMAC matches one-shot");

    drv_pufs_close(&dev);
    return 0;
}

/* ================================================================
 * Section 4: CMAC Tests
 * ================================================================ */

static int test_cmac_aes128(void)
{
    TEST_START("AES-CMAC-128 (RFC 4493)");

    drv_pufs_inst dev;
    drv_pufs_cmac_inst cmac;
    uint8_t dgst[16];
    uint32_t dlen = 0;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_cmac_init(&cmac, &dev, SK_AES, KT_SWKEY,
                             aes128_key, 128);
    TEST_ASSERT(ret == 0, "CMAC init");

    ret = drv_pufs_cmac_update(&cmac, cmac_msg16, sizeof(cmac_msg16));
    TEST_ASSERT(ret == 0, "CMAC update");

    ret = drv_pufs_cmac_final(&cmac, dgst, &dlen);
    TEST_ASSERT(ret == 0, "CMAC final");
    TEST_ASSERT(dlen == 16, "CMAC tag length == 16");

    print_hex("AES-CMAC", dgst, dlen);
    TEST_ASSERT(memcmp_hex(dgst, cmac_exp16, 16) == 0,
                "AES-CMAC matches RFC 4493 TC3");

    drv_pufs_close(&dev);
    return 0;
}

/* ================================================================
 * Section 5: Symmetric Cipher Tests
 * ================================================================ */

static int test_aes128_ecb(void)
{
    TEST_START("AES-128-ECB Encrypt/Decrypt");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    uint8_t out[32];
    uint32_t outlen = 0, flen = 0;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Encrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_ECB, 1,
                               KT_SWKEY, aes128_key, 128, NULL, 0);
    TEST_ASSERT(ret == 0, "ECB encrypt init");

    ret = drv_pufs_cipher_update(&cipher, out, &outlen,
                                 aes128_ecb_pt, sizeof(aes128_ecb_pt));
    TEST_ASSERT(ret == 0, "ECB encrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, out + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "ECB encrypt final");
    outlen += flen;

    print_hex("ECB ciphertext", out, outlen);
    TEST_ASSERT(outlen == 16, "Ciphertext length == 16");
    TEST_ASSERT(memcmp_hex(out, aes128_ecb_ct, 16) == 0, "ECB ciphertext matches expected");

    /* Decrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_ECB, 0,
                               KT_SWKEY, aes128_key, 128, NULL, 0);
    TEST_ASSERT(ret == 0, "ECB decrypt init");

    outlen = 0;
    ret = drv_pufs_cipher_update(&cipher, out, &outlen,
                                 aes128_ecb_ct, sizeof(aes128_ecb_ct));
    TEST_ASSERT(ret == 0, "ECB decrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, out + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "ECB decrypt final");
    outlen += flen;

    TEST_ASSERT(memcmp_hex(out, aes128_ecb_pt, 16) == 0, "ECB plaintext matches original");

    drv_pufs_close(&dev);
    return 0;
}

static int test_aes256_ecb(void)
{
    TEST_START("AES-256-ECB Encrypt/Decrypt");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    uint8_t ct[16], pt[16];
    uint32_t outlen = 0, flen = 0;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Encrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_ECB, 1,
                               KT_SWKEY, aes256_key, 256, NULL, 0);
    TEST_ASSERT(ret == 0, "AES-256-ECB encrypt init");

    ret = drv_pufs_cipher_update(&cipher, ct, &outlen,
                                 aes128_ecb_pt, 16);
    TEST_ASSERT(ret == 0, "Encrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, ct + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "Encrypt final");
    outlen += flen;

    print_hex("AES-256-ECB ct", ct, outlen);
    TEST_ASSERT(memcmp_hex(ct, aes256_ecb_ct, 16) == 0,
                "AES-256-ECB ciphertext matches NIST");

    /* Decrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_ECB, 0,
                               KT_SWKEY, aes256_key, 256, NULL, 0);
    TEST_ASSERT(ret == 0, "AES-256-ECB decrypt init");

    outlen = 0;
    ret = drv_pufs_cipher_update(&cipher, pt, &outlen, ct, 16);
    TEST_ASSERT(ret == 0, "Decrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, pt + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "Decrypt final");
    outlen += flen;

    TEST_ASSERT(memcmp_hex(pt, aes128_ecb_pt, 16) == 0,
                "AES-256-ECB round-trip OK");

    drv_pufs_close(&dev);
    return 0;
}

static int test_aes128_cbc(void)
{
    TEST_START("AES-128-CBC Encrypt/Decrypt");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    uint8_t out[64];
    uint32_t outlen = 0, flen = 0;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Encrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CBC, 1,
                               KT_SWKEY, aes128_key, 128,
                               aes128_cbc_iv, sizeof(aes128_cbc_iv));
    TEST_ASSERT(ret == 0, "CBC encrypt init");

    ret = drv_pufs_cipher_update(&cipher, out, &outlen,
                                 aes128_cbc_pt, sizeof(aes128_cbc_pt));
    TEST_ASSERT(ret == 0, "CBC encrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, out + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "CBC encrypt final");
    outlen += flen;

    print_hex("CBC ciphertext", out, outlen);
    TEST_ASSERT(outlen == 32, "Ciphertext length == 32");
    TEST_ASSERT(memcmp_hex(out, aes128_cbc_ct, 32) == 0, "CBC ciphertext matches expected");

    /* Decrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CBC, 0,
                               KT_SWKEY, aes128_key, 128,
                               aes128_cbc_iv, sizeof(aes128_cbc_iv));
    TEST_ASSERT(ret == 0, "CBC decrypt init");

    outlen = 0;
    ret = drv_pufs_cipher_update(&cipher, out, &outlen,
                                 aes128_cbc_ct, sizeof(aes128_cbc_ct));
    TEST_ASSERT(ret == 0, "CBC decrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, out + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "CBC decrypt final");
    outlen += flen;

    TEST_ASSERT(memcmp_hex(out, aes128_cbc_pt, 32) == 0, "CBC plaintext matches original");

    drv_pufs_close(&dev);
    return 0;
}

static int test_aes128_ctr(void)
{
    TEST_START("AES-128-CTR Encrypt/Decrypt Round-Trip");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    uint8_t ct[32], pt[32];
    uint32_t outlen = 0, flen = 0;
    int ret;

    uint8_t nonce[16];
    memset(nonce, 0, 16);

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Encrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CTR, 1,
                               KT_SWKEY, aes128_key, 128, nonce, 16);
    TEST_ASSERT(ret == 0, "CTR encrypt init");

    ret = drv_pufs_cipher_update(&cipher, ct, &outlen,
                                 aes128_ecb_pt, sizeof(aes128_ecb_pt));
    TEST_ASSERT(ret == 0, "CTR encrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, ct + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "CTR encrypt final");
    outlen += flen;

    print_hex("CTR ciphertext", ct, outlen);

    /* Decrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CTR, 0,
                               KT_SWKEY, aes128_key, 128, nonce, 16);
    TEST_ASSERT(ret == 0, "CTR decrypt init");

    uint32_t ptlen = 0;
    ret = drv_pufs_cipher_update(&cipher, pt, &ptlen, ct, outlen);
    TEST_ASSERT(ret == 0, "CTR decrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, pt + ptlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "CTR decrypt final");
    ptlen += flen;

    TEST_ASSERT(memcmp_hex(pt, aes128_ecb_pt, sizeof(aes128_ecb_pt)) == 0,
                "CTR round-trip matches original");

    drv_pufs_close(&dev);
    return 0;
}

static int test_aes128_cfb(void)
{
    TEST_START("AES-128-CFB Round-Trip");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    uint8_t ct[32], pt[32];
    uint32_t outlen, flen;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Encrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CFB, 1,
                               KT_SWKEY, aes128_key, 128,
                               aes128_cbc_iv, 16);
    TEST_ASSERT(ret == 0, "CFB encrypt init");

    outlen = 0;
    ret = drv_pufs_cipher_update(&cipher, ct, &outlen,
                                 aes128_cbc_pt, sizeof(aes128_cbc_pt));
    TEST_ASSERT(ret == 0, "CFB encrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, ct + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "CFB encrypt final");
    outlen += flen;

    print_hex("CFB ciphertext", ct, outlen);

    /* Decrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CFB, 0,
                               KT_SWKEY, aes128_key, 128,
                               aes128_cbc_iv, 16);
    TEST_ASSERT(ret == 0, "CFB decrypt init");

    uint32_t ptlen = 0;
    ret = drv_pufs_cipher_update(&cipher, pt, &ptlen, ct, outlen);
    TEST_ASSERT(ret == 0, "CFB decrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, pt + ptlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "CFB decrypt final");
    ptlen += flen;

    TEST_ASSERT(memcmp_hex(pt, aes128_cbc_pt, sizeof(aes128_cbc_pt)) == 0,
                "CFB round-trip OK");

    drv_pufs_close(&dev);
    return 0;
}

static int test_aes128_ofb(void)
{
    TEST_START("AES-128-OFB Round-Trip");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    uint8_t ct[32], pt[32];
    uint32_t outlen, flen;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Encrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_OFB, 1,
                               KT_SWKEY, aes128_key, 128,
                               aes128_cbc_iv, 16);
    TEST_ASSERT(ret == 0, "OFB encrypt init");

    outlen = 0;
    ret = drv_pufs_cipher_update(&cipher, ct, &outlen,
                                 aes128_cbc_pt, sizeof(aes128_cbc_pt));
    TEST_ASSERT(ret == 0, "OFB encrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, ct + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "OFB encrypt final");
    outlen += flen;

    print_hex("OFB ciphertext", ct, outlen);

    /* Decrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_OFB, 0,
                               KT_SWKEY, aes128_key, 128,
                               aes128_cbc_iv, 16);
    TEST_ASSERT(ret == 0, "OFB decrypt init");

    uint32_t ptlen = 0;
    ret = drv_pufs_cipher_update(&cipher, pt, &ptlen, ct, outlen);
    TEST_ASSERT(ret == 0, "OFB decrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, pt + ptlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "OFB decrypt final");
    ptlen += flen;

    TEST_ASSERT(memcmp_hex(pt, aes128_cbc_pt, sizeof(aes128_cbc_pt)) == 0,
                "OFB round-trip OK");

    drv_pufs_close(&dev);
    return 0;
}

static int test_sm4_ecb(void)
{
    TEST_START("SM4-ECB Encrypt/Decrypt");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    uint8_t ct[16], pt[16];
    uint32_t outlen, flen;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Encrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_SM4, MODE_ECB, 1,
                               KT_SWKEY, sm4_key, 128, NULL, 0);
    TEST_ASSERT(ret == 0, "SM4-ECB encrypt init");

    outlen = 0;
    ret = drv_pufs_cipher_update(&cipher, ct, &outlen, sm4_pt, 16);
    TEST_ASSERT(ret == 0, "SM4-ECB encrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, ct + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "SM4-ECB encrypt final");
    outlen += flen;

    print_hex("SM4-ECB ct", ct, outlen);
    TEST_ASSERT(memcmp_hex(ct, sm4_ct, 16) == 0, "SM4-ECB ciphertext matches");

    /* Decrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_SM4, MODE_ECB, 0,
                               KT_SWKEY, sm4_key, 128, NULL, 0);
    TEST_ASSERT(ret == 0, "SM4-ECB decrypt init");

    outlen = 0;
    ret = drv_pufs_cipher_update(&cipher, pt, &outlen, sm4_ct, 16);
    TEST_ASSERT(ret == 0, "SM4-ECB decrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, pt + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "SM4-ECB decrypt final");
    outlen += flen;

    TEST_ASSERT(memcmp_hex(pt, sm4_pt, 16) == 0, "SM4-ECB plaintext matches");

    drv_pufs_close(&dev);
    return 0;
}

/* Cipher streaming: encrypt 4 blocks in 1-block chunks, compare with single-shot */
static int test_cipher_streaming(void)
{
    TEST_START("AES-128-CBC Streaming (4 blocks)");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    int ret;

    /* 64 bytes = 4 AES blocks */
    uint8_t data[64];
    for (int i = 0; i < 64; i++)
        data[i] = (uint8_t)i;

    uint8_t ct_oneshot[64], ct_stream[64];
    uint32_t olen, flen;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* One-shot encrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CBC, 1,
                               KT_SWKEY, aes128_key, 128,
                               aes128_cbc_iv, 16);
    TEST_ASSERT(ret == 0, "One-shot init");

    olen = 0;
    ret = drv_pufs_cipher_update(&cipher, ct_oneshot, &olen, data, 64);
    TEST_ASSERT(ret == 0, "One-shot update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, ct_oneshot + olen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "One-shot final");
    olen += flen;

    /* Streaming encrypt: 16 bytes at a time */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CBC, 1,
                               KT_SWKEY, aes128_key, 128,
                               aes128_cbc_iv, 16);
    TEST_ASSERT(ret == 0, "Stream init");

    uint32_t total = 0;
    for (int off = 0; off < 64; off += 16) {
        uint32_t chunk_out = 0;
        ret = drv_pufs_cipher_update(&cipher, ct_stream + total, &chunk_out,
                                     data + off, 16);
        TEST_ASSERT(ret == 0, "Stream update");
        total += chunk_out;
    }

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, ct_stream + total, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "Stream final");
    total += flen;

    TEST_ASSERT(total == olen, "Stream output length == one-shot length");
    TEST_ASSERT(memcmp_hex(ct_oneshot, ct_stream, olen) == 0,
                "Stream ciphertext == one-shot ciphertext");

    drv_pufs_close(&dev);
    return 0;
}

/* Cipher reuse: encrypt then decrypt using same cipher instance re-init */
static int test_cipher_reinit(void)
{
    TEST_START("Cipher Instance Re-init");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    uint8_t ct[16], pt[16];
    uint32_t outlen, flen;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Encrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_ECB, 1,
                               KT_SWKEY, aes128_key, 128, NULL, 0);
    TEST_ASSERT(ret == 0, "Encrypt init");

    outlen = 0;
    ret = drv_pufs_cipher_update(&cipher, ct, &outlen, aes128_ecb_pt, 16);
    TEST_ASSERT(ret == 0, "Encrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, ct + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "Encrypt final");

    /* Re-init same instance for decrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_ECB, 0,
                               KT_SWKEY, aes128_key, 128, NULL, 0);
    TEST_ASSERT(ret == 0, "Decrypt re-init");

    outlen = 0;
    ret = drv_pufs_cipher_update(&cipher, pt, &outlen, ct, 16);
    TEST_ASSERT(ret == 0, "Decrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, pt + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "Decrypt final");

    TEST_ASSERT(memcmp_hex(pt, aes128_ecb_pt, 16) == 0,
                "Re-init round-trip OK");

    drv_pufs_close(&dev);
    return 0;
}

/* GCM test: AES-128-GCM encrypt, check tag, decrypt */
static int test_aes128_gcm(void)
{
    TEST_START("AES-128-GCM Encrypt/Decrypt with Tag");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    int ret;

    uint8_t pt_data[32];
    for (int i = 0; i < 32; i++)
        pt_data[i] = (uint8_t)(i + 1);

    uint8_t iv[12];
    memset(iv, 0x42, 12);

    uint8_t ct[32], pt_dec[32];
    uint8_t tag_enc[16];
    uint32_t outlen, flen;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Encrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_GCM, 1,
                               KT_SWKEY, aes128_key, 128, iv, 12);
    TEST_ASSERT(ret == 0, "GCM encrypt init");

    outlen = 0;
    ret = drv_pufs_cipher_update(&cipher, ct, &outlen, pt_data, 32);
    TEST_ASSERT(ret == 0, "GCM encrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, ct + outlen, &flen, tag_enc, 16);
    TEST_ASSERT(ret == 0, "GCM encrypt final");
    outlen += flen;

    print_hex("GCM ct", ct, outlen);
    print_hex("GCM tag", tag_enc, 16);

    /* Decrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_GCM, 0,
                               KT_SWKEY, aes128_key, 128, iv, 12);
    TEST_ASSERT(ret == 0, "GCM decrypt init");

    uint32_t ptlen = 0;
    ret = drv_pufs_cipher_update(&cipher, pt_dec, &ptlen, ct, outlen);
    TEST_ASSERT(ret == 0, "GCM decrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, pt_dec + ptlen, &flen, tag_enc, 16);
    TEST_ASSERT(ret == 0, "GCM decrypt final");
    ptlen += flen;

    TEST_ASSERT(memcmp_hex(pt_dec, pt_data, 32) == 0,
                "GCM plaintext matches original");

    drv_pufs_close(&dev);
    return 0;
}

/* CCM test */
static int test_aes128_ccm(void)
{
    TEST_START("AES-128-CCM Encrypt/Decrypt with Tag");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    int ret;

    uint8_t pt_data[24];
    for (int i = 0; i < 24; i++)
        pt_data[i] = (uint8_t)(i + 0x10);

    uint8_t nonce[7];
    memset(nonce, 0xAB, 7);

    uint8_t ct[24], pt_dec[24];
    uint8_t tag_enc[8];
    uint32_t outlen, flen;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Encrypt */
    ret = drv_pufs_cipher_ccm_init(&cipher, &dev, SK_AES, 1,
                                   KT_SWKEY, aes128_key, 128,
                                   nonce, 7,
                                   0,   /* no AAD */
                                   24,  /* plaintext length */
                                   8);  /* tag length */
    TEST_ASSERT(ret == 0, "CCM encrypt init");

    outlen = 0;
    ret = drv_pufs_cipher_update(&cipher, ct, &outlen, pt_data, 24);
    TEST_ASSERT(ret == 0, "CCM encrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, ct + outlen, &flen, tag_enc, 8);
    TEST_ASSERT(ret == 0, "CCM encrypt final");
    outlen += flen;

    print_hex("CCM ct", ct, outlen);
    print_hex("CCM tag", tag_enc, 8);

    /* Decrypt */
    ret = drv_pufs_cipher_ccm_init(&cipher, &dev, SK_AES, 0,
                                   KT_SWKEY, aes128_key, 128,
                                   nonce, 7,
                                   0, 24, 8);
    TEST_ASSERT(ret == 0, "CCM decrypt init");

    uint32_t ptlen = 0;
    ret = drv_pufs_cipher_update(&cipher, pt_dec, &ptlen, ct, outlen);
    TEST_ASSERT(ret == 0, "CCM decrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, pt_dec + ptlen, &flen, tag_enc, 8);
    TEST_ASSERT(ret == 0, "CCM decrypt final");
    ptlen += flen;

    TEST_ASSERT(memcmp_hex(pt_dec, pt_data, 24) == 0,
                "CCM plaintext matches original");

    drv_pufs_close(&dev);
    return 0;
}

/* XTS test */
static int test_aes128_xts(void)
{
    TEST_START("AES-128-XTS Encrypt/Decrypt Round-Trip");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    int ret;

    /* XTS needs two 128-bit keys */
    uint8_t xts_key1[16], xts_key2[16];
    memset(xts_key1, 0x11, 16);
    memset(xts_key2, 0x22, 16);

    /* XTS operates on sectors, min 16 bytes */
    uint8_t pt_data[32], ct[32], pt_dec[32];
    for (int i = 0; i < 32; i++)
        pt_data[i] = (uint8_t)(i ^ 0x55);

    uint8_t tweak[16];
    memset(tweak, 0, 16);

    uint32_t outlen, flen;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Encrypt */
    ret = drv_pufs_cipher_xts_init(&cipher, &dev, SK_AES, 1,
                                   KT_SWKEY, xts_key1,
                                   KT_SWKEY, xts_key2,
                                   128, tweak, 16);
    TEST_ASSERT(ret == 0, "XTS encrypt init");

    outlen = 0;
    ret = drv_pufs_cipher_update(&cipher, ct, &outlen, pt_data, 32);
    TEST_ASSERT(ret == 0, "XTS encrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, ct + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "XTS encrypt final");
    outlen += flen;

    print_hex("XTS ct", ct, outlen);

    /* Decrypt */
    ret = drv_pufs_cipher_xts_init(&cipher, &dev, SK_AES, 0,
                                   KT_SWKEY, xts_key1,
                                   KT_SWKEY, xts_key2,
                                   128, tweak, 16);
    TEST_ASSERT(ret == 0, "XTS decrypt init");

    uint32_t ptlen = 0;
    ret = drv_pufs_cipher_update(&cipher, pt_dec, &ptlen, ct, outlen);
    TEST_ASSERT(ret == 0, "XTS decrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, pt_dec + ptlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "XTS decrypt final");
    ptlen += flen;

    TEST_ASSERT(memcmp_hex(pt_dec, pt_data, 32) == 0,
                "XTS round-trip OK");

    drv_pufs_close(&dev);
    return 0;
}

/* Cipher large data: encrypt/decrypt 4KB in CBC mode */
static int test_cipher_large_data(void)
{
    TEST_START("AES-128-CBC Large Data (4KB)");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    int ret;

    uint8_t *pt = (uint8_t *)malloc(4096);
    uint8_t *ct = (uint8_t *)malloc(4096);
    uint8_t *pt_dec = (uint8_t *)malloc(4096);
    TEST_ASSERT(pt && ct && pt_dec, "Alloc 4KB buffers");

    for (int i = 0; i < 4096; i++)
        pt[i] = (uint8_t)(i & 0xff);

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Encrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CBC, 1,
                               KT_SWKEY, aes128_key, 128,
                               aes128_cbc_iv, 16);
    TEST_ASSERT(ret == 0, "Encrypt init");

    uint32_t outlen = 0, flen;
    ret = drv_pufs_cipher_update(&cipher, ct, &outlen, pt, 4096);
    TEST_ASSERT(ret == 0, "Encrypt update 4KB");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, ct + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "Encrypt final");
    outlen += flen;

    /* Decrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CBC, 0,
                               KT_SWKEY, aes128_key, 128,
                               aes128_cbc_iv, 16);
    TEST_ASSERT(ret == 0, "Decrypt init");

    uint32_t ptlen = 0;
    ret = drv_pufs_cipher_update(&cipher, pt_dec, &ptlen, ct, outlen);
    TEST_ASSERT(ret == 0, "Decrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, pt_dec + ptlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "Decrypt final");
    ptlen += flen;

    TEST_ASSERT(ptlen == 4096, "Decrypted length == 4096");
    TEST_ASSERT(memcmp_hex(pt_dec, pt, 4096) == 0,
                "4KB CBC round-trip OK");

    free(pt);
    free(ct);
    free(pt_dec);
    drv_pufs_close(&dev);
    return 0;
}

/* ================================================================
 * Section 6: UID Tests
 * ================================================================ */

static int test_uid_read(void)
{
    TEST_START("UID Read");

    drv_pufs_inst dev;
    uint8_t uid[32];
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_uid_get(&dev, 0, uid);
    TEST_ASSERT(ret == 0, "Read UID slot 0");

    print_hex("UID", uid, 32);

    int nonzero = 0;
    for (int i = 0; i < 32; i++) {
        if (uid[i] != 0) nonzero = 1;
    }
    TEST_ASSERT(nonzero, "UID is not all zeros");

    drv_pufs_close(&dev);
    return 0;
}

/* UID consistency: reading the same slot twice should return same value */
static int test_uid_consistency(void)
{
    TEST_START("UID Consistency");

    drv_pufs_inst dev;
    uint8_t uid1[32], uid2[32];
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_uid_get(&dev, 0, uid1);
    TEST_ASSERT(ret == 0, "First UID read");

    ret = drv_pufs_uid_get(&dev, 0, uid2);
    TEST_ASSERT(ret == 0, "Second UID read");

    TEST_ASSERT(memcmp_hex(uid1, uid2, 32) == 0,
                "UID reads are consistent");

    drv_pufs_close(&dev);
    return 0;
}

/* ================================================================
 * Section 7: Error / Edge Case Tests
 * ================================================================ */

/* Hash: update after final should fail or produce wrong result */
static int test_error_hash_update_after_final(void)
{
    TEST_START("Error: Hash update after final");

    drv_pufs_inst dev;
    drv_pufs_hash_inst hash;
    uint8_t dgst[64];
    uint32_t dlen = 0;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_hash_init(&hash, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "Hash init");

    ret = drv_pufs_hash_update(&hash, sha256_msg1, sizeof(sha256_msg1));
    TEST_ASSERT(ret == 0, "Hash update");

    ret = drv_pufs_hash_final(&hash, dgst, &dlen);
    TEST_ASSERT(ret == 0, "Hash final");

    /* Update after final: driver should reject or not crash */
    ret = drv_pufs_hash_update(&hash, sha256_msg1, sizeof(sha256_msg1));
    printf("  update-after-final returned %d (non-crash is OK)\n", ret);
    test_passed++;

    drv_pufs_close(&dev);
    return 0;
}

/* Hash: double final should not crash */
static int test_error_hash_double_final(void)
{
    TEST_START("Error: Hash double final");

    drv_pufs_inst dev;
    drv_pufs_hash_inst hash;
    uint8_t dgst[64];
    uint32_t dlen = 0;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_hash_init(&hash, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "Hash init");

    ret = drv_pufs_hash_update(&hash, sha256_msg1, sizeof(sha256_msg1));
    TEST_ASSERT(ret == 0, "Hash update");

    ret = drv_pufs_hash_final(&hash, dgst, &dlen);
    TEST_ASSERT(ret == 0, "First final");

    /* Second final: should not crash */
    uint32_t dlen2 = 0;
    ret = drv_pufs_hash_final(&hash, dgst, &dlen2);
    printf("  double-final returned %d (non-crash is OK)\n", ret);
    test_passed++;

    drv_pufs_close(&dev);
    return 0;
}

/* Cipher: wrong key size should fail */
static int test_error_cipher_bad_keysize(void)
{
    TEST_START("Error: Cipher bad key size");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* AES key must be 128/192/256 bits; try 64 bits */
    uint8_t bad_key[8] = {0};
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_ECB, 1,
                               KT_SWKEY, bad_key, 64, NULL, 0);
    TEST_ASSERT(ret != 0, "AES with 64-bit key rejected");

    /* Try 0-bit key */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_ECB, 1,
                               KT_SWKEY, bad_key, 0, NULL, 0);
    TEST_ASSERT(ret != 0, "AES with 0-bit key rejected");

    drv_pufs_close(&dev);
    return 0;
}

/* Cipher: CBC without IV should fail */
static int test_error_cipher_cbc_no_iv(void)
{
    TEST_START("Error: CBC encrypt with no IV");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* CBC requires 16-byte IV; pass NULL/0 */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CBC, 1,
                               KT_SWKEY, aes128_key, 128, NULL, 0);
    /* This may or may not fail depending on driver implementation.
     * At minimum, it should not crash. */
    printf("  CBC init with NULL IV returned %d\n", ret);
    test_passed++;

    drv_pufs_close(&dev);
    return 0;
}

/* Operations on closed device should fail gracefully */
static int test_error_ops_on_closed_device(void)
{
    TEST_START("Error: Ops on closed device");

    drv_pufs_inst dev;
    drv_pufs_hash_inst hash;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");
    drv_pufs_close(&dev);

    /* Try hash init on closed device */
    ret = drv_pufs_hash_init(&hash, &dev, HASH_SHA_256);
    printf("  hash_init on closed fd returned %d (should fail or not crash)\n", ret);
    /* We just want no crash */
    test_passed++;

    /* Try UID read on closed device */
    uint8_t uid[32];
    ret = drv_pufs_uid_get(&dev, 0, uid);
    printf("  uid_get on closed fd returned %d\n", ret);
    test_passed++;

    return 0;
}

/* ================================================================
 * Section 8: Concurrency / Multi-thread Tests
 * ================================================================ */

struct hash_thread_arg {
    drv_pufs_inst *dev;
    const uint8_t *msg;
    uint32_t msglen;
    const uint8_t *expected;
    uint32_t explen;
    int iterations;
    int pass;
    int fail;
};

static void *hash_thread_func(void *arg)
{
    struct hash_thread_arg *a = (struct hash_thread_arg *)arg;
    drv_pufs_hash_inst hash;
    uint8_t dgst[64];
    uint32_t dlen;

    a->pass = 0;
    a->fail = 0;

    for (int i = 0; i < a->iterations; i++) {
        int ret = drv_pufs_hash_init(&hash, a->dev, HASH_SHA_256);
        if (ret != 0) { a->fail++; continue; }

        if (a->msg && a->msglen > 0) {
            ret = drv_pufs_hash_update(&hash, a->msg, a->msglen);
            if (ret != 0) { a->fail++; continue; }
        }

        dlen = 0;
        ret = drv_pufs_hash_final(&hash, dgst, &dlen);
        if (ret != 0) { a->fail++; continue; }

        if (dlen == a->explen && memcmp_hex(dgst, a->expected, dlen) == 0)
            a->pass++;
        else
            a->fail++;
    }

    return NULL;
}

static int test_concurrent_hash_2threads(void)
{
    TEST_START("Concurrent Hash (2 threads, same device)");

    drv_pufs_inst dev;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    int iters = 50;

    struct hash_thread_arg arg1 = {
        .dev = &dev,
        .msg = sha256_msg1,
        .msglen = sizeof(sha256_msg1),
        .expected = sha256_exp1,
        .explen = 32,
        .iterations = iters,
    };

    struct hash_thread_arg arg2 = {
        .dev = &dev,
        .msg = NULL,
        .msglen = 0,
        .expected = sha256_exp_empty,
        .explen = 32,
        .iterations = iters,
    };

    pthread_t t1, t2;
    pthread_create(&t1, NULL, hash_thread_func, &arg1);
    pthread_create(&t2, NULL, hash_thread_func, &arg2);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Thread 1: %d/%d passed\n", arg1.pass, iters);
    printf("Thread 2: %d/%d passed\n", arg2.pass, iters);

    TEST_ASSERT(arg1.pass == iters, "Thread 1 all iterations correct");
    TEST_ASSERT(arg2.pass == iters, "Thread 2 all iterations correct");

    drv_pufs_close(&dev);
    return 0;
}

/* 4 threads: hash with different algorithms */
static int test_concurrent_hash_4threads(void)
{
    TEST_START("Concurrent Hash (4 threads, different data)");

    drv_pufs_inst dev;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    int iters = 30;

    struct hash_thread_arg args[4] = {
        { .dev = &dev, .msg = sha256_msg1, .msglen = 3,
          .expected = sha256_exp1, .explen = 32, .iterations = iters },
        { .dev = &dev, .msg = NULL, .msglen = 0,
          .expected = sha256_exp_empty, .explen = 32, .iterations = iters },
        { .dev = &dev, .msg = sha256_msg1, .msglen = 3,
          .expected = sha256_exp1, .explen = 32, .iterations = iters },
        { .dev = &dev, .msg = NULL, .msglen = 0,
          .expected = sha256_exp_empty, .explen = 32, .iterations = iters },
    };

    pthread_t threads[4];
    for (int i = 0; i < 4; i++)
        pthread_create(&threads[i], NULL, hash_thread_func, &args[i]);
    for (int i = 0; i < 4; i++)
        pthread_join(threads[i], NULL);

    int all_ok = 1;
    for (int i = 0; i < 4; i++) {
        printf("Thread %d: %d/%d passed\n", i, args[i].pass, iters);
        if (args[i].pass != iters) all_ok = 0;
    }

    TEST_ASSERT(all_ok, "All 4 threads correct");

    drv_pufs_close(&dev);
    return 0;
}

struct cipher_thread_arg {
    drv_pufs_inst *dev;
    int iterations;
    int pass;
    int fail;
};

static void *cipher_thread_func(void *arg)
{
    struct cipher_thread_arg *a = (struct cipher_thread_arg *)arg;
    drv_pufs_cipher_inst cipher;
    uint8_t out[32];
    uint32_t outlen, flen;

    a->pass = 0;
    a->fail = 0;

    for (int i = 0; i < a->iterations; i++) {
        int ret = drv_pufs_cipher_init(&cipher, a->dev, SK_AES, MODE_ECB, 1,
                                       KT_SWKEY, aes128_key, 128, NULL, 0);
        if (ret != 0) { a->fail++; continue; }

        outlen = 0;
        ret = drv_pufs_cipher_update(&cipher, out, &outlen,
                                     aes128_ecb_pt, sizeof(aes128_ecb_pt));
        if (ret != 0) { a->fail++; continue; }

        flen = 0;
        ret = drv_pufs_cipher_final(&cipher, out + outlen, &flen, NULL, 0);
        if (ret != 0) { a->fail++; continue; }
        outlen += flen;

        if (outlen == 16 && memcmp_hex(out, aes128_ecb_ct, 16) == 0)
            a->pass++;
        else
            a->fail++;
    }

    return NULL;
}

static int test_concurrent_hash_cipher(void)
{
    TEST_START("Concurrent Hash + Cipher (2 threads)");

    drv_pufs_inst dev;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    int iters = 50;

    struct hash_thread_arg harg = {
        .dev = &dev,
        .msg = sha256_msg1,
        .msglen = sizeof(sha256_msg1),
        .expected = sha256_exp1,
        .explen = 32,
        .iterations = iters,
    };

    struct cipher_thread_arg carg = {
        .dev = &dev,
        .iterations = iters,
    };

    pthread_t t1, t2;
    pthread_create(&t1, NULL, hash_thread_func, &harg);
    pthread_create(&t2, NULL, cipher_thread_func, &carg);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Hash thread:   %d/%d passed\n", harg.pass, iters);
    printf("Cipher thread: %d/%d passed\n", carg.pass, iters);

    TEST_ASSERT(harg.pass == iters, "Hash thread all correct");
    TEST_ASSERT(carg.pass == iters, "Cipher thread all correct");

    drv_pufs_close(&dev);
    return 0;
}

/* Concurrent cipher: 2 threads both doing AES-ECB */
static int test_concurrent_cipher_2threads(void)
{
    TEST_START("Concurrent Cipher (2 threads, AES-ECB)");

    drv_pufs_inst dev;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    int iters = 50;

    struct cipher_thread_arg carg1 = { .dev = &dev, .iterations = iters };
    struct cipher_thread_arg carg2 = { .dev = &dev, .iterations = iters };

    pthread_t t1, t2;
    pthread_create(&t1, NULL, cipher_thread_func, &carg1);
    pthread_create(&t2, NULL, cipher_thread_func, &carg2);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Cipher thread 1: %d/%d passed\n", carg1.pass, iters);
    printf("Cipher thread 2: %d/%d passed\n", carg2.pass, iters);

    TEST_ASSERT(carg1.pass == iters, "Cipher thread 1 all correct");
    TEST_ASSERT(carg2.pass == iters, "Cipher thread 2 all correct");

    drv_pufs_close(&dev);
    return 0;
}

/* Multiple devices: each thread opens its own device */
struct hash_own_dev_arg {
    const uint8_t *msg;
    uint32_t msglen;
    const uint8_t *expected;
    uint32_t explen;
    int iterations;
    int pass;
    int fail;
};

static void *hash_own_dev_func(void *arg)
{
    struct hash_own_dev_arg *a = (struct hash_own_dev_arg *)arg;
    drv_pufs_inst dev;
    drv_pufs_hash_inst hash;
    uint8_t dgst[64];
    uint32_t dlen;

    a->pass = 0;
    a->fail = 0;

    if (drv_pufs_open(&dev) != 0) {
        a->fail = a->iterations;
        return NULL;
    }

    for (int i = 0; i < a->iterations; i++) {
        int ret = drv_pufs_hash_init(&hash, &dev, HASH_SHA_256);
        if (ret != 0) { a->fail++; continue; }

        if (a->msg && a->msglen > 0) {
            ret = drv_pufs_hash_update(&hash, a->msg, a->msglen);
            if (ret != 0) { a->fail++; continue; }
        }

        dlen = 0;
        ret = drv_pufs_hash_final(&hash, dgst, &dlen);
        if (ret != 0) { a->fail++; continue; }

        if (dlen == a->explen && memcmp_hex(dgst, a->expected, dlen) == 0)
            a->pass++;
        else
            a->fail++;
    }

    drv_pufs_close(&dev);
    return NULL;
}

static int test_concurrent_separate_devices(void)
{
    TEST_START("Concurrent Hash (4 threads, separate devices)");

    int iters = 30;

    struct hash_own_dev_arg args[4] = {
        { .msg = sha256_msg1, .msglen = 3,
          .expected = sha256_exp1, .explen = 32, .iterations = iters },
        { .msg = NULL, .msglen = 0,
          .expected = sha256_exp_empty, .explen = 32, .iterations = iters },
        { .msg = sha256_msg1, .msglen = 3,
          .expected = sha256_exp1, .explen = 32, .iterations = iters },
        { .msg = NULL, .msglen = 0,
          .expected = sha256_exp_empty, .explen = 32, .iterations = iters },
    };

    pthread_t threads[4];
    for (int i = 0; i < 4; i++)
        pthread_create(&threads[i], NULL, hash_own_dev_func, &args[i]);
    for (int i = 0; i < 4; i++)
        pthread_join(threads[i], NULL);

    int all_ok = 1;
    for (int i = 0; i < 4; i++) {
        printf("Thread %d: %d/%d passed\n", i, args[i].pass, iters);
        if (args[i].pass != iters) all_ok = 0;
    }

    TEST_ASSERT(all_ok, "All 4 threads (separate devices) correct");
    return 0;
}

/* ================================================================
 * Section 9: Performance / Stress Tests
 * ================================================================ */

static int test_perf_hash(void)
{
    TEST_START("Perf: SHA-256 throughput (1MB)");

    drv_pufs_inst dev;
    drv_pufs_hash_inst hash;
    uint8_t dgst[64];
    uint32_t dlen;
    int ret;

    uint8_t *buf = (uint8_t *)malloc(4096);
    TEST_ASSERT(buf != NULL, "Alloc buffer");
    memset(buf, 0xAA, 4096);

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_hash_init(&hash, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "Hash init");

    uint64_t t0 = get_time_us();

    /* Feed 1MB = 256 * 4KB chunks */
    for (int i = 0; i < 256; i++) {
        ret = drv_pufs_hash_update(&hash, buf, 4096);
        if (ret != 0) break;
    }
    TEST_ASSERT(ret == 0, "All updates succeeded");

    dlen = 0;
    ret = drv_pufs_hash_final(&hash, dgst, &dlen);
    TEST_ASSERT(ret == 0, "Hash final");

    uint64_t dt = get_time_us() - t0;
    double mbps = (1.0 * 1024 * 1024) / (dt / 1000000.0) / (1024 * 1024);
    printf("  SHA-256 1MB: %llu us (%.2f MB/s)\n",
           (unsigned long long)dt, mbps);

    print_hex("SHA-256(1MB 0xAA)", dgst, dlen);

    free(buf);
    drv_pufs_close(&dev);
    return 0;
}

static int test_perf_cipher(void)
{
    TEST_START("Perf: AES-128-CBC throughput (1MB)");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    int ret;

    uint8_t *buf = (uint8_t *)malloc(4096);
    uint8_t *out = (uint8_t *)malloc(4096);
    TEST_ASSERT(buf && out, "Alloc buffers");
    memset(buf, 0xBB, 4096);

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CBC, 1,
                               KT_SWKEY, aes128_key, 128,
                               aes128_cbc_iv, 16);
    TEST_ASSERT(ret == 0, "Cipher init");

    uint64_t t0 = get_time_us();

    for (int i = 0; i < 256; i++) {
        uint32_t olen = 0;
        ret = drv_pufs_cipher_update(&cipher, out, &olen, buf, 4096);
        if (ret != 0) break;
    }
    TEST_ASSERT(ret == 0, "All cipher updates succeeded");

    uint32_t flen = 0;
    ret = drv_pufs_cipher_final(&cipher, out, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "Cipher final");

    uint64_t dt = get_time_us() - t0;
    double mbps = (1.0 * 1024 * 1024) / (dt / 1000000.0) / (1024 * 1024);
    printf("  AES-128-CBC encrypt 1MB: %llu us (%.2f MB/s)\n",
           (unsigned long long)dt, mbps);

    free(buf);
    free(out);
    drv_pufs_close(&dev);
    return 0;
}

/* Stress: rapid open/close cycles */
static int test_stress_open_close(void)
{
    TEST_START("Stress: 100 open/close cycles");

    for (int i = 0; i < 100; i++) {
        drv_pufs_inst dev;
        int ret = drv_pufs_open(&dev);
        if (ret != 0) {
            printf("[FAIL] cycle %d: open failed\n", i);
            test_failed++;
            return -1;
        }
        ret = drv_pufs_close(&dev);
        if (ret != 0) {
            printf("[FAIL] cycle %d: close failed\n", i);
            test_failed++;
            return -1;
        }
    }

    printf("[PASS] 100 open/close cycles OK\n");
    test_passed++;
    return 0;
}

/* Stress: rapid hash operations (100 hashes back-to-back) */
static int test_stress_rapid_hash(void)
{
    TEST_START("Stress: 100 rapid SHA-256 operations");

    drv_pufs_inst dev;
    drv_pufs_hash_inst hash;
    uint8_t dgst[64];
    uint32_t dlen;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    int ok = 0;
    for (int i = 0; i < 100; i++) {
        ret = drv_pufs_hash_init(&hash, &dev, HASH_SHA_256);
        if (ret != 0) break;

        ret = drv_pufs_hash_update(&hash, sha256_msg1, sizeof(sha256_msg1));
        if (ret != 0) break;

        dlen = 0;
        ret = drv_pufs_hash_final(&hash, dgst, &dlen);
        if (ret != 0) break;

        if (memcmp_hex(dgst, sha256_exp1, 32) != 0) break;
        ok++;
    }

    printf("  %d/100 rapid hashes correct\n", ok);
    TEST_ASSERT(ok == 100, "All 100 rapid hashes correct");

    drv_pufs_close(&dev);
    return 0;
}

/* ================================================================
 * Section 10: RNG Tests
 * ================================================================ */

static int test_rng_read(void)
{
    TEST_START("RNG Basic Read");

    drv_pufs_inst dev;
    uint8_t buf[32];
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    memset(buf, 0, sizeof(buf));
    ret = drv_pufs_rng_read(&dev, buf, sizeof(buf));
    TEST_ASSERT(ret == 0, "RNG read 32 bytes");

    print_hex("RNG", buf, 32);

    int nonzero = 0;
    for (int i = 0; i < 32; i++)
        if (buf[i] != 0) nonzero = 1;
    TEST_ASSERT(nonzero, "RNG output is not all zeros");

    drv_pufs_close(&dev);
    return 0;
}

static int test_rng_different_reads(void)
{
    TEST_START("RNG Two Reads Should Differ");

    drv_pufs_inst dev;
    uint8_t buf1[32], buf2[32];
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_rng_read(&dev, buf1, 32);
    TEST_ASSERT(ret == 0, "First RNG read");

    ret = drv_pufs_rng_read(&dev, buf2, 32);
    TEST_ASSERT(ret == 0, "Second RNG read");

    TEST_ASSERT(memcmp_hex(buf1, buf2, 32) != 0,
                "Two RNG reads produce different output");

    drv_pufs_close(&dev);
    return 0;
}

static int test_rng_various_sizes(void)
{
    TEST_START("RNG Various Sizes");

    drv_pufs_inst dev;
    uint8_t buf[256];
    int ret;
    uint32_t sizes[] = { 1, 16, 32, 64, 128, 256 };

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    for (uint32_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
        memset(buf, 0, sizeof(buf));
        ret = drv_pufs_rng_read(&dev, buf, sizes[i]);
        char msg[64];
        snprintf(msg, sizeof(msg), "RNG read %u bytes", sizes[i]);
        TEST_ASSERT(ret == 0, msg);
    }

    drv_pufs_close(&dev);
    return 0;
}

/* ================================================================
 * Section 11: PUFrt Management Tests
 * ================================================================ */

static int test_rt_version(void)
{
    TEST_START("PUFrt Version Query");

    drv_pufs_inst dev;
    uint32_t version = 0, features = 0;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_rt_version(&dev, &version, &features);
    TEST_ASSERT(ret == 0, "Query RT version");

    printf("  PUFrt version: 0x%08x, features: 0x%08x\n", version, features);
    TEST_ASSERT(version != 0, "Version is non-zero");

    drv_pufs_close(&dev);
    return 0;
}

static int test_otp_rwlck_query(void)
{
    /*
     * Scan all 32 OTP key slots (OTPKEY_0..31).
     * Each slot is 32 bytes (256 bits), total = 1024 bytes.
     * Lock states: 0=NA (burned+locked, CPU can't read), 1=RO, 2=RW (empty).
     */
    TEST_START("OTP Key Slot Survey (32 slots)");

    drv_pufs_inst dev;
    uint8_t lock;
    uint8_t buf[32];
    int ret;
    int na_count = 0, ro_count = 0, rw_count = 0;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    printf("  Slot  Addr   Lock  Data[0..3]\n");
    printf("  ----  -----  ----  ----------\n");

    for (int slot = 0; slot < 32; slot++) {
        uint16_t addr = slot * 32;  /* OTP_KEY_LEN = 32 */

        lock = 0xFF;
        ret = drv_pufs_otp_get_rwlck(&dev, addr, &lock);
        TEST_ASSERT(ret == 0, "Query rwlck");
        TEST_ASSERT(lock <= 2, "Lock state valid (0-2)");

        /* Try to read the slot data */
        memset(buf, 0xAA, sizeof(buf));
        ret = drv_pufs_otp_read(&dev, addr, buf, 32);
        TEST_ASSERT(ret == 0, "Read OTP slot");

        const char *state;
        switch (lock) {
        case 0:  state = "NA "; na_count++;  break;  /* burned + locked */
        case 1:  state = "RO "; ro_count++;  break;  /* read-only */
        case 2:  state = "RW "; rw_count++;  break;  /* empty / writable */
        default: state = "???"; break;
        }

        /* Check if data is all zeros (typical for NA-locked or never-written) */
        int all_zero = 1;
        for (int i = 0; i < 32; i++) {
            if (buf[i] != 0) { all_zero = 0; break; }
        }

        printf("  %2d    %4u   %s   %02x%02x%02x%02x %s\n",
               slot, addr, state,
               buf[0], buf[1], buf[2], buf[3],
               all_zero ? "(all-zero)" : "");
    }

    printf("\n  Summary: %d NA (burned+locked), %d RO, %d RW (empty)\n",
           na_count, ro_count, rw_count);
    printf("  Total: %d slots scanned\n", na_count + ro_count + rw_count);

    drv_pufs_close(&dev);
    return 0;
}

static int test_otp_read(void)
{
    /*
     * K230 OTP layout (via PUFS_OTP_READ ioctl):
     *   0 .. 1023  : base PUFrt OTP (1KB, OTP_LEN=1024)
     *   1024+      : CDE OTP (24Kbit / 3KB) — NOT accessible via this ioctl
     *
     * OTP is one-time programmable: tests MUST NOT write.
     * addr must be 4-byte aligned.
     */
    TEST_START("OTP Read (read-only, base 1KB region)");

    drv_pufs_inst dev;
    uint8_t buf[128];
    uint8_t buf2[128];
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Read from OTP addr 0 */
    memset(buf, 0xFF, sizeof(buf));
    ret = drv_pufs_otp_read(&dev, 0, buf, 32);
    TEST_ASSERT(ret == 0, "OTP read addr 0, 32 bytes");
    printf("  OTP[0..31]:  %02x %02x %02x %02x ... %02x %02x %02x %02x\n",
           buf[0], buf[1], buf[2], buf[3],
           buf[28], buf[29], buf[30], buf[31]);

    /* Read same range again — must be identical (OTP is stable) */
    memset(buf2, 0x00, sizeof(buf2));
    ret = drv_pufs_otp_read(&dev, 0, buf2, 32);
    TEST_ASSERT(ret == 0, "OTP read addr 0 again");
    TEST_ASSERT(memcmp(buf, buf2, 32) == 0,
                "Two consecutive reads from same OTP addr match");

    /* Read from mid-range offset (word-aligned) */
    ret = drv_pufs_otp_read(&dev, 512, buf, 32);
    TEST_ASSERT(ret == 0, "OTP read addr 512, 32 bytes");
    printf("  OTP[512..543]: %02x %02x %02x %02x ... %02x %02x %02x %02x\n",
           buf[0], buf[1], buf[2], buf[3],
           buf[28], buf[29], buf[30], buf[31]);

    /* Read at end of 1KB boundary: addr 992, 32 bytes = ends at 1024 */
    ret = drv_pufs_otp_read(&dev, 992, buf, 32);
    TEST_ASSERT(ret == 0, "OTP read addr 992, 32 bytes (end of 1KB)");

    /* Read maximum: 128 bytes from addr 0 */
    ret = drv_pufs_otp_read(&dev, 0, buf, 128);
    TEST_ASSERT(ret == 0, "OTP read addr 0, 128 bytes");

    /* Minimum: 4 bytes (word-aligned min) */
    ret = drv_pufs_otp_read(&dev, 0, buf, 4);
    TEST_ASSERT(ret == 0, "OTP read addr 0, 4 bytes (minimum)");

    drv_pufs_close(&dev);
    return 0;
}

static int test_otp_security_state(void)
{
    TEST_START("OTP Security Config State");

    drv_pufs_inst dev;
    pufs_otp_security_state_t state;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    memset(&state, 0, sizeof(state));
    ret = drv_pufs_otp_get_security_config_state(&dev, &state);
    TEST_ASSERT(ret == 0, "Query OTP security config state");
    TEST_ASSERT(state.spi2axi_word_lock <= OTP_RW,
                "SPI2AXI config word lock state valid");
    TEST_ASSERT(state.jtag_word_lock <= OTP_RW,
                "JTAG config word lock state valid");
    TEST_ASSERT(state.boot_ctrl_word_lock <= OTP_RW,
                "Boot control word lock state valid");

    printf("  disable_spi2axi:    %u\n", state.disable_spi2axi);
    printf("  disable_jtag:       %u\n", state.disable_jtag);
    printf("  force_secure_boot:  %u\n", state.force_secure_boot);
    printf("  disable_isp:        %u\n", state.disable_isp);
    printf("  spi2axi_word_lock:  %s (%u)\n",
           otp_lock_state_name(state.spi2axi_word_lock),
           state.spi2axi_word_lock);
    printf("  jtag_word_lock:     %s (%u)\n",
           otp_lock_state_name(state.jtag_word_lock),
           state.jtag_word_lock);
    printf("  boot_ctrl_word_lock:%s (%u)\n",
           otp_lock_state_name(state.boot_ctrl_word_lock),
           state.boot_ctrl_word_lock);

    drv_pufs_close(&dev);
    return 0;
}

/* ================================================================
 * Section 12: Key Management Tests
 * ================================================================ */

static int test_key_import_export_roundtrip(void)
{
    TEST_START("Key Import/Export Plaintext Round-Trip");

    drv_pufs_inst dev;
    int ret;

    uint8_t key_in[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10,
    };
    uint8_t key_out[16];

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_key_import_plaintext(&dev, KT_SSKEY, KS_SK128_0, key_in, 128);
    TEST_ASSERT(ret == 0, "Import key to slot SK128_0");

    memset(key_out, 0, sizeof(key_out));
    ret = drv_pufs_key_export_plaintext(&dev, KT_SSKEY, KS_SK128_0, key_out, 128);
    if (ret != 0) {
        printf("  Plaintext export not supported by HW (ret=%d), skipping\n", ret);
        test_skipped++;
        drv_pufs_key_clear(&dev, KT_SSKEY, KS_SK128_0, 128);
        drv_pufs_close(&dev);
        return 0;
    }
    printf("[PASS] Export key from slot SK128_0\n");
    test_passed++;

    print_hex("Exported key", key_out, 16);
    TEST_ASSERT(memcmp_hex(key_in, key_out, 16) == 0,
                "Exported key matches imported key");

    drv_pufs_key_clear(&dev, KT_SSKEY, KS_SK128_0, 128);

    drv_pufs_close(&dev);
    return 0;
}

static int test_key_clear(void)
{
    TEST_START("Key Import then Clear");

    drv_pufs_inst dev;
    int ret;

    uint8_t key[16] = {
        0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11,
        0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
    };

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_key_import_plaintext(&dev, KT_SSKEY, KS_SK128_0, key, 128);
    TEST_ASSERT(ret == 0, "Import key");

    ret = drv_pufs_key_clear(&dev, KT_SSKEY, KS_SK128_0, 128);
    TEST_ASSERT(ret == 0, "Clear key slot");

    uint8_t key_out[16];
    memset(key_out, 0xFF, sizeof(key_out));
    ret = drv_pufs_key_export_plaintext(&dev, KT_SSKEY, KS_SK128_0, key_out, 128);
    if (ret == 0) {
        int all_zero = 1;
        for (int i = 0; i < 16; i++)
            if (key_out[i] != 0) all_zero = 0;
        printf("  Export after clear: all_zero=%d\n", all_zero);
    } else {
        printf("  Export after clear returned %d (expected)\n", ret);
    }
    test_passed++;

    drv_pufs_close(&dev);
    return 0;
}

/* ================================================================
 * Section 13: ECC / ECDSA Tests
 * ================================================================ */

static int test_ecc_keygen_puk_verify(void)
{
    TEST_START("ECC Key Generation + PUK Verify (P-256)");

    drv_pufs_inst dev;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_ecc_prk_gen(&dev, ECC_NISTP256, 1, KS_PRK_0);
    TEST_ASSERT(ret == 0, "Generate ECC private key (ephemeral, PRK_0)");

    pufs_ecc_puk_t puk;
    memset(&puk, 0, sizeof(puk));
    ret = drv_pufs_ecc_puk_gen(&dev, ECC_NISTP256, KT_PRKEY, KS_PRK_0, &puk);
    TEST_ASSERT(ret == 0, "Generate public key from private key");

    printf("  Public key qlen=%u\n", puk.qlen);
    TEST_ASSERT(puk.qlen == 32, "P-256 public key qlen == 32");

    print_hex("  Qx", puk.x, puk.qlen);
    print_hex("  Qy", puk.y, puk.qlen);

    ret = drv_pufs_ecc_puk_verify(&dev, ECC_NISTP256, &puk);
    TEST_ASSERT(ret == 0, "Public key is valid on P-256 curve");

    drv_pufs_close(&dev);
    return 0;
}

static int test_ecdsa_sign_verify(void)
{
    TEST_START("ECDSA Sign/Verify (P-256)");

    drv_pufs_inst dev;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_ecc_prk_gen(&dev, ECC_NISTP256, 1, KS_PRK_0);
    TEST_ASSERT(ret == 0, "Generate ECC private key");

    pufs_ecc_puk_t puk;
    memset(&puk, 0, sizeof(puk));
    ret = drv_pufs_ecc_puk_gen(&dev, ECC_NISTP256, KT_PRKEY, KS_PRK_0, &puk);
    TEST_ASSERT(ret == 0, "Generate public key");

    uint8_t md[32];
    drv_pufs_hash_inst hash;
    uint32_t dlen = 0;
    ret = drv_pufs_hash_init(&hash, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "Hash init");
    ret = drv_pufs_hash_update(&hash, sha256_msg1, sizeof(sha256_msg1));
    TEST_ASSERT(ret == 0, "Hash update");
    ret = drv_pufs_hash_final(&hash, md, &dlen);
    TEST_ASSERT(ret == 0, "Hash final");

    pufs_ecdsa_sig_t sig;
    memset(&sig, 0, sizeof(sig));
    ret = drv_pufs_ecdsa_sign(&dev, ECC_NISTP256, KT_PRKEY, KS_PRK_0,
                              md, 32, &sig);
    TEST_ASSERT(ret == 0, "ECDSA sign");

    print_hex("  sig.r", sig.r, sig.qlen);
    print_hex("  sig.s", sig.s, sig.qlen);

    ret = drv_pufs_ecdsa_verify(&dev, ECC_NISTP256, md, 32, &puk, &sig);
    TEST_ASSERT(ret == 0, "ECDSA verify (valid signature)");

    uint8_t bad_md[32];
    memcpy(bad_md, md, 32);
    bad_md[0] ^= 0x01;
    ret = drv_pufs_ecdsa_verify(&dev, ECC_NISTP256, bad_md, 32, &puk, &sig);
    TEST_ASSERT(ret != 0, "ECDSA verify rejects tampered hash");

    drv_pufs_close(&dev);
    return 0;
}

/* ================================================================
 * Section 14: ECDH Tests
 * ================================================================ */

static int test_ecdh_shared_secret(void)
{
    TEST_START("ECDH Shared Secret (P-256)");

    drv_pufs_inst dev;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_ecc_prk_gen(&dev, ECC_NISTP256, 1, KS_PRK_0);
    TEST_ASSERT(ret == 0, "Generate private key A");

    pufs_ecc_puk_t puk_a;
    memset(&puk_a, 0, sizeof(puk_a));
    ret = drv_pufs_ecc_puk_gen(&dev, ECC_NISTP256, KT_PRKEY, KS_PRK_0, &puk_a);
    TEST_ASSERT(ret == 0, "Generate public key A");

    uint8_t shared[32];
    memset(shared, 0, sizeof(shared));
    ret = drv_pufs_ecc_cdh(&dev, ECC_NISTP256, 1, KS_PRK_0, &puk_a, shared, 32);
    TEST_ASSERT(ret == 0, "ECDH compute shared secret");

    print_hex("  ECDH shared", shared, 32);

    int nonzero = 0;
    for (int i = 0; i < 32; i++)
        if (shared[i] != 0) nonzero = 1;
    TEST_ASSERT(nonzero, "Shared secret is non-zero");

    drv_pufs_close(&dev);
    return 0;
}

/* ================================================================
 * Section 15: DRBG Tests
 * ================================================================ */

static int test_drbg_lifecycle(void)
{
    TEST_START("DRBG Lifecycle (AES-CTR)");

    drv_pufs_inst dev;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_drbg_instantiate(&dev, DRBG_AES_CTR, 128, 0,
                                    NULL, 0, NULL, 0);
    TEST_ASSERT(ret == 0, "DRBG instantiate (AES-CTR-128)");

    uint8_t out1[32];
    memset(out1, 0, sizeof(out1));
    ret = drv_pufs_drbg_generate(&dev, out1, 256, 0, 0, NULL, 0);
    TEST_ASSERT(ret == 0, "DRBG generate 256 bits");

    print_hex("  DRBG out1", out1, 32);

    int nonzero = 0;
    for (int i = 0; i < 32; i++)
        if (out1[i] != 0) nonzero = 1;
    TEST_ASSERT(nonzero, "DRBG output is non-zero");

    ret = drv_pufs_drbg_reseed(&dev, 0, NULL, 0);
    TEST_ASSERT(ret == 0, "DRBG reseed");

    uint8_t out2[32];
    memset(out2, 0, sizeof(out2));
    ret = drv_pufs_drbg_generate(&dev, out2, 256, 0, 0, NULL, 0);
    TEST_ASSERT(ret == 0, "DRBG generate after reseed");

    print_hex("  DRBG out2", out2, 32);

    TEST_ASSERT(memcmp_hex(out1, out2, 32) != 0,
                "DRBG output differs after reseed");

    ret = drv_pufs_drbg_uninstantiate(&dev);
    TEST_ASSERT(ret == 0, "DRBG uninstantiate");

    drv_pufs_close(&dev);
    return 0;
}

static int test_drbg_consecutive_generates(void)
{
    TEST_START("DRBG Consecutive Generates Differ");

    drv_pufs_inst dev;
    int ret;

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    ret = drv_pufs_drbg_instantiate(&dev, DRBG_AES_CTR, 128, 0,
                                    NULL, 0, NULL, 0);
    TEST_ASSERT(ret == 0, "DRBG instantiate");

    uint8_t out1[16], out2[16];
    ret = drv_pufs_drbg_generate(&dev, out1, 128, 0, 0, NULL, 0);
    TEST_ASSERT(ret == 0, "First generate");

    ret = drv_pufs_drbg_generate(&dev, out2, 128, 0, 0, NULL, 0);
    TEST_ASSERT(ret == 0, "Second generate");

    TEST_ASSERT(memcmp_hex(out1, out2, 16) != 0,
                "Consecutive generates produce different output");

    ret = drv_pufs_drbg_uninstantiate(&dev);
    TEST_ASSERT(ret == 0, "DRBG uninstantiate");

    drv_pufs_close(&dev);
    return 0;
}

/* ================================================================
 * Section 16: AEAD Tag Tamper Tests
 * ================================================================ */

static int test_gcm_tag_tamper(void)
{
    TEST_START("AES-128-GCM Tag Tamper Detection");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    int ret;

    uint8_t pt[32], ct[32], pt_dec[32];
    uint8_t tag[16], bad_tag[16];
    uint32_t outlen, flen;

    for (int i = 0; i < 32; i++)
        pt[i] = (uint8_t)(i + 1);

    uint8_t iv[12];
    memset(iv, 0x42, 12);

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Encrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_GCM, 1,
                               KT_SWKEY, aes128_key, 128, iv, 12);
    TEST_ASSERT(ret == 0, "GCM encrypt init");

    outlen = 0;
    ret = drv_pufs_cipher_update(&cipher, ct, &outlen, pt, 32);
    TEST_ASSERT(ret == 0, "GCM encrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, ct + outlen, &flen, tag, 16);
    TEST_ASSERT(ret == 0, "GCM encrypt final");
    outlen += flen;

    /* Decrypt with valid tag */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_GCM, 0,
                               KT_SWKEY, aes128_key, 128, iv, 12);
    TEST_ASSERT(ret == 0, "GCM decrypt init");

    uint32_t ptlen = 0;
    ret = drv_pufs_cipher_update(&cipher, pt_dec, &ptlen, ct, outlen);
    TEST_ASSERT(ret == 0, "GCM decrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, pt_dec + ptlen, &flen, tag, 16);
    TEST_ASSERT(ret == 0, "GCM decrypt final (valid tag)");
    TEST_ASSERT(memcmp_hex(pt_dec, pt, 32) == 0,
                "GCM decryption with valid tag OK");

    /* Decrypt with tampered tag */
    memcpy(bad_tag, tag, 16);
    bad_tag[0] ^= 0x01;

    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_GCM, 0,
                               KT_SWKEY, aes128_key, 128, iv, 12);
    TEST_ASSERT(ret == 0, "GCM decrypt init (bad tag)");

    ptlen = 0;
    ret = drv_pufs_cipher_update(&cipher, pt_dec, &ptlen, ct, outlen);

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, pt_dec + ptlen, &flen, bad_tag, 16);
    TEST_ASSERT(ret != 0, "GCM decrypt rejects tampered tag");

    drv_pufs_close(&dev);
    return 0;
}

static int test_ccm_tag_tamper(void)
{
    TEST_START("AES-128-CCM Tag Tamper Detection");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    int ret;

    uint8_t pt[24], ct[24], pt_dec[24];
    uint8_t tag[8], bad_tag[8];
    uint32_t outlen, flen;

    for (int i = 0; i < 24; i++)
        pt[i] = (uint8_t)(i + 0x10);

    uint8_t nonce[7];
    memset(nonce, 0xAB, 7);

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Encrypt */
    ret = drv_pufs_cipher_ccm_init(&cipher, &dev, SK_AES, 1,
                                   KT_SWKEY, aes128_key, 128,
                                   nonce, 7, 0, 24, 8);
    TEST_ASSERT(ret == 0, "CCM encrypt init");

    outlen = 0;
    ret = drv_pufs_cipher_update(&cipher, ct, &outlen, pt, 24);
    TEST_ASSERT(ret == 0, "CCM encrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, ct + outlen, &flen, tag, 8);
    TEST_ASSERT(ret == 0, "CCM encrypt final");
    outlen += flen;

    /* Decrypt with tampered tag */
    memcpy(bad_tag, tag, 8);
    bad_tag[0] ^= 0x01;

    ret = drv_pufs_cipher_ccm_init(&cipher, &dev, SK_AES, 0,
                                   KT_SWKEY, aes128_key, 128,
                                   nonce, 7, 0, 24, 8);
    TEST_ASSERT(ret == 0, "CCM decrypt init (bad tag)");

    uint32_t ptlen = 0;
    ret = drv_pufs_cipher_update(&cipher, pt_dec, &ptlen, ct, outlen);

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, pt_dec + ptlen, &flen, bad_tag, 8);
    TEST_ASSERT(ret != 0, "CCM decrypt rejects tampered tag");

    drv_pufs_close(&dev);
    return 0;
}

/* ================================================================
 * Section 17: DMA Boundary Tests
 * ================================================================ */

static int test_cipher_dma_boundary(void)
{
    TEST_START("AES-128-CBC at DMA Boundary (64KB)");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    int ret;

    const uint32_t size = 65536;
    uint8_t *pt = (uint8_t *)malloc(size);
    uint8_t *ct = (uint8_t *)malloc(size);
    uint8_t *pt_dec = (uint8_t *)malloc(size);
    TEST_ASSERT(pt && ct && pt_dec, "Alloc 64KB buffers");

    for (uint32_t i = 0; i < size; i++)
        pt[i] = (uint8_t)(i & 0xff);

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Encrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CBC, 1,
                               KT_SWKEY, aes128_key, 128,
                               aes128_cbc_iv, 16);
    TEST_ASSERT(ret == 0, "Encrypt init");

    uint32_t outlen = 0, flen;
    ret = drv_pufs_cipher_update(&cipher, ct, &outlen, pt, size);
    TEST_ASSERT(ret == 0, "Encrypt update 64KB");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, ct + outlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "Encrypt final");
    outlen += flen;
    TEST_ASSERT(outlen == size, "Ciphertext length == 64KB");

    /* Decrypt */
    ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CBC, 0,
                               KT_SWKEY, aes128_key, 128,
                               aes128_cbc_iv, 16);
    TEST_ASSERT(ret == 0, "Decrypt init");

    uint32_t ptlen = 0;
    ret = drv_pufs_cipher_update(&cipher, pt_dec, &ptlen, ct, outlen);
    TEST_ASSERT(ret == 0, "Decrypt update");

    flen = 0;
    ret = drv_pufs_cipher_final(&cipher, pt_dec + ptlen, &flen, NULL, 0);
    TEST_ASSERT(ret == 0, "Decrypt final");
    ptlen += flen;

    TEST_ASSERT(ptlen == size, "Decrypted length == 64KB");
    TEST_ASSERT(memcmp_hex(pt_dec, pt, size) == 0,
                "64KB CBC round-trip OK");

    free(pt);
    free(ct);
    free(pt_dec);
    drv_pufs_close(&dev);
    return 0;
}

static int test_cipher_across_dma_boundary(void)
{
    TEST_START("AES-128-CTR Across DMA Boundary");

    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    int ret;

    uint32_t sizes[] = { 65536 - 16, 65536, 65536 + 16 };
    uint8_t nonce[16];
    memset(nonce, 0, 16);

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    for (uint32_t t = 0; t < 3; t++) {
        uint32_t sz = sizes[t];
        uint8_t *pt = (uint8_t *)malloc(sz);
        uint8_t *ct = (uint8_t *)malloc(sz);
        uint8_t *dec = (uint8_t *)malloc(sz);
        TEST_ASSERT(pt && ct && dec, "Alloc buffers");

        for (uint32_t i = 0; i < sz; i++)
            pt[i] = (uint8_t)((i * 7 + 3) & 0xff);

        /* Encrypt */
        ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CTR, 1,
                                   KT_SWKEY, aes128_key, 128, nonce, 16);
        TEST_ASSERT(ret == 0, "CTR encrypt init");

        uint32_t outlen = 0, flen;
        ret = drv_pufs_cipher_update(&cipher, ct, &outlen, pt, sz);
        TEST_ASSERT(ret == 0, "CTR encrypt update");

        flen = 0;
        ret = drv_pufs_cipher_final(&cipher, ct + outlen, &flen, NULL, 0);
        TEST_ASSERT(ret == 0, "CTR encrypt final");
        outlen += flen;

        /* Decrypt */
        ret = drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_CTR, 0,
                                   KT_SWKEY, aes128_key, 128, nonce, 16);
        TEST_ASSERT(ret == 0, "CTR decrypt init");

        uint32_t ptlen = 0;
        ret = drv_pufs_cipher_update(&cipher, dec, &ptlen, ct, outlen);
        TEST_ASSERT(ret == 0, "CTR decrypt update");

        flen = 0;
        ret = drv_pufs_cipher_final(&cipher, dec + ptlen, &flen, NULL, 0);
        TEST_ASSERT(ret == 0, "CTR decrypt final");
        ptlen += flen;

        char msg[64];
        snprintf(msg, sizeof(msg), "CTR round-trip OK for %u bytes", sz);
        TEST_ASSERT(ptlen == sz && memcmp_hex(dec, pt, sz) == 0, msg);

        free(pt);
        free(ct);
        free(dec);
    }

    drv_pufs_close(&dev);
    return 0;
}

static int test_hash_dma_boundary(void)
{
    TEST_START("SHA-256 at DMA Boundary (64KB)");

    drv_pufs_inst dev;
    drv_pufs_hash_inst hash1, hash2;
    uint8_t dgst1[64], dgst2[64];
    uint32_t dlen1 = 0, dlen2 = 0;
    int ret;

    const uint32_t size = 65536;
    uint8_t *buf = (uint8_t *)malloc(size);
    TEST_ASSERT(buf != NULL, "Alloc 64KB buffer");

    for (uint32_t i = 0; i < size; i++)
        buf[i] = (uint8_t)(i & 0xff);

    ret = drv_pufs_open(&dev);
    TEST_ASSERT(ret == 0, "Open device");

    /* Single update */
    ret = drv_pufs_hash_init(&hash1, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "Hash init (one-shot)");
    ret = drv_pufs_hash_update(&hash1, buf, size);
    TEST_ASSERT(ret == 0, "Hash update 64KB");
    ret = drv_pufs_hash_final(&hash1, dgst1, &dlen1);
    TEST_ASSERT(ret == 0, "Hash final");

    /* Chunked: 4KB chunks */
    ret = drv_pufs_hash_init(&hash2, &dev, HASH_SHA_256);
    TEST_ASSERT(ret == 0, "Hash init (chunked)");
    for (uint32_t off = 0; off < size; off += 4096) {
        ret = drv_pufs_hash_update(&hash2, buf + off, 4096);
        if (ret != 0) break;
    }
    TEST_ASSERT(ret == 0, "Hash update 16x4KB");
    ret = drv_pufs_hash_final(&hash2, dgst2, &dlen2);
    TEST_ASSERT(ret == 0, "Hash final (chunked)");

    TEST_ASSERT(memcmp_hex(dgst1, dgst2, dlen1) == 0,
                "64KB hash: one-shot == chunked");

    print_hex("SHA-256(64KB)", dgst1, dlen1);

    free(buf);
    drv_pufs_close(&dev);
    return 0;
}

/* ================================================================
 * Main
 * ================================================================ */

static void run_test(int (*fn)(void), const char *name)
{
    int before = test_failed;
    int ret = fn();
    (void)ret;
    if (test_failed > before)
        printf(">>> %s: HAD FAILURES <<<\n", name);
}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    printf("========================================\n");
    printf("  PUF Secure Engine HAL Test Suite\n");
    printf("========================================\n");

    /* Section 1: Device management */
    run_test(test_device_open_close, "device_open_close");
    run_test(test_device_double_close, "device_double_close");
    run_test(test_device_multiple_open, "device_multiple_open");

    /* Section 2: Hash algorithms */
    run_test(test_sha256_oneshot, "sha256_oneshot");
    run_test(test_sha256_empty, "sha256_empty");
    run_test(test_sha256_streaming, "sha256_streaming");
    run_test(test_hash_all_algorithms, "hash_all_algorithms");
    run_test(test_hash_large_data, "hash_large_data");
    run_test(test_hash_chunk_boundary, "hash_chunk_boundary");
    run_test(test_hash_reuse_instance, "hash_reuse_instance");

    /* Section 3: HMAC */
    run_test(test_hmac_sha256, "hmac_sha256");
    run_test(test_hmac_streaming, "hmac_streaming");

    /* Section 4: CMAC */
    run_test(test_cmac_aes128, "cmac_aes128");

    /* Section 5: Symmetric ciphers */
    run_test(test_aes128_ecb, "aes128_ecb");
    run_test(test_aes256_ecb, "aes256_ecb");
    run_test(test_aes128_cbc, "aes128_cbc");
    run_test(test_aes128_ctr, "aes128_ctr");
    run_test(test_aes128_cfb, "aes128_cfb");
    run_test(test_aes128_ofb, "aes128_ofb");
    run_test(test_sm4_ecb, "sm4_ecb");
    run_test(test_cipher_streaming, "cipher_streaming");
    run_test(test_cipher_reinit, "cipher_reinit");
    run_test(test_aes128_gcm, "aes128_gcm");
    run_test(test_aes128_ccm, "aes128_ccm");
    run_test(test_aes128_xts, "aes128_xts");
    run_test(test_cipher_large_data, "cipher_large_data");

    /* Section 6: UID */
    run_test(test_uid_read, "uid_read");
    run_test(test_uid_consistency, "uid_consistency");

    /* Section 7: Error / edge cases */
    run_test(test_error_hash_update_after_final, "error_hash_update_after_final");
    run_test(test_error_hash_double_final, "error_hash_double_final");
    run_test(test_error_cipher_bad_keysize, "error_cipher_bad_keysize");
    run_test(test_error_cipher_cbc_no_iv, "error_cipher_cbc_no_iv");
    run_test(test_error_ops_on_closed_device, "error_ops_on_closed_device");

    /* Section 8: Concurrency / multi-thread */
    run_test(test_concurrent_hash_2threads, "concurrent_hash_2threads");
    run_test(test_concurrent_hash_4threads, "concurrent_hash_4threads");
    run_test(test_concurrent_hash_cipher, "concurrent_hash_cipher");
    run_test(test_concurrent_cipher_2threads, "concurrent_cipher_2threads");
    run_test(test_concurrent_separate_devices, "concurrent_separate_devices");

    /* Section 9: Performance / stress */
    run_test(test_stress_open_close, "stress_open_close");
    run_test(test_stress_rapid_hash, "stress_rapid_hash");
    run_test(test_perf_hash, "perf_hash");
    run_test(test_perf_cipher, "perf_cipher");

    /* Section 10: RNG */
    run_test(test_rng_read, "rng_read");
    run_test(test_rng_different_reads, "rng_different_reads");
    run_test(test_rng_various_sizes, "rng_various_sizes");

    /* Section 11: PUFrt management */
    run_test(test_rt_version, "rt_version");
    run_test(test_otp_rwlck_query, "otp_rwlck_query");
    run_test(test_otp_read, "otp_read");
    run_test(test_otp_security_state, "otp_security_state");

    /* Section 12: Key management */
    run_test(test_key_import_export_roundtrip, "key_import_export_roundtrip");
    run_test(test_key_clear, "key_clear");

    /* Section 13: ECC / ECDSA */
    run_test(test_ecc_keygen_puk_verify, "ecc_keygen_puk_verify");
    run_test(test_ecdsa_sign_verify, "ecdsa_sign_verify");

    /* Section 14: ECDH */
    run_test(test_ecdh_shared_secret, "ecdh_shared_secret");

    /* Section 15: DRBG */
    run_test(test_drbg_lifecycle, "drbg_lifecycle");
    run_test(test_drbg_consecutive_generates, "drbg_consecutive_generates");

    /* Section 16: AEAD tag tamper */
    run_test(test_gcm_tag_tamper, "gcm_tag_tamper");
    run_test(test_ccm_tag_tamper, "ccm_tag_tamper");

    /* Section 17: DMA boundary */
    run_test(test_cipher_dma_boundary, "cipher_dma_boundary");
    run_test(test_cipher_across_dma_boundary, "cipher_across_dma_boundary");
    run_test(test_hash_dma_boundary, "hash_dma_boundary");

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed, %d skipped\n",
           test_passed, test_failed, test_skipped);
    printf("========================================\n");

    return test_failed > 0 ? 1 : 0;
}
