/* SPDX-License-Identifier: BSD-2-Clause */

#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <openssl/evp.h>
#include <openssl/hmac.h>

#include "drv_pufs.h"
#include "hal_rvv_ops.h"

static bool slot_supported = true;
static bool fail_hash_final;
static int status_errno, hw_status;
static unsigned cipher_calls;
static pufs_cipher_slot_t slots[2];
static const uint8_t *data;
static uint32_t data_len;

int test_hmac_long_keys(void);

void *hal_rvv_memset(void *dst, int value, size_t n) { return memset(dst, value, n); }
void *hal_rvv_memcpy(void *dst, const void *src, size_t n) { return memcpy(dst, src, n); }

int __wrap_open(const char *path, int flags, ...)
{
    (void)flags;
    assert(strcmp(path, "/dev/pufs") == 0);
    return 42;
}

int __wrap_close(int fd)
{
    assert(fd == 42);
    return 0;
}

static const EVP_MD *digest_method(pufs_hashtype_t hash)
{
    switch (hash) {
    case HASH_SHA_256: return EVP_sha256();
    case HASH_SHA_384: return EVP_sha384();
    case HASH_SHA_512: return EVP_sha512();
    case HASH_SHA_512_224: return EVP_sha512_224();
    case HASH_SHA_512_256: return EVP_sha512_256();
    default: assert(0); return NULL;
    }
}

int __wrap_ioctl(int fd, unsigned long request, ...)
{
    va_list args;
    void *arg;

    assert(fd == 42);
    va_start(args, request);
    arg = va_arg(args, void *);
    va_end(args);

    switch (request) {
    case PUFS_ECC_PRK_GEN:
    case PUFS_ECC_PRK_GEN_STATUS: {
        const pufs_ecc_prk_gen_t *op = arg;
        assert(op->ecctype == ECC_NISTP384 && op->is_ephemeral && op->prkslot == KS_PRK_1);
        if (!status_errno && request == PUFS_ECC_PRK_GEN_STATUS)
            return hw_status;
        errno = status_errno ? status_errno : hw_status;
        return errno ? -1 : 0;
    }
    case PUFS_KEY_INOUT_STATUS:
        if (status_errno) {
            errno = status_errno;
            return -1;
        }
        return hw_status;
    case PUFS_CIPHER_CAPS:
        if (!slot_supported) {
            errno = EINVAL; /* Legacy kernel's unknown-ioctl result. */
            return -1;
        }
        *(uint32_t *)arg = PUFS_CIPHER_CAP_HW_SLOT;
        return 0;
    case PUFS_SP38A_OP:
        slots[0] = ((pufs_sp38a_op_t *)arg)->hwkey;
        break;
    case PUFS_SP38C_OP:
        slots[0] = ((pufs_sp38c_op_t *)arg)->hwkey;
        break;
    case PUFS_SP38D_OP:
        slots[0] = ((pufs_sp38d_op_t *)arg)->hwkey;
        break;
    case PUFS_SP38E_OP:
        slots[0] = ((pufs_sp38e_op_t *)arg)->hwkey1;
        slots[1] = ((pufs_sp38e_op_t *)arg)->hwkey2;
        break;
    case PUFS_HASH_OP: {
        pufs_hash_op_t *op = arg;
        if (op->op == PUFS_OP_UPDATE) {
            data = (const uint8_t *)(uintptr_t)op->msg_phys;
            data_len = op->msglen;
        } else {
            unsigned len = 0;
            uint8_t *out = (uint8_t *)(uintptr_t)op->dgst_phys;

            assert(op->op == PUFS_OP_FINAL);
            assert(EVP_Digest(data, data_len, out, &len, digest_method(op->ctx.hash), NULL) == 1);
            if (fail_hash_final) {
                errno = EBUSY;
                return -1;
            }
            op->dlen = len;
        }
        return 0;
    }
    case PUFS_HMAC_OP: {
        pufs_hmac_op_t *op = arg;
        if (op->op == PUFS_OP_UPDATE) {
            data = (const uint8_t *)(uintptr_t)op->msg_phys;
            data_len = op->msglen;
        } else {
            unsigned len = 0;
            uint8_t *out = (uint8_t *)(uintptr_t)op->dgst_phys;

            assert(op->op == PUFS_OP_FINAL);
            assert(op->keybits == (uint32_t)EVP_MD_size(digest_method(op->hash)) * 8);
            assert(HMAC(digest_method(op->hash), op->ctx.key, op->keybits / 8,
                        data, data_len, out, &len) == out);
            op->dlen = len;
        }
        return 0;
    }
    default:
        assert(0);
    }
    ++cipher_calls;
    return 0;
}

static void test_status(drv_pufs_inst *dev)
{
    pufs_key_io_t op = { .mode = KM_EXPORT_WRAP };

    assert(drv_pufs_ecc_prk_gen_status(dev, ECC_NISTP384, 1, KS_PRK_1) == 0);
    hw_status = PUFS_STATUS_ECMPROG;
    assert(drv_pufs_ecc_prk_gen(dev, ECC_NISTP384, 1, KS_PRK_1) == -PUFS_STATUS_ECMPROG);
    assert(drv_pufs_ecc_prk_gen_status(dev, ECC_NISTP384, 1, KS_PRK_1) == PUFS_STATUS_ECMPROG);
    status_errno = EBADF;
    assert(drv_pufs_ecc_prk_gen_status(dev, ECC_NISTP384, 1, KS_PRK_1) == -EBADF);
    status_errno = EINVAL;
    assert(drv_pufs_ecc_prk_gen_status(dev, ECC_NISTP384, 1, KS_PRK_1) == -EINVAL);

    hw_status = PUFS_STATUS_UNSUPPORT;
    status_errno = 0;
    assert(drv_pufs_ioctl(dev->fd, PUFS_KEY_INOUT_STATUS, &op) == PUFS_STATUS_UNSUPPORT);
    status_errno = EAGAIN;
    assert(drv_pufs_ioctl(dev->fd, PUFS_KEY_INOUT_STATUS, &op) == -EAGAIN);
    puts("[PASS] PUFS hardware statuses are distinct from OS errors");
}

/* Bad slots and missing software keys fail init with EINVAL and send
 * nothing to the kernel; NULL stays raw slot 0 for HMAC and CMAC only. */
static int test_invalid_keys(drv_pufs_inst *dev)
{
    const uint8_t *bad = PUFS_CIPHER_SLOT(40);
    const uint8_t *raw40 = (const uint8_t *)(uintptr_t)40;
    uint8_t swkey[PUFS_SW_KEY_MAXLEN + 1] = {0};
    const uint32_t extra_bit = PUFS_SW_KEY_MAXLEN * 8U + 1;
    drv_pufs_cipher_inst cipher;
    drv_pufs_hash_inst hmac;
    drv_pufs_cmac_inst cmac;
    unsigned calls = cipher_calls;

    assert(bad && (uintptr_t)bad == PUFS_CIPHER_SLOT_INVALID);
    assert(PUFS_CIPHER_SLOT(UINT32_MAX) == bad && PUFS_CIPHER_SLOT(OTPKEY_31 + 1) == bad);

#define EXPECT_EINVAL(call) do { errno = 0; assert((call) == -1 && errno == EINVAL); } while (0)
    /* Cipher: invalid token, raw out-of-range slot, NULL software key. */
    EXPECT_EINVAL(drv_pufs_cipher_init(&cipher, dev, SK_AES, MODE_ECB, 1, KT_OTPKEY, bad, 256, NULL, 0));
    EXPECT_EINVAL(drv_pufs_cipher_init(&cipher, dev, SK_AES, MODE_ECB, 1, KT_RANDKEY, bad, 256, NULL, 0));
    EXPECT_EINVAL(drv_pufs_cipher_init(&cipher, dev, SK_AES, MODE_ECB, 1, KT_SSKEY, raw40, 128, NULL, 0));
    EXPECT_EINVAL(drv_pufs_cipher_init(&cipher, dev, SK_AES, MODE_CBC, 1, KT_SWKEY, NULL, 256, NULL, 0));
    EXPECT_EINVAL(drv_pufs_cipher_ccm_init(&cipher, dev, SK_AES, 1, KT_SWKEY, NULL, 128, NULL, 0, 0, 16, 16));
    EXPECT_EINVAL(drv_pufs_cipher_xts_init(&cipher, dev, SK_AES, 1, KT_SSKEY, PUFS_CIPHER_SLOT(KS_SK128_0),
                                           KT_SSKEY, bad, 128, NULL, 0));
    EXPECT_EINVAL(drv_pufs_cipher_init(&cipher, dev, SK_AES, MODE_ECB, 1, KT_SWKEY,
                                      swkey, extra_bit, NULL, 0));
    EXPECT_EINVAL(drv_pufs_cipher_init_update(&cipher, dev, SK_AES, MODE_GCM, 1, KT_SWKEY,
                                             swkey, extra_bit, NULL, 0, NULL, NULL, NULL, 0));
    EXPECT_EINVAL(drv_pufs_cipher_ccm_init(&cipher, dev, SK_AES, 1, KT_SWKEY,
                                          swkey, extra_bit, NULL, 0, 0, 16, 16));
    EXPECT_EINVAL(drv_pufs_cipher_xts_init(&cipher, dev, SK_AES, 1, KT_SSKEY, PUFS_CIPHER_SLOT(KS_SK128_0),
                                          KT_SWKEY, swkey, extra_bit, NULL, 0));
    assert(cipher_calls == calls);

    /* HMAC: same rules, plus raw slots; NULL is raw slot 0. */
    EXPECT_EINVAL(drv_pufs_hmac_init(&hmac, dev, HASH_SHA_256, KT_SSKEY, bad, 128));
    EXPECT_EINVAL(drv_pufs_hmac_init(&hmac, dev, HASH_SHA_256, KT_SSKEY, raw40, 128));
    EXPECT_EINVAL(drv_pufs_hmac_init(&hmac, dev, HASH_SHA_256, KT_PRKEY + 1, NULL, 128));
    assert(drv_pufs_hmac_init(&hmac, dev, HASH_SHA_256, KT_SSKEY, NULL, 128) == 0 && hmac.ctx.keyslot == 0);
    assert(drv_pufs_hmac_init(&hmac, dev, HASH_SHA_256, KT_SSKEY, PUFS_CIPHER_SLOT(KS_SK128_0), 128) == 0 &&
           hmac.ctx.keyslot == KS_SK128_0);
    assert(drv_pufs_hmac_init(&hmac, dev, HASH_SHA_256, KT_OTPKEY, (const uint8_t *)(uintptr_t)OTPKEY_31,
                              256) == 0 && hmac.ctx.keyslot == OTPKEY_31);

    /* CMAC had no key checks at all. */
    EXPECT_EINVAL(drv_pufs_cmac_init(&cmac, dev, SK_AES, KT_SSKEY, bad, 128));
    EXPECT_EINVAL(drv_pufs_cmac_init(&cmac, dev, SK_AES, KT_SSKEY, raw40, 128));
    EXPECT_EINVAL(drv_pufs_cmac_init(&cmac, dev, SK_AES, KT_SWKEY, NULL, 128));
    EXPECT_EINVAL(drv_pufs_cmac_init(&cmac, dev, SK_AES, KT_SWKEY, swkey, extra_bit));
    EXPECT_EINVAL(drv_pufs_cmac_init(&cmac, dev, SK_AES, KT_SWKEY, swkey, UINT32_MAX));
    EXPECT_EINVAL(drv_pufs_cmac_init(&cmac, dev, SK_AES, KT_PRKEY + 1, NULL, 128));
#if UINTPTR_MAX > UINT32_MAX
    EXPECT_EINVAL(drv_pufs_cmac_init(&cmac, dev, SK_AES, KT_SSKEY,
                                     (const uint8_t *)((uintptr_t)UINT32_MAX + 3), 128));
#endif
    assert(drv_pufs_cmac_init(&cmac, dev, SK_AES, KT_SWKEY, NULL, 0) == 0);
    swkey[PUFS_SW_KEY_MAXLEN - 1] = 0xa5;
    assert(drv_pufs_cmac_init(&cmac, dev, SK_AES, KT_SWKEY, swkey, extra_bit - 1) == 0 &&
           memcmp(cmac.ctx.key, swkey, PUFS_SW_KEY_MAXLEN) == 0);
    assert(drv_pufs_cmac_init(&cmac, dev, SK_AES, KT_SSKEY, NULL, 128) == 0 && cmac.ctx.keyslot == 0);
    assert(drv_pufs_cmac_init(&cmac, dev, SK_AES, KT_OTPKEY, PUFS_CIPHER_SLOT(OTPKEY_10), 256) == 0 &&
           cmac.ctx.keyslot == OTPKEY_10);
#undef EXPECT_EINVAL
    puts("[PASS] invalid slot tokens, NULL keys, and oversized cipher/CMAC keys rejected");
    return 0;
}

int main(void)
{
    drv_pufs_inst dev;
    drv_pufs_cipher_inst cipher;
    drv_pufs_hash_inst hash;
    const uint8_t *slot = PUFS_CIPHER_SLOT(OTPKEY_10);
    uint8_t iv[16] = {0};
    uint8_t key[131] = {0};
    uint8_t zero[PUFS_SW_KEY_MAXLEN] = {0};

    assert(test_hmac_long_keys() == 0);
    assert(drv_pufs_open(&dev) == 0);

    fail_hash_final = true;
    assert(drv_pufs_hmac_init(&hash, &dev, HASH_SHA_256, KT_SWKEY, key, sizeof(key) * 8) == -1);
    assert(errno == EBUSY && memcmp(hash.ctx.key, zero, sizeof(zero)) == 0);
    assert(drv_pufs_hmac_init(&hash, &dev, HASH_SHA_256, KT_SWKEY, NULL, 8) == -1 && errno == EINVAL);

    assert(drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_ECB, 1, KT_OTPKEY, NULL, 256, NULL, 0) == -1);
    assert(errno == EINVAL && cipher_calls == 0);
#if UINTPTR_MAX > UINT32_MAX
    const uint8_t *invalid = (const uint8_t *)((uintptr_t)UINT32_MAX + 1);
    assert(drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_ECB, 1, KT_OTPKEY, invalid, 256, NULL, 0) == -1);
    assert(errno == EINVAL && cipher_calls == 0);
#endif
    assert(test_invalid_keys(&dev) == 0);
    test_status(&dev);
    slot_supported = false;
    assert(drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_ECB, 1, KT_OTPKEY, slot, 256, NULL, 0) == -1);
    assert(errno == ENOTSUP && cipher_calls == 0);
    assert(drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_ECB, 1, KT_SWKEY, key, 256, NULL, 0) == 0);
    assert(cipher_calls == 1);
    assert(drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_ECB, 1, KT_RANDKEY, NULL, 256, NULL, 0) == 0);
    assert(slots[0].present == 0 && cipher_calls == 2);
    slot_supported = true;

    assert(drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_ECB, 1, KT_RANDKEY, NULL, 256, NULL, 0) == 0);
    assert(slots[0].present == 0 && slots[0].slot == 0);
    assert(drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_ECB, 1, KT_OTPKEY, slot, 256, NULL, 0) == 0);
    assert(slots[0].present == PUFS_CIPHER_SLOT_PRESENT && slots[0].slot == OTPKEY_10);
    assert(drv_pufs_cipher_init(&cipher, &dev, SK_AES, MODE_GCM, 1, KT_OTPKEY, slot, 256, iv, 12) == 0);
    assert(slots[0].present == PUFS_CIPHER_SLOT_PRESENT && slots[0].slot == OTPKEY_10);
    assert(drv_pufs_cipher_init_update(&cipher, &dev, SK_AES, MODE_GCM, 1, KT_OTPKEY, slot,
                                      256, iv, 12, NULL, NULL, NULL, 0) == 0);
    assert(slots[0].present == PUFS_CIPHER_SLOT_PRESENT && slots[0].slot == OTPKEY_10);
    assert(drv_pufs_cipher_ccm_init(&cipher, &dev, SK_AES, 1, KT_OTPKEY, slot, 256, iv, 12, 0, 16, 16) == 0);
    assert(slots[0].present == PUFS_CIPHER_SLOT_PRESENT && slots[0].slot == OTPKEY_10);
    assert(drv_pufs_cipher_xts_init(&cipher, &dev, SK_AES, 1, KT_OTPKEY, slot,
                                   KT_SSKEY, PUFS_CIPHER_SLOT(KS_SK128_0), 128, iv, sizeof(iv)) == 0);
    assert(slots[0].present == PUFS_CIPHER_SLOT_PRESENT && slots[0].slot == OTPKEY_10);
    assert(slots[1].present == PUFS_CIPHER_SLOT_PRESENT && slots[1].slot == KS_SK128_0);
    assert(drv_pufs_close(&dev) == 0);
    puts("[PASS] cipher slot encoding, NULL and legacy kernel handling");
    return 0;
}
