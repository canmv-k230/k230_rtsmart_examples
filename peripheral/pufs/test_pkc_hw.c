/* SPDX-License-Identifier: BSD-2-Clause */

/*
 * PKC timing and correctness (RSA, ECDSA), CCM with incomplete AAD, and
 * HMAC/CMAC hardware-key slot encoding. Vectors: test_pkc_vectors.h, made by
 * host/gen_pkc_vectors.py.
 */

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "drv_pufs.h"
#include "test_pkc_vectors.h"

static int failures;
static int skipped;

#define CHECK(cond, ...) do { \
    if (cond) { \
        printf("[PASS] "); \
    } else { \
        printf("[FAIL] "); \
        failures++; \
    } \
    printf(__VA_ARGS__); \
    printf("\n"); \
} while (0)

static double now_ms(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ===== RSA and ECDSA ===== */

struct rsa_case {
    const char *name;
    pufs_rsatype_t type;
    const uint8_t *n, *d, *m, *sig;
};

static const struct rsa_case rsa_cases[] = {
    { "RSA-1024", RSA_1024, rsa1024_n, rsa1024_d, rsa1024_m, rsa1024_sig },
    { "RSA-2048", RSA_2048, rsa2048_n, rsa2048_d, rsa2048_m, rsa2048_sig },
    { "RSA-3072", RSA_3072, rsa3072_n, rsa3072_d, rsa3072_m, rsa3072_sig },
    { "RSA-4096", RSA_4096, rsa4096_n, rsa4096_d, rsa4096_m, rsa4096_sig },
};

#define PKC_RUNS 3

static void rsa_case_run(drv_pufs_inst *dev, const struct rsa_case *tc)
{
    uint32_t elen = (tc->type + 1U) * 128U;
    uint8_t sig[512];
    double max_ms = 0;
    int ok = 1;

    for (int run = 0; run < PKC_RUNS; run++) {
        memset(sig, 0, sizeof(sig));
        double start = now_ms();
        int ret = drv_pufs_rsa_sign(dev, RSA_BASE, tc->type, HASH_SHA_256, 65537,
                                    tc->n, tc->d, tc->m, elen, NULL, 0, sig);
        double ms = now_ms() - start;
        if (ms > max_ms)
            max_ms = ms;
        if (ret != 0 || memcmp(sig, tc->sig, elen) != 0) {
            printf("  %s sign run %d: ret=%d (%s), %.1f ms\n", tc->name, run, ret,
                   ret != 0 ? "error" : "wrong signature", ms);
            ok = 0;
        }
    }
    CHECK(ok, "%s sign matches the host signature (%d runs)", tc->name, PKC_RUNS);
    printf("PKC_MAX_MS %s_sign %.1f\n", tc->name, max_ms);

    double start = now_ms();
    int ret = drv_pufs_rsa_verify(dev, RSA_BASE, tc->type, HASH_SHA_256, 65537,
                                  tc->n, tc->m, elen, tc->sig);
    printf("PKC_MAX_MS %s_verify %.1f\n", tc->name, now_ms() - start);
    CHECK(ret == 0, "%s verify accepts the host signature", tc->name);
}

/* Timing only: an all-ones 4095-bit exponent maximizes multiplies in a
 * square-and-multiply engine. The result is not a valid signature. */
static void rsa4096_worst_case(drv_pufs_inst *dev)
{
    uint8_t d[512], sig[512];
    double max_ms = 0;

    memset(d, 0xff, sizeof(d));
    d[0] = 0x7f; /* below n, whose top bit is set */
    for (int run = 0; run < PKC_RUNS; run++) {
        double start = now_ms();
        int ret = drv_pufs_rsa_sign(dev, RSA_BASE, RSA_4096, HASH_SHA_256, 65537,
                                    rsa4096_n, d, rsa4096_m, 512, NULL, 0, sig);
        double ms = now_ms() - start;
        if (ms > max_ms)
            max_ms = ms;
        printf("  RSA-4096 all-ones exponent run %d: ret=%d, %.1f ms\n", run, ret, ms);
    }
    printf("PKC_MAX_MS RSA-4096_sign_all_ones_exponent %.1f\n", max_ms);
    /* The engine must still work afterwards. */
    CHECK(drv_pufs_rsa_verify(dev, RSA_BASE, RSA_4096, HASH_SHA_256, 65537,
                              rsa4096_n, rsa4096_m, 512, rsa4096_sig) == 0,
          "RSA-4096 verify works after the worst-case exponent runs");
}

struct ecc_case {
    const char *name;
    pufs_ecctype_t type;
    uint32_t mdlen;
    uint32_t keybits;
    uint8_t slot;
    bool keygen_may_be_unsupported;
};

static const struct ecc_case ecc_cases[] = {
    { "P-256", ECC_NISTP256, 32, 256, KS_PRK_0, false },
    /* Ephemeral key generation reports a microprogram error (E_ECMPROG) for
     * these curves on K230, also into a cleared slot and with a 256-bit DRBG
     * instantiated (k230-2, 2026-11-22). Their verify is covered below. */
    { "P-384", ECC_NISTP384, 48, 384, KS_PRK_1, true },
    { "P-521", ECC_NISTP521, 64, 521, KS_PRK_2, true },
};

static void ecc_case_run(drv_pufs_inst *dev, const struct ecc_case *tc)
{
    uint8_t md[64];
    pufs_ecc_puk_t puk;
    pufs_ecdsa_sig_t sig;
    double max_sign = 0, max_puk = 0;
    int ok = 1;

    for (uint32_t i = 0; i < sizeof(md); i++)
        md[i] = (uint8_t)(0x31 * i + 7);

    /* Earlier tests may leave a key of another size in the slot. */
    int cleared = drv_pufs_key_clear(dev, KT_PRKEY, tc->slot, tc->keybits);
    printf("  %s: clear PRK slot %u ret=%d\n", tc->name, tc->slot - KS_PRK_0, cleared);
    for (int run = 0; run < PKC_RUNS && ok; run++) {
        int ret = drv_pufs_ecc_prk_gen_status(dev, tc->type, 1, tc->slot);
        if (ret == PUFS_STATUS_ECMPROG && tc->keygen_may_be_unsupported) {
            skipped++;
            printf("[SKIP] %s ephemeral key generation unsupported (E_ECMPROG); "
                   "sign not timed\n", tc->name);
            return;
        }
        if (ret != 0) {
            printf("  %s prk_gen ret=%d\n", tc->name, ret);
            ok = 0;
            break;
        }
        memset(&puk, 0, sizeof(puk));
        double start = now_ms();
        ret = drv_pufs_ecc_puk_gen(dev, tc->type, KT_PRKEY, tc->slot, &puk);
        double ms = now_ms() - start;
        if (ms > max_puk)
            max_puk = ms;
        if (ret != 0) {
            printf("  %s puk_gen ret=%d\n", tc->name, ret);
            ok = 0;
            break;
        }
        memset(&sig, 0, sizeof(sig));
        start = now_ms();
        ret = drv_pufs_ecdsa_sign(dev, tc->type, KT_PRKEY, tc->slot, md, tc->mdlen, &sig);
        ms = now_ms() - start;
        if (ms > max_sign)
            max_sign = ms;
        if (ret != 0) {
            printf("  %s sign ret=%d\n", tc->name, ret);
            ok = 0;
            break;
        }
        ret = drv_pufs_ecdsa_verify(dev, tc->type, md, tc->mdlen, &puk, &sig);
        if (ret != 0) {
            printf("  %s verify ret=%d\n", tc->name, ret);
            ok = 0;
        }
    }
    CHECK(ok, "%s keygen, sign and verify (%d runs)", tc->name, PKC_RUNS);
    printf("PKC_MAX_MS %s_puk_gen %.1f\n", tc->name, max_puk);
    printf("PKC_MAX_MS %s_sign %.1f\n", tc->name, max_sign);
}

struct ecdsa_vector {
    const char *name;
    pufs_ecctype_t type;
    uint32_t qlen, mdlen;
    const uint8_t *qx, *qy, *r, *s, *md;
};

static const struct ecdsa_vector ecdsa_vectors[] = {
    { "P-384", ECC_NISTP384, 48, 48, ecdsa_p384_qx, ecdsa_p384_qy, ecdsa_p384_r,
      ecdsa_p384_s, ecdsa_p384_md },
    { "P-521", ECC_NISTP521, 66, 64, ecdsa_p521_qx, ecdsa_p521_qy, ecdsa_p521_r,
      ecdsa_p521_s, ecdsa_p521_md },
};

/* Verify host-made signatures: accept the real one, refuse a tampered digest. */
static void ecdsa_vector_run(drv_pufs_inst *dev, const struct ecdsa_vector *tv)
{
    pufs_ecc_puk_t puk;
    pufs_ecdsa_sig_t sig;
    uint8_t md[64];
    double max_ms = 0;
    int ok = 1;

    memset(&puk, 0, sizeof(puk));
    memset(&sig, 0, sizeof(sig));
    puk.qlen = sig.qlen = tv->qlen;
    memcpy(puk.x, tv->qx, tv->qlen);
    memcpy(puk.y, tv->qy, tv->qlen);
    memcpy(sig.r, tv->r, tv->qlen);
    memcpy(sig.s, tv->s, tv->qlen);
    memcpy(md, tv->md, tv->mdlen);
    for (int run = 0; run < PKC_RUNS; run++) {
        double start = now_ms();
        int ret = drv_pufs_ecdsa_verify(dev, tv->type, md, tv->mdlen, &puk, &sig);
        double ms = now_ms() - start;
        if (ms > max_ms)
            max_ms = ms;
        if (ret != 0) {
            printf("  %s verify run %d ret=%d\n", tv->name, run, ret);
            ok = 0;
        }
    }
    CHECK(ok, "%s verify accepts the host signature (%d runs)", tv->name, PKC_RUNS);
    printf("PKC_MAX_MS %s_verify %.1f\n", tv->name, max_ms);
    md[0] ^= 1;
    CHECK(drv_pufs_ecdsa_verify(dev, tv->type, md, tv->mdlen, &puk, &sig) != 0,
          "%s verify refuses a tampered digest", tv->name);
}

int test_pkc_timing(void)
{
    drv_pufs_inst dev;

    failures = 0;
    skipped = 0;
    if (drv_pufs_open(&dev) != 0) {
        printf("[FAIL] open /dev/pufs\n");
        return 1;
    }
    for (size_t i = 0; i < sizeof(rsa_cases) / sizeof(rsa_cases[0]); i++)
        rsa_case_run(&dev, &rsa_cases[i]);
    rsa4096_worst_case(&dev);
    for (size_t i = 0; i < sizeof(ecc_cases) / sizeof(ecc_cases[0]); i++)
        ecc_case_run(&dev, &ecc_cases[i]);
    for (size_t i = 0; i < sizeof(ecdsa_vectors) / sizeof(ecdsa_vectors[0]); i++)
        ecdsa_vector_run(&dev, &ecdsa_vectors[i]);
    drv_pufs_close(&dev);
    printf("PKC_TIMING %s (%d failures, %d skipped)\n", failures ? "FAIL" : "PASS", failures, skipped);
    return failures;
}

/* ===== CCM with incomplete AAD ===== */

static int ccm_init(drv_pufs_cipher_inst *c, drv_pufs_inst *dev, int encrypt, uint64_t aadlen)
{
    return drv_pufs_cipher_ccm_init(c, dev, SK_AES, encrypt, KT_SWKEY, ccm_key, 128,
                                    ccm_nonce, sizeof(ccm_nonce), aadlen,
                                    sizeof(ccm_pt), sizeof(ccm_tag));
}

/* AAD supplied as 5 + 15 bytes must give the reference result. */
static void ccm_split_aad(drv_pufs_inst *dev, int encrypt)
{
    drv_pufs_cipher_inst c;
    uint8_t out[sizeof(ccm_pt) + 16], tag[16];
    const uint8_t *in = encrypt ? ccm_pt : ccm_ct;
    const uint8_t *expect = encrypt ? ccm_ct : ccm_pt;
    uint32_t len = 0, flen = 0;
    int ret;

    memcpy(tag, ccm_tag, sizeof(ccm_tag));
    ret = ccm_init(&c, dev, encrypt, sizeof(ccm_aad));
    if (ret == 0)
        ret = drv_pufs_cipher_update(&c, NULL, NULL, ccm_aad, 5);
    if (ret == 0)
        ret = drv_pufs_cipher_update(&c, NULL, NULL, ccm_aad + 5, sizeof(ccm_aad) - 5);
    if (ret == 0)
        ret = drv_pufs_cipher_update(&c, out, &len, in, sizeof(ccm_pt));
    if (ret == 0)
        ret = drv_pufs_cipher_final(&c, out + len, &flen, tag, sizeof(ccm_tag));
    len += flen;
    CHECK(ret == 0 && len == sizeof(ccm_pt) && memcmp(out, expect, len) == 0 &&
          (!encrypt || memcmp(tag, ccm_tag, sizeof(ccm_tag)) == 0),
          "CCM %s with AAD split 5+15 matches the reference (ret=%d, len=%u)",
          encrypt ? "encrypt" : "decrypt", ret, len);
}

/* After 5 of 20 declared AAD bytes, text and final must be refused. */
static void ccm_incomplete(drv_pufs_inst *dev, int encrypt, int final_out, int text_first)
{
    drv_pufs_cipher_inst c;
    uint8_t out[sizeof(ccm_pt) + 16], tag[16];
    uint32_t len = 0;
    int ret;

    memcpy(tag, ccm_tag, sizeof(ccm_tag));
    ret = ccm_init(&c, dev, encrypt, sizeof(ccm_aad));
    if (ret == 0)
        ret = drv_pufs_cipher_update(&c, NULL, NULL, ccm_aad, 5);
    if (ret != 0) {
        CHECK(0, "CCM incomplete AAD setup (ret=%d)", ret);
        return;
    }
    if (text_first) {
        ret = drv_pufs_cipher_update(&c, out, &len, encrypt ? ccm_pt : ccm_ct, sizeof(ccm_pt));
        CHECK(ret != 0, "CCM %s: text update after 5 of 20 AAD bytes refused (ret=%d)",
              encrypt ? "encrypt" : "decrypt", ret);
        return;
    }
    ret = drv_pufs_cipher_final(&c, final_out ? out : NULL, &len, tag, sizeof(ccm_tag));
    CHECK(ret != 0, "CCM %s: final (out=%s) after 5 of 20 AAD bytes refused (ret=%d)",
          encrypt ? "encrypt" : "decrypt", final_out ? "buffer" : "NULL", ret);
}

/* A context with no AAD declared, edited to look like a partly built B1. */
static void ccm_forged_final(drv_pufs_inst *dev, int encrypt, int with_out)
{
    drv_pufs_cipher_inst c;
    pufs_sp38c_op_t op;
    uint8_t tag[16], out[32];
    int ret;

    memcpy(tag, ccm_tag, sizeof(ccm_tag));
    ret = ccm_init(&c, dev, encrypt, 0);
    if (ret != 0) {
        CHECK(0, "CCM forged-context setup (ret=%d)", ret);
        return;
    }
    memset(&op, 0, sizeof(op));
    op.op = PUFS_OP_FINAL;
    memcpy(&op.ctx, &c.ctx.sp38c, sizeof(op.ctx));
    op.ctx.stage = 0; /* SP38C_NONE */
    op.ctx.buflen = 1;
    op.ctx.aadlen = 0;
    op.out_phys = with_out ? (uint64_t)(uintptr_t)out : 0;
    op.tag_phys = (uint64_t)(uintptr_t)tag;
    ret = drv_pufs_ioctl(dev->fd, PUFS_SP38C_OP, &op);
    if (with_out)
        CHECK(ret != 0, "CCM %s: forged stage-NONE context with buffered byte and an output "
              "buffer refused (ret=%d)", encrypt ? "encrypt" : "decrypt", ret);
    else
        CHECK(ret == -EINVAL, "CCM %s: forged stage-NONE context with buffered byte and NULL "
              "out refused with EINVAL (ret=%d)", encrypt ? "encrypt" : "decrypt", ret);
}

/* AAD past the first block: 16 of 20 bytes then text, or 21 bytes; and any
 * AAD when none was declared. All must be refused. */
static void ccm_aad_amount(drv_pufs_inst *dev)
{
    drv_pufs_cipher_inst c;
    uint8_t out[sizeof(ccm_pt) + 16], tag[16], extra[21];
    uint32_t len = 0;
    int ret;

    memcpy(extra, ccm_aad, sizeof(ccm_aad));
    extra[20] = 0x5a;
    ret = ccm_init(&c, dev, 1, sizeof(ccm_aad));
    if (ret == 0)
        ret = drv_pufs_cipher_update(&c, NULL, NULL, ccm_aad, 16);
    CHECK(ret == 0, "CCM: 16 of 20 AAD bytes accepted (ret=%d)", ret);
    ret = drv_pufs_cipher_update(&c, out, &len, ccm_pt, sizeof(ccm_pt));
    CHECK(ret != 0, "CCM: text after 16 of 20 AAD bytes refused (ret=%d)", ret);
    ret = drv_pufs_cipher_final(&c, out, &len, tag, sizeof(ccm_tag));
    CHECK(ret != 0, "CCM: final after 16 of 20 AAD bytes refused (ret=%d)", ret);

    ret = ccm_init(&c, dev, 1, sizeof(ccm_aad));
    if (ret == 0)
        ret = drv_pufs_cipher_update(&c, NULL, NULL, extra, sizeof(extra));
    CHECK(ret != 0, "CCM: 21 AAD bytes when 20 were declared refused (ret=%d)", ret);
    ret = ccm_init(&c, dev, 1, sizeof(ccm_aad));
    if (ret == 0)
        ret = drv_pufs_cipher_update(&c, NULL, NULL, extra, 16);
    if (ret == 0)
        ret = drv_pufs_cipher_update(&c, NULL, NULL, extra + 16, 5);
    CHECK(ret != 0, "CCM: 16 + 5 AAD bytes when 20 were declared refused (ret=%d)", ret);

    ret = ccm_init(&c, dev, 1, 0);
    if (ret == 0)
        ret = drv_pufs_cipher_update(&c, NULL, NULL, ccm_aad, 4);
    CHECK(ret != 0, "CCM: AAD when none was declared refused (ret=%d)", ret);
}

/* Declared AAD never supplied: text must be refused. */
static void ccm_missing_aad(drv_pufs_inst *dev)
{
    drv_pufs_cipher_inst c;
    uint8_t out[sizeof(ccm_pt) + 16];
    uint32_t len = 0;
    int ret = ccm_init(&c, dev, 1, sizeof(ccm_aad));

    if (ret == 0)
        ret = drv_pufs_cipher_update(&c, out, &len, ccm_pt, sizeof(ccm_pt));
    CHECK(ret != 0, "CCM encrypt: text update with declared AAD never supplied refused (ret=%d)", ret);
}

int test_ccm_incomplete_aad(void)
{
    drv_pufs_inst dev;

    failures = 0;
    if (drv_pufs_open(&dev) != 0) {
        printf("[FAIL] open /dev/pufs\n");
        return 1;
    }
    ccm_split_aad(&dev, 1);
    ccm_split_aad(&dev, 0);
    for (int encrypt = 0; encrypt <= 1; encrypt++) {
        ccm_incomplete(&dev, encrypt, 0, 0);
        ccm_incomplete(&dev, encrypt, 1, 0);
        ccm_incomplete(&dev, encrypt, 0, 1);
        ccm_forged_final(&dev, encrypt, 0);
        ccm_forged_final(&dev, encrypt, 1);
    }
    ccm_missing_aad(&dev);
    ccm_aad_amount(&dev);
    /* The kernel survived every refusal: a normal operation still works. */
    ccm_split_aad(&dev, 1);
    drv_pufs_close(&dev);
    printf("CCM_INCOMPLETE_AAD %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures;
}

/* ===== HMAC/CMAC hardware-key slot encoding ===== */

static int cmac_run(drv_pufs_inst *dev, pufs_keytype_t kt, const uint8_t *key, uint8_t tag[16])
{
    drv_pufs_cmac_inst m;
    uint32_t dlen = 0;
    int ret = drv_pufs_cmac_init(&m, dev, SK_AES, kt, key, 128);

    if (ret == 0)
        ret = drv_pufs_cmac_update(&m, ccm_pt, sizeof(ccm_pt));
    if (ret == 0)
        ret = drv_pufs_cmac_final(&m, tag, &dlen);
    return ret != 0 ? ret : (dlen == 16 ? 0 : -1);
}

static int hmac_run(drv_pufs_inst *dev, pufs_keytype_t kt, const uint8_t *key, uint8_t mac[32])
{
    drv_pufs_hash_inst h;
    uint32_t dlen = 0;
    int ret = drv_pufs_hmac_init(&h, dev, HASH_SHA_256, kt, key, 128);

    if (ret == 0)
        ret = drv_pufs_hmac_update(&h, ccm_pt, sizeof(ccm_pt));
    if (ret == 0)
        ret = drv_pufs_hmac_final(&h, mac, &dlen);
    return ret != 0 ? ret : (dlen == 32 ? 0 : -1);
}

static void mac_slot_case(drv_pufs_inst *dev, uint32_t slot, const char *slotname)
{
    uint8_t sw[32], tok[32], raw[32];
    int r_sw, r_tok, r_raw;

    if (drv_pufs_key_import_plaintext(dev, KT_SSKEY, slot, ccm_key, 128) != 0) {
        CHECK(0, "import AES-128 key into %s", slotname);
        return;
    }

    r_sw = cmac_run(dev, KT_SWKEY, ccm_key, sw);
    r_tok = cmac_run(dev, KT_SSKEY, PUFS_CIPHER_SLOT(slot), tok);
    r_raw = cmac_run(dev, KT_SSKEY, (const uint8_t *)(uintptr_t)slot, raw);
    CHECK(r_sw == 0 && r_tok == 0 && memcmp(sw, tok, 16) == 0,
          "CMAC with PUFS_CIPHER_SLOT(%s) equals the software-key CMAC (ret %d/%d)",
          slotname, r_sw, r_tok);
    CHECK(r_raw == 0 && memcmp(sw, raw, 16) == 0,
          "CMAC with raw slot %s equals the software-key CMAC (ret %d)", slotname, r_raw);

    r_sw = hmac_run(dev, KT_SWKEY, ccm_key, sw);
    r_tok = hmac_run(dev, KT_SSKEY, PUFS_CIPHER_SLOT(slot), tok);
    r_raw = hmac_run(dev, KT_SSKEY, (const uint8_t *)(uintptr_t)slot, raw);
    CHECK(r_sw == 0 && r_tok == 0 && memcmp(sw, tok, 32) == 0,
          "HMAC with PUFS_CIPHER_SLOT(%s) equals the software-key HMAC (ret %d/%d)",
          slotname, r_sw, r_tok);
    CHECK(r_raw == 0 && memcmp(sw, raw, 32) == 0,
          "HMAC with raw slot %s equals the software-key HMAC (ret %d)", slotname, r_raw);
    drv_pufs_key_clear(dev, KT_SSKEY, slot, 128);
}

/* Bad hardware slots and NULL software keys fail init with EINVAL; before,
 * HMAC/CMAC silently used slot 0 and a NULL software key an all-zero key. */
static void mac_invalid_keys(drv_pufs_inst *dev)
{
    const uint8_t *bad = PUFS_CIPHER_SLOT(40);
    const uint8_t *raw40 = (const uint8_t *)(uintptr_t)40;
    drv_pufs_hash_inst h;
    drv_pufs_cmac_inst m;
    drv_pufs_cipher_inst c;
    int ret;

#define CHECK_EINVAL(call, what) do { \
        errno = 0; \
        ret = (call); \
        CHECK(ret == -1 && errno == EINVAL, "%s fails with EINVAL (ret %d, errno %d)", \
              what, ret, errno); \
    } while (0)
    CHECK_EINVAL(drv_pufs_hmac_init(&h, dev, HASH_SHA_256, KT_SSKEY, bad, 128),
                 "HMAC init with PUFS_CIPHER_SLOT(40)");
    CHECK_EINVAL(drv_pufs_hmac_init(&h, dev, HASH_SHA_256, KT_SSKEY, raw40, 128),
                 "HMAC init with raw slot 40");
    CHECK_EINVAL(drv_pufs_cmac_init(&m, dev, SK_AES, KT_SSKEY, bad, 128),
                 "CMAC init with PUFS_CIPHER_SLOT(40)");
    CHECK_EINVAL(drv_pufs_cmac_init(&m, dev, SK_AES, KT_SSKEY, raw40, 128),
                 "CMAC init with raw slot 40");
    CHECK_EINVAL(drv_pufs_cmac_init(&m, dev, SK_AES, KT_SWKEY, NULL, 128),
                 "CMAC init with a NULL 128-bit software key");
    CHECK_EINVAL(drv_pufs_cipher_init(&c, dev, SK_AES, MODE_ECB, 1, KT_SSKEY, bad, 128, NULL, 0),
                 "cipher init with PUFS_CIPHER_SLOT(40)");
    CHECK_EINVAL(drv_pufs_cipher_init(&c, dev, SK_AES, MODE_CBC, 1, KT_SWKEY, NULL, 256, ccm_key, 16),
                 "cipher init with a NULL 256-bit software key");
#undef CHECK_EINVAL
}

int test_mac_cipher_slot(void)
{
    drv_pufs_inst dev;

    failures = 0;
    if (drv_pufs_open(&dev) != 0) {
        printf("[FAIL] open /dev/pufs\n");
        return 1;
    }
    mac_slot_case(&dev, KS_SK128_0, "SK128_0");
    mac_slot_case(&dev, KS_SK128_1, "SK128_1");
    mac_invalid_keys(&dev);
    /* Nothing was left locked or half-initialized: a valid slot still works. */
    mac_slot_case(&dev, KS_SK128_0, "SK128_0");
    drv_pufs_close(&dev);
    printf("MAC_CIPHER_SLOT %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures;
}
