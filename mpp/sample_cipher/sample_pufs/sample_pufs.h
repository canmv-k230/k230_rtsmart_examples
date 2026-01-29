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

#ifndef __PUFS_H__
#define __PUFS_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include "sys/ioctl.h"

#define PUFS_UID_GET _IOWR('P', 0x00, int)

#define PUFS_KEY_INOUT _IOWR('P', 0x10, int)
#define PUFS_KEY_DERIVE _IOWR('P', 0x11, int)

#define PUFS_HASH_INIT _IOWR('P', 0x20, int)
#define PUFS_HASH_UPDATE _IOWR('P', 0x21, int)
#define PUFS_HASH_FINAL _IOWR('P', 0x22, int)

#define PUFS_MAC_INIT _IOWR('P', 0x28, int)
#define PUFS_MAC_UPDATE _IOWR('P', 0x29, int)
#define PUFS_MAC_FINAL _IOWR('P', 0x2a, int)

#define PUFS_SKCIPHER_INIT _IOWR('P', 0x30, int)
#define PUFS_SKCIPHER_UPDATE _IOWR('P', 0x31, int)
#define PUFS_SKCIPHER_FINAL _IOWR('P', 0x32, int)

#define PUFS_ECC_PRK_GEN _IOWR('P', 0x40, int)
#define PUFS_ECC_PUK_GEN _IOWR('P', 0x41, int)
#define PUFS_ECC_PUK_VERIFY _IOWR('P', 0x42, int)
#define PUFS_ECC_CDH _IOWR('P', 0x43, int)

#define PUFS_ECDSA_SIGN _IOWR('P', 0x45, int)
#define PUFS_ECDSA_VERIFY _IOWR('P', 0x46, int)

#define PUFS_SM2_SIGN _IOWR('P', 0x48, int)
#define PUFS_SM2_VERIFY _IOWR('P', 0x49, int)
#define PUFS_SM2_ENC _IOWR('P', 0x4A, int)
#define PUFS_SM2_DEC _IOWR('P', 0x4B, int)
#define PUFS_SM2_KEX _IOWR('P', 0x4C, int)

#define PUFS_RSA_SIGN _IOWR('P', 0x4E, int)
#define PUFS_RSA_VERIFY _IOWR('P', 0x4F, int)


typedef enum {
    KT_SWKEY,
    KT_OTPKEY,
    KT_PUFKEY,
    KT_RANDKEY,
    KT_SHARESEC,
    KT_SSKEY,
    KT_PRKEY,
} pufs_keytype_t;

typedef enum {
    KS_SK128_0,
    KS_SK128_1,
    KS_SK128_2,
    KS_SK128_3,
    KS_SK128_4,
    KS_SK128_5,
    KS_SK128_6,
    KS_SK128_7,
    KS_SK256_0,
    KS_SK256_1,
    KS_SK256_2,
    KS_SK256_3,
    KS_SK512_0,
    KS_SK512_1,
    KS_PRK_0,
    KS_PRK_1,
    KS_PRK_2,
    KS_SHARESEC_0,
    KS_PUFSLOT_0 = 0,
    KS_PUFSLOT_1,
    KS_PUFSLOT_2,
    KS_PUFSLOT_3,
    KS_OTPKEY_0,
    KS_OTPKEY_1,
    KS_OTPKEY_2,
    KS_OTPKEY_3,
    KS_OTPKEY_4,
    KS_OTPKEY_5,
    KS_OTPKEY_6,
    KS_OTPKEY_7,
    KS_OTPKEY_8,
    KS_OTPKEY_9,
    KS_OTPKEY_10,
    KS_OTPKEY_11,
    KS_OTPKEY_12,
    KS_OTPKEY_13,
    KS_OTPKEY_14,
    KS_OTPKEY_15,
    KS_OTPKEY_16,
    KS_OTPKEY_17,
    KS_OTPKEY_18,
    KS_OTPKEY_19,
    KS_OTPKEY_20,
    KS_OTPKEY_21,
    KS_OTPKEY_22,
    KS_OTPKEY_23,
    KS_OTPKEY_24,
    KS_OTPKEY_25,
    KS_OTPKEY_26,
    KS_OTPKEY_27,
    KS_OTPKEY_28,
    KS_OTPKEY_29,
    KS_OTPKEY_30,
    KS_OTPKEY_31,
} pufs_keyslot_t;

typedef enum {
    KW_AES_CBC_CS2,
    KW_AES_KW,
    KW_AES_KWP,
    KW_AES_KW_INV,
    KW_AES_KWP_INV,
} pufs_keywrap_t;

typedef enum {
    KM_IMPORT_PT,
    KM_IMPORT_WRAP,
    KM_EXPORT_WRAP,
    KM_CLEAR,
} pufs_keymode_t;

typedef enum {
    KD_METHOD_PBKDF,
    KD_METHOD_KBKDF_EXPAND,
    KD_METHOD_KBKDF_EXTRACT,
    KD_METHOD_KBKDF_EXPAND_EXTRACT,
    KD_METHOD_SM2,
} pufs_kd_md_t;

typedef enum {
    KD_PRF_HMAC,
    KD_PRF_HASH,
    KD_PRF_CMAC,
} pufs_kd_prf_t;

typedef enum {
    HASH_SHA_224,
    HASH_SHA_256,
    HASH_SHA_384,
    HASH_SHA_512,
    HASH_SHA_512_224,
    HASH_SHA_512_256,
    HASH_SM3,
} pufs_hashtype_t;

typedef enum {
    ECC_NISTB163,
    ECC_NISTB233,
    ECC_NISTB283,
    ECC_NISTB409,
    ECC_NISTB571,
    ECC_NISTK163,
    ECC_NISTK233,
    ECC_NISTK283,
    ECC_NISTK409,
    ECC_NISTK571,
    ECC_NISTP192,
    ECC_NISTP224,
    ECC_NISTP256,
    ECC_NISTP384,
    ECC_NISTP521,
    ECC_SM2,
} pufs_ecctype_t;

typedef enum {
    RSA_1024,
    RSA_2048,
    RSA_3072,
    RSA_4096,
} pufs_rsatype_t;

typedef enum {
    RSA_BASE,
    RSA_X931,
    RSA_P1V15,
    RSA_PSS,
} pufs_rsamode_t;

typedef enum {
    SM2CT_C1C2C3,
    SM2CT_C1C3C2
} pufs_sm2ct_format_t;

typedef enum {
    MAC_HMAC,
    MAC_CMAC,
} pufs_mac_cipher_t;

typedef enum {
    SK_AES,
    SK_SM4,
} pufs_skcipher_t;

typedef enum {
    MODE_ECB = 1,
    MODE_CFB,
    MODE_OFB,
    MODE_CBC,
    MODE_CBC_CS1,
    MODE_CBC_CS2,
    MODE_CBC_CS3,
    MODE_CTR_32,
    MODE_CTR_64,
    MODE_CTR,
    MODE_GCM = 0x10,
    MODE_CCM,
    MODE_XTS,
} pufs_skcipher_mode_t;

typedef struct {
    uint8_t uid[32];
} pufs_uid_t;

typedef struct {
    uint8_t slot;
    pufs_uid_t *uid;
} pufs_uid_get_t;

typedef struct {
    uint8_t mode;
    uint8_t keytype;
    uint8_t keyslot;
    uint8_t* keyaddr;
    uint32_t keybits;
    uint8_t keywrap;
    uint8_t kwslot;
    uint32_t kwbits;
} pufs_key_io_t;

typedef struct {
    uint8_t keytype;
    uint8_t keyslot;
    uint8_t method;
    uint8_t prf;
    uint8_t hash;
    uint8_t feedback;
    uint8_t ztype;
    uint32_t outbits;
    uint32_t iter;
    uint32_t ctrpos;
    uint32_t ctrlen;
    uint32_t zbits;
    uint32_t saltlen;
    uint32_t infolen;
    uint8_t* iv;
    uint8_t* zaddr;
    uint8_t* salt;
    uint8_t* info;
    uint8_t* out;
} pufs_key_derive_t;

typedef struct {
    uint8_t mode;
} pufs_hash_init_t;

typedef struct {
    uint8_t cipher;
    uint8_t mode;
    uint8_t keytype;
    uint32_t keybits;
    uint8_t* keyaddr;
} pufs_mac_init_t;

typedef struct {
    uint8_t* msg;
    uint32_t msglen;
} pufs_hash_update_t, pufs_mac_update_t;

typedef struct {
    uint8_t* dgst;
    uint32_t* dlen;
} pufs_hash_final_t, pufs_mac_final_t;

typedef struct {
    uint8_t cipher;
    uint8_t mode;
    uint8_t encrypt;
    union {
        struct {
            uint8_t keytype;
            uint32_t keybits;
            uint32_t ivlen;
            uint8_t* keyaddr;
            uint8_t* iv;
        } aes, sm4, aes_gcm;
        struct {
            uint8_t keytype;
            uint32_t keybits;
            uint32_t noncelen;
            uint32_t aadlen;
            uint32_t inlen;
            uint32_t taglen;
            uint8_t* keyaddr;
            uint8_t* nonce;
        } aes_ccm;
        struct {
            uint8_t keytype1;
            uint8_t keytype2;
            uint32_t keybits;
            uint32_t ivlen;
            uint8_t* keyaddr1;
            uint8_t* keyaddr2;
            uint8_t* iv;
        } aes_xts;
    };
} pufs_skcipher_init_t;

typedef struct {
    uint8_t* out;
    uint32_t* outlen;
    uint8_t* in;
    uint32_t inlen;
} pufs_skcipher_update_t;

typedef struct {
    uint8_t* out;
    uint32_t* outlen;
    uint8_t* tag;
    uint32_t taglen;
} pufs_skcipher_final_t;

typedef struct {
    uint8_t ecctype;
    uint8_t is_ephemeral;
    uint8_t prkslot;
    uint8_t keytype;
    uint8_t hashtype;
    uint32_t keybits;
    uint32_t saltlen;
    uint32_t infolen;
    uint8_t* keyaddr;
    uint8_t* salt;
    uint8_t* info;
} pufs_ecc_prk_gen_t;

#define QLEN_MAX 72
typedef struct {
    uint32_t qlen;
    uint8_t x[QLEN_MAX];
    uint8_t y[QLEN_MAX];
} pufs_ecc_puk_t;

typedef struct {
    uint8_t ecctype;
    uint8_t prktype;
    uint8_t prkslot;
    pufs_ecc_puk_t* puk;
} pufs_ecc_puk_gen_t;

typedef struct {
    uint8_t ecctype;
    pufs_ecc_puk_t* puk;
} pufs_ecc_puk_verify_t;

typedef struct {
    uint8_t ecctype;
    uint8_t is_ephemeral;
    uint8_t prkslot_e;
    uint8_t prktype_s;
    uint8_t prkslot_s;
    pufs_ecc_puk_t* puk_e;
    pufs_ecc_puk_t* puk_s;
    uint8_t* out;
} pufs_ecc_cdh_t;

#define NLEN_MAX 72
typedef struct {
    uint32_t qlen;
    uint8_t r[NLEN_MAX];
    uint8_t s[NLEN_MAX];
} pufs_ecdsa_sig_t;

typedef struct {
    uint8_t ecctype;
    uint8_t prktype;
    uint8_t prkslot;
    uint32_t mdlen;
    uint8_t* md;
    pufs_ecdsa_sig_t* sig;
} pufs_ecdsa_sign_t;

typedef struct {
    uint8_t ecctype;
    uint32_t mdlen;
    union {
        pufs_ecc_puk_t* puk;
        uint64_t otpslot;
    };
    uint8_t* md;
    pufs_ecdsa_sig_t* sig;
} pufs_ecdsa_verify_t;

typedef struct {
    uint8_t prktype;
    uint8_t prkslot;
    uint32_t idlen;
    uint32_t msglen;
    uint8_t* id;
    uint8_t* msg;
    pufs_ecdsa_sig_t* sig;
} pufs_sm2_sign_t;

typedef struct {
    uint32_t idlen;
    uint32_t msglen;
    pufs_ecc_puk_t* puk;
    uint8_t* id;
    uint8_t* msg;
    pufs_ecdsa_sig_t* sig;
} pufs_sm2_verify_t;

typedef struct {
    uint8_t format;
    uint32_t inlen;
    uint8_t* in;
    uint8_t* out;
    uint32_t* outlen;
    pufs_ecc_puk_t* puk;
} pufs_sm2_enc_t;

typedef struct {
    uint8_t format;
    uint8_t prkslot;
    uint32_t inlen;
    uint8_t* in;
    uint8_t* out;
    uint32_t* outlen;
} pufs_sm2_dec_t;

typedef struct {
    uint8_t init;
    uint8_t prkslotl;
    uint8_t tprkslotl;
    uint32_t idllen;
    uint32_t idrlen;
    uint32_t keybits;
    pufs_ecc_puk_t* pukr;
    pufs_ecc_puk_t* tpukr;
    uint8_t* idl;
    uint8_t* idr;
    uint8_t* key;
    uint8_t* dgst2;
    uint32_t* dlen2;
    uint8_t* dgst3;
    uint32_t* dlen3;
} pufs_sm2_kex_t;

typedef struct {
    uint8_t rsamode;
    uint8_t rsatype;
    uint8_t hashtype;
    uint32_t puk;
    uint32_t msglen;
    uint32_t saltlen;
    uint8_t* sig;
    uint8_t* n;
    uint8_t* prk;
    uint8_t* msg;
    uint8_t* salt;
} pufs_rsa_sign_t;

typedef struct {
    uint8_t rsamode;
    uint8_t rsatype;
    uint8_t hashtype;
    uint32_t puk;
    uint32_t msglen;
    uint8_t* sig;
    uint8_t* n;
    uint8_t* msg;
} pufs_rsa_verify_t;

#endif
