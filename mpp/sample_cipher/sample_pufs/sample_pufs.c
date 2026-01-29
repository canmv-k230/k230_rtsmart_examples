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

extern int key_inout_test(void);
extern int hash_test(void);
extern int hmac_test(void);
extern int cmac_test(void);
extern int skcipher_test(void);
extern int akcipher_test(void);

int pufs_uid_test(void)
{
    int ret;
    int fd;

    printf("##########################################################\n");
    printf("PUFS UID TEST DEMO\n");
    printf("##########################################################\n");

    fd = open("/dev/pufs", O_RDWR);
    if (fd < 0) {
        printf("open /dev/pufs err!\n");
        return -1;
    }

    for (int i = KS_PUFSLOT_0; i <= KS_PUFSLOT_3; i++) {
        pufs_uid_t uid;
        pufs_uid_get_t get;
        get.slot = i;
        get.uid = &uid;
        ret = ioctl(fd, PUFS_UID_GET, &get);
        if (ret) {
            printf("ioctl PUFS_UID_GET err!\n");
            break;
        }
        printf("UID[%d]: ", i);
        for (int j = 0; j < sizeof(uid.uid); j++)
            printf("%02x", uid.uid[j]);
        printf("\n");
    }

    if (ret == 0)
        printf("Success!\n");
    else
        printf("Fail!\n");

    close(fd);

    return ret;
}

int main(int argc, char* argv[])
{
    int ret;

    ret = pufs_uid_test();
    if (ret)
        return ret;
    ret = key_inout_test();
    if (ret)
        return ret;
    ret = hash_test();
    if (ret)
        return ret;
    ret = hmac_test();
    if (ret)
        return ret;
    ret = cmac_test();
    if (ret)
        return ret;
    ret = skcipher_test();
    if (ret)
        return ret;
    ret = akcipher_test();
    if (ret)
        return ret;

    return 0;
}
