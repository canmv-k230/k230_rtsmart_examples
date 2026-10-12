/* Copyright (c) 2026 Canaan Bright Sight Co., Ltd */

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "hal_syscall.h"
#include "k_type.h"
#include "k_module.h"
#include "k_sys_ioctl.h"
#include "k_vb_comm.h"
#include "k_vb_ioctl.h"
#include "k_vicap_ioctl.h"
#include "k_vdec_comm.h"
#include "k_vdec_ioctl.h"
#include "k_venc_comm.h"
#include "k_venc_ioctl.h"
#include "k_vpu_test_ioctl.h"

#define VB_DEVICE "/dev/vb_device"
#define VDEC_DEVICE "/dev/vdec_device"
#define VENC_DEVICE "/dev/venc_device"
#define NONAI_2D_DEVICE "/dev/nonai_2d_device"
#define SYS_DEVICE "/dev/sys"
#define VICAP_DEVICE "/dev/vicap"
#define TEST_MAX_POOL_CNT 4U
#define TEST_BLOCK_SIZE 4096U
#define TEST_BLOCK_COUNT 2U
#define TEST_VENC_WIDTH 320U
#define TEST_VENC_HEIGHT 240U
#define TEST_VENC_Y_SIZE 77824U
#define TEST_VENC_INPUT_SIZE 118784U
#define TEST_VENC_OUTPUT_SIZE 131072U
#define TEST_VENC_OUTPUT_COUNT 4U
#define TEST_VDEC_OUTPUT_SIZE 131072U
#define TEST_VDEC_OUTPUT_COUNT 4U
#define TEST_CHILD_READY_RETRIES 100U
#define TEST_LIFETIME_ITERATIONS 1000U
#define TEST_LIFETIME_CAPACITY_INTERVAL 100U
#define TEST_LIFETIME_SEED 0x023011U

static const k_vb_pool_info readonly_pool_request = {
    .blk_cnt = 1,
    .blk_size = TEST_BLOCK_SIZE,
    .remap_mode = VB_REMAP_MODE_NONE,
};

static const k_vb_blk_info readonly_block_requests[TEST_MAX_POOL_CNT] = {
    { .pool_id = 0, .blk_size = TEST_BLOCK_SIZE },
    { .pool_id = 1, .blk_size = TEST_BLOCK_SIZE },
    { .pool_id = 2, .blk_size = TEST_BLOCK_SIZE },
    { .pool_id = 3, .blk_size = TEST_BLOCK_SIZE },
};

static const int readonly_channel_request = -1;

struct vb_fixture {
    int owner_a;
    int owner_b;
    k_bool initialized;
    k_u32 pool_id;
    k_vb_blk_handle control_handle;
};

struct control_state {
    k_u64 phys_addr;
    k_u32 pool_id;
    k_u32 user_count;
};

static void fixture_init(struct vb_fixture *fixture)
{
    memset(fixture, 0, sizeof(*fixture));
    fixture->owner_a = -1;
    fixture->owner_b = -1;
    fixture->pool_id = VB_INVALID_POOLID;
    fixture->control_handle = VB_INVALID_HANDLE;
}

static int ioctl_expect_success(int fd, const char *name, unsigned long command,
                                void *argument)
{
    int ret = ioctl(fd, command, argument);

    if (ret != K_SUCCESS) {
        printf("FAIL: %s returned 0x%x\n", name, (unsigned int)ret);
        return -1;
    }
    return 0;
}

static int ioctl_expect_error(int fd, const char *owner, const char *name,
                              unsigned long command, void *argument)
{
    int ret = ioctl(fd, command, argument);

    if (ret == K_SUCCESS) {
        printf("FAIL: owner %s %s unexpectedly succeeded\n", owner, name);
        return -1;
    }
    printf("PASS: owner %s %-22s rejected with 0x%x\n",
           owner, name, (unsigned int)ret);
    return 0;
}

static int open_owners(struct vb_fixture *fixture)
{
    fixture->owner_a = open(VB_DEVICE, O_RDWR);
    if (fixture->owner_a < 0) {
        printf("FAIL: open owner A\n");
        return -1;
    }

    fixture->owner_b = open(VB_DEVICE, O_RDWR);
    if (fixture->owner_b < 0) {
        printf("FAIL: open owner B\n");
        close(fixture->owner_a);
        fixture->owner_a = -1;
        return -1;
    }

    printf("owners: A fd=%d, B fd=%d\n", fixture->owner_a, fixture->owner_b);
    return 0;
}

static int setup_vb_fixture(struct vb_fixture *fixture)
{
    k_vb_config config;
    k_vb_pool_info pool_info;
    k_vb_blk_info block_info;

    if (open_owners(fixture) != 0)
        return -1;

    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = TEST_MAX_POOL_CNT;
    if (ioctl_expect_success(fixture->owner_a, "K_VB_SetConfig",
                             K_VB_SetConfig, &config) != 0)
        return -1;

    if (ioctl(fixture->owner_a, K_VB_Init) != K_SUCCESS) {
        printf("FAIL: K_VB_Init\n");
        return -1;
    }
    fixture->initialized = K_TRUE;

    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.blk_cnt = TEST_BLOCK_COUNT;
    pool_info.blk_size = TEST_BLOCK_SIZE;
    pool_info.remap_mode = VB_REMAP_MODE_NONE;
    if (ioctl_expect_success(fixture->owner_a, "K_VB_CreatePool",
                             K_VB_CreatePool, &pool_info) != 0)
        return -1;
    fixture->pool_id = pool_info.pool_id;

    memset(&block_info, 0, sizeof(block_info));
    block_info.pool_id = fixture->pool_id;
    block_info.blk_size = TEST_BLOCK_SIZE;
    if (ioctl_expect_success(fixture->owner_a, "K_VB_GetBlock",
                             K_VB_GetBlock, &block_info) != 0)
        return -1;
    fixture->control_handle = block_info.blk_handle;

    printf("fixture: max_pool_cnt=%u pool=%u control=0x%08x\n",
           TEST_MAX_POOL_CNT, fixture->pool_id, fixture->control_handle);
    return 0;
}

static int read_control_state(int fd, k_vb_blk_handle handle,
                              struct control_state *state)
{
    k_u64 phys_arg = handle;
    k_u32 pool_arg = handle;
    k_u32 count_arg = handle;

    if (ioctl_expect_success(fd, "control Handle2PhysAddr",
                             K_VB_Handle2PhysAddr, &phys_arg) != 0)
        return -1;
    if (ioctl_expect_success(fd, "control Handle2PoolId",
                             K_VB_Handle2PoolId, &pool_arg) != 0)
        return -1;
    if (ioctl_expect_success(fd, "control InquireUserCnt",
                             K_VB_InquireUserCnt, &count_arg) != 0)
        return -1;

    state->phys_addr = phys_arg;
    state->pool_id = pool_arg;
    state->user_count = count_arg;
    return 0;
}

static int test_handle_requests(int fd, const char *owner,
                                k_vb_blk_handle handle)
{
    k_vb_blk_handle release_arg = handle;
    k_u64 phys_arg = handle;
    k_u32 pool_arg = handle;
    k_u32 count_arg = handle;
    k_vb_supplement_info supplement_arg;
    k_u32 ref_arg;
    int failures = 0;

    memset(&supplement_arg, 0, sizeof(supplement_arg));
    supplement_arg.handle = handle;

    failures += ioctl_expect_error(fd, owner, "ReleaseBlock",
                                   K_VB_ReleaseBlock, &release_arg) != 0;
    failures += ioctl_expect_error(fd, owner, "Handle2PhysAddr",
                                   K_VB_Handle2PhysAddr, &phys_arg) != 0;
    failures += ioctl_expect_error(fd, owner, "Handle2PoolId",
                                   K_VB_Handle2PoolId, &pool_arg) != 0;
    failures += ioctl_expect_error(fd, owner, "InquireUserCnt",
                                   K_VB_InquireUserCnt, &count_arg) != 0;
    failures += ioctl_expect_error(fd, owner, "GetSupplementAttr",
                                   K_VB_GetSupplementAttr,
                                   &supplement_arg) != 0;

    ref_arg = handle;
    failures += ioctl_expect_error(fd, owner, "IncBlockRefCnt",
                                   K_VB_IncBlockRefCnt, &ref_arg) != 0;
    ref_arg = handle;
    failures += ioctl_expect_error(fd, owner, "DecBlockRefCnt",
                                   K_VB_DecBlockRefCnt, &ref_arg) != 0;
    ref_arg = handle;
    failures += ioctl_expect_error(fd, owner, "GetBlockRefCnt",
                                   K_VB_GetBlockRefCnt, &ref_arg) != 0;
    return failures == 0 ? 0 : -1;
}

static int test_pool_requests(int fd, const char *owner, k_u32 pool_id)
{
    k_u32 destroy_arg = pool_id;
    k_vb_blk_info block_info;
    int failures = 0;

    memset(&block_info, 0, sizeof(block_info));
    block_info.pool_id = pool_id;
    block_info.blk_size = TEST_BLOCK_SIZE;

    failures += ioctl_expect_error(fd, owner, "DestroyPool",
                                   K_VB_DestoryPool, &destroy_arg) != 0;
    failures += ioctl_expect_error(fd, owner, "GetBlock(pool_id)",
                                   K_VB_GetBlock, &block_info) != 0;
    return failures == 0 ? 0 : -1;
}

static int verify_control_unchanged(struct vb_fixture *fixture,
                                    const struct control_state *before)
{
    struct control_state after;
    k_vb_blk_info second;
    k_vb_blk_info exhausted;
    k_vb_blk_handle release_arg;
    int ret = -1;

    if (read_control_state(fixture->owner_a, fixture->control_handle,
                           &after) != 0)
        return -1;

    if (memcmp(before, &after, sizeof(after)) != 0) {
        printf("FAIL: control state changed: phys 0x%lx->0x%lx, "
               "pool %u->%u, users %u->%u\n",
               before->phys_addr, after.phys_addr,
               before->pool_id, after.pool_id,
               before->user_count, after.user_count);
        return -1;
    }

    memset(&second, 0, sizeof(second));
    second.pool_id = fixture->pool_id;
    second.blk_size = TEST_BLOCK_SIZE;
    if (ioctl_expect_success(fixture->owner_a, "control second GetBlock",
                             K_VB_GetBlock, &second) != 0)
        return -1;

    memset(&exhausted, 0, sizeof(exhausted));
    exhausted.pool_id = fixture->pool_id;
    exhausted.blk_size = TEST_BLOCK_SIZE;
    if (ioctl(fixture->owner_a, K_VB_GetBlock, &exhausted) == K_SUCCESS) {
        printf("FAIL: invalid requests changed two-block pool capacity\n");
        release_arg = exhausted.blk_handle;
        (void)ioctl(fixture->owner_a, K_VB_ReleaseBlock, &release_arg);
        goto out;
    }

    printf("PASS: control handle and two-block pool capacity unchanged\n");
    ret = 0;

out:
    release_arg = second.blk_handle;
    if (ioctl(fixture->owner_a, K_VB_ReleaseBlock, &release_arg) != K_SUCCESS) {
        printf("FAIL: release control second block\n");
        ret = -1;
    }
    return ret;
}

static int cleanup_vb_fixture(struct vb_fixture *fixture)
{
    int failures = 0;

    if (fixture->control_handle != VB_INVALID_HANDLE) {
        k_vb_blk_handle handle = fixture->control_handle;

        if (ioctl(fixture->owner_a, K_VB_ReleaseBlock, &handle) != K_SUCCESS) {
            printf("FAIL: cleanup K_VB_ReleaseBlock\n");
            failures++;
        } else {
            fixture->control_handle = VB_INVALID_HANDLE;
        }
    }

    if (fixture->pool_id != VB_INVALID_POOLID) {
        k_u32 pool_id = fixture->pool_id;

        if (ioctl(fixture->owner_a, K_VB_DestoryPool, &pool_id) != K_SUCCESS) {
            printf("FAIL: cleanup K_VB_DestoryPool\n");
            failures++;
        } else {
            fixture->pool_id = VB_INVALID_POOLID;
        }
    }

    if (fixture->initialized) {
        if (ioctl(fixture->owner_a, K_VB_Exit) != K_SUCCESS) {
            printf("FAIL: cleanup K_VB_Exit\n");
            failures++;
        } else {
            fixture->initialized = K_FALSE;
        }
    }

    if (fixture->owner_b >= 0) {
        if (close(fixture->owner_b) != 0)
            failures++;
        fixture->owner_b = -1;
    }
    if (fixture->owner_a >= 0) {
        if (close(fixture->owner_a) != 0)
            failures++;
        fixture->owner_a = -1;
    }

    return failures == 0 ? 0 : -1;
}

static int run_vb_invalid_handles(void)
{
    static const k_vb_blk_handle invalid_handles[] = {
        (TEST_MAX_POOL_CNT << 16),
        ((TEST_MAX_POOL_CNT + 1U) << 16),
        (TEST_MAX_POOL_CNT << 16) | UINT16_MAX,
        0x7fff0000U,
        0xffff0000U,
        UINT32_MAX,
    };
    static const k_u32 invalid_pool_ids[] = {
        TEST_MAX_POOL_CNT,
        TEST_MAX_POOL_CNT + 1U,
        UINT16_MAX,
        UINT32_MAX,
    };
    struct vb_fixture fixture;
    struct control_state before;
    size_t i;
    int failures = 0;

    fixture_init(&fixture);
    printf("M-01 VB invalid-handle boundary test\n");

    if (setup_vb_fixture(&fixture) != 0) {
        failures++;
        goto out;
    }
    if (read_control_state(fixture.owner_a, fixture.control_handle,
                           &before) != 0) {
        failures++;
        goto out;
    }

    for (i = 0; i < sizeof(invalid_handles) / sizeof(invalid_handles[0]); i++) {
        printf("forged handle[%u]=0x%08x pool=%u block=%u\n",
               (unsigned int)i, invalid_handles[i],
               invalid_handles[i] >> 16, invalid_handles[i] & UINT16_MAX);
        failures += test_handle_requests(fixture.owner_a, "A",
                                         invalid_handles[i]) != 0;
        failures += test_handle_requests(fixture.owner_b, "B",
                                         invalid_handles[i]) != 0;
    }

    for (i = 0; i < sizeof(invalid_pool_ids) / sizeof(invalid_pool_ids[0]); i++) {
        printf("invalid pool_id[%u]=%u\n",
               (unsigned int)i, invalid_pool_ids[i]);
        failures += test_pool_requests(fixture.owner_a, "A",
                                       invalid_pool_ids[i]) != 0;
        failures += test_pool_requests(fixture.owner_b, "B",
                                       invalid_pool_ids[i]) != 0;
    }

    failures += verify_control_unchanged(&fixture, &before) != 0;

out:
    failures += cleanup_vb_fixture(&fixture) != 0;
    if (failures != 0) {
        printf("M-01 FAIL: %d failure(s)\n", failures);
        return 1;
    }

    printf("M-01 PASS\n");
    return 0;
}

static int run_vb_owner_isolation(void)
{
    struct vb_fixture fixture;
    struct vb_fixture fresh;
    k_vb_pool_info empty_pool;
    k_u32 empty_pool_id = VB_INVALID_POOLID;
    k_u32 destroy_arg;
    k_vb_blk_handle release_arg;
    int ret;
    int failures = 0;

    fixture_init(&fixture);
    fixture_init(&fresh);
    printf("M-02 VB owner-isolation and close-cleanup test\n");

    if (setup_vb_fixture(&fixture) != 0) {
        failures++;
        goto cleanup;
    }

    memset(&empty_pool, 0, sizeof(empty_pool));
    empty_pool.blk_cnt = 1;
    empty_pool.blk_size = TEST_BLOCK_SIZE;
    empty_pool.remap_mode = VB_REMAP_MODE_NONE;
    if (ioctl_expect_success(fixture.owner_a, "owner A second CreatePool",
                             K_VB_CreatePool, &empty_pool) != 0) {
        failures++;
        goto cleanup;
    }
    empty_pool_id = empty_pool.pool_id;

    ret = ioctl(fixture.owner_b, K_VB_Exit);
    if (ret == K_SUCCESS) {
        printf("FAIL: owner B exited owner A's VB instance\n");
        failures++;
        fixture.initialized = K_FALSE;
    } else {
        printf("PASS: owner B exit rejected with 0x%x\n", (unsigned int)ret);
    }

    destroy_arg = empty_pool_id;
    ret = ioctl(fixture.owner_b, K_VB_DestoryPool, &destroy_arg);
    if (ret == K_SUCCESS) {
        printf("FAIL: owner B destroyed owner A's pool %u\n", empty_pool_id);
        failures++;
        empty_pool_id = VB_INVALID_POOLID;
    } else {
        printf("PASS: owner B destroy rejected with 0x%x\n",
               (unsigned int)ret);
    }

    release_arg = fixture.control_handle;
    ret = ioctl(fixture.owner_b, K_VB_ReleaseBlock, &release_arg);
    if (ret == K_SUCCESS) {
        printf("FAIL: owner B released owner A's block 0x%08x\n",
               fixture.control_handle);
        failures++;
    } else {
        printf("PASS: owner B release rejected with 0x%x\n",
               (unsigned int)ret);
    }

    printf("closing owner A without explicit cleanup\n");
    if (close(fixture.owner_a) != 0) {
        printf("FAIL: closing owner A\n");
        failures++;
    }
    fixture.owner_a = -1;
    fixture.initialized = K_FALSE;
    fixture.pool_id = VB_INVALID_POOLID;
    fixture.control_handle = VB_INVALID_HANDLE;
    empty_pool_id = VB_INVALID_POOLID;

    if (close(fixture.owner_b) != 0) {
        printf("FAIL: closing owner B\n");
        failures++;
    }
    fixture.owner_b = -1;

    printf("starting fresh VB run after owner A close\n");
    if (setup_vb_fixture(&fresh) != 0) {
        printf("FAIL: fresh VB run could not allocate resources\n");
        failures++;
    } else {
        printf("PASS: fresh VB run allocated resources\n");
    }
    failures += cleanup_vb_fixture(&fresh) != 0;

cleanup:
    if (fixture.owner_a >= 0 && empty_pool_id != VB_INVALID_POOLID) {
        destroy_arg = empty_pool_id;
        (void)ioctl(fixture.owner_a, K_VB_DestoryPool, &destroy_arg);
    }
    failures += cleanup_vb_fixture(&fixture) != 0;

    if (failures != 0) {
        printf("M-02 FAIL: %d failure(s)\n", failures);
        return 1;
    }

    printf("M-02 PASS\n");
    return 0;
}

static k_u32 read_readonly_u32(const k_u32 *value)
{
    return *(const volatile k_u32 *)value;
}

static int run_vb_copyout_rollback(void)
{
    struct vb_fixture fixture;
    k_vb_config config;
    k_vb_pool_info pools[TEST_MAX_POOL_CNT];
    k_vb_pool_info retry_pool;
    k_vb_blk_info blocks[TEST_BLOCK_COUNT];
    k_vb_blk_info exhausted;
    k_u32 pool_ids[TEST_MAX_POOL_CNT];
    k_u32 retry_pool_id = VB_INVALID_POOLID;
    k_u32 value;
    unsigned int created_pools = 0;
    unsigned int acquired_blocks = 0;
    unsigned int i;
    int ret;
    int failures = 0;

    fixture_init(&fixture);
    memset(pool_ids, 0xff, sizeof(pool_ids));
    printf("M-03 VB copyout rollback test\n");

    if (open_owners(&fixture) != 0)
        return 1;

    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = TEST_MAX_POOL_CNT;
    if (ioctl_expect_success(fixture.owner_a, "K_VB_SetConfig",
                             K_VB_SetConfig, &config) != 0) {
        failures++;
        goto cleanup;
    }
    if (ioctl(fixture.owner_a, K_VB_Init) != K_SUCCESS) {
        printf("FAIL: K_VB_Init\n");
        failures++;
        goto cleanup;
    }
    fixture.initialized = K_TRUE;

    ret = ioctl(fixture.owner_a, K_VB_CreatePool,
                (void *)(uintptr_t)&readonly_pool_request);
    if (ret == K_SUCCESS) {
        value = read_readonly_u32(&readonly_pool_request.pool_id);
        printf("FAIL: CreatePool wrote to read-only output, pool=%u\n", value);
        failures++;
        if (value < TEST_MAX_POOL_CNT)
            (void)ioctl(fixture.owner_a, K_VB_DestoryPool, &value);
    } else {
        printf("PASS: CreatePool copyout failed with 0x%x\n",
               (unsigned int)ret);
    }

    for (i = 0; i < TEST_MAX_POOL_CNT; i++) {
        memset(&pools[i], 0, sizeof(pools[i]));
        pools[i].blk_cnt = 1;
        pools[i].blk_size = TEST_BLOCK_SIZE;
        pools[i].remap_mode = VB_REMAP_MODE_NONE;
        if (ioctl(fixture.owner_a, K_VB_CreatePool, &pools[i]) != K_SUCCESS) {
            printf("FAIL: pool capacity reduced at allocation %u\n", i);
            failures++;
            break;
        }
        pool_ids[i] = pools[i].pool_id;
        created_pools++;
    }
    if (created_pools == TEST_MAX_POOL_CNT)
        printf("PASS: all %u pool slots available after CreatePool probe\n",
               TEST_MAX_POOL_CNT);

    for (i = 0; i < created_pools; i++) {
        value = pool_ids[i];
        if (ioctl(fixture.owner_a, K_VB_DestoryPool, &value) != K_SUCCESS) {
            printf("FAIL: destroy capacity-check pool %u\n", pool_ids[i]);
            failures++;
        }
    }
    created_pools = 0;

    memset(&retry_pool, 0, sizeof(retry_pool));
    retry_pool.blk_cnt = TEST_BLOCK_COUNT;
    retry_pool.blk_size = TEST_BLOCK_SIZE;
    retry_pool.remap_mode = VB_REMAP_MODE_NONE;
    if (ioctl(fixture.owner_a, K_VB_CreatePool, &retry_pool) != K_SUCCESS) {
        printf("FAIL: create get-block retry pool\n");
        failures++;
        goto cleanup;
    }
    retry_pool_id = retry_pool.pool_id;
    if (retry_pool_id >= TEST_MAX_POOL_CNT) {
        printf("FAIL: retry pool id %u is outside configured range\n",
               retry_pool_id);
        failures++;
        goto cleanup;
    }

    ret = ioctl(fixture.owner_a, K_VB_GetBlock,
                (void *)(uintptr_t)&readonly_block_requests[retry_pool_id]);
    if (ret == K_SUCCESS) {
        value = read_readonly_u32(
            &readonly_block_requests[retry_pool_id].blk_handle);
        printf("FAIL: GetBlock wrote to read-only output, handle=0x%08x\n",
               value);
        failures++;
        if (value != VB_INVALID_HANDLE)
            (void)ioctl(fixture.owner_a, K_VB_ReleaseBlock, &value);
    } else {
        printf("PASS: GetBlock copyout failed with 0x%x\n",
               (unsigned int)ret);
    }

    for (i = 0; i < TEST_BLOCK_COUNT; i++) {
        memset(&blocks[i], 0, sizeof(blocks[i]));
        blocks[i].pool_id = retry_pool_id;
        blocks[i].blk_size = TEST_BLOCK_SIZE;
        if (ioctl(fixture.owner_a, K_VB_GetBlock, &blocks[i]) != K_SUCCESS) {
            printf("FAIL: block capacity reduced at allocation %u\n", i);
            failures++;
            break;
        }
        acquired_blocks++;
    }

    memset(&exhausted, 0, sizeof(exhausted));
    exhausted.pool_id = retry_pool_id;
    exhausted.blk_size = TEST_BLOCK_SIZE;
    if (acquired_blocks == TEST_BLOCK_COUNT &&
        ioctl(fixture.owner_a, K_VB_GetBlock, &exhausted) == K_SUCCESS) {
        printf("FAIL: two-block pool unexpectedly provided a third block\n");
        value = exhausted.blk_handle;
        (void)ioctl(fixture.owner_a, K_VB_ReleaseBlock, &value);
        failures++;
    } else if (acquired_blocks == TEST_BLOCK_COUNT) {
        printf("PASS: both block slots available after GetBlock probe\n");
    }

cleanup:
    for (i = 0; i < acquired_blocks; i++) {
        value = blocks[i].blk_handle;
        if (ioctl(fixture.owner_a, K_VB_ReleaseBlock, &value) != K_SUCCESS)
            failures++;
    }
    if (retry_pool_id != VB_INVALID_POOLID) {
        value = retry_pool_id;
        if (ioctl(fixture.owner_a, K_VB_DestoryPool, &value) != K_SUCCESS)
            failures++;
    }
    failures += cleanup_vb_fixture(&fixture) != 0;

    if (failures != 0) {
        printf("M-03 FAIL: %d failure(s)\n", failures);
        return 1;
    }

    printf("M-03 PASS\n");
    return 0;
}

struct venc_fixture {
    int vb_fd;
    int owner_a;
    int owner_b;
    int nonai_2d_fd;
    k_bool vb_initialized;
    k_bool channel_requested;
    k_bool pool_attached;
    k_bool channel_created;
    k_bool channel_started;
    k_bool two_d_attached;
    k_u32 output_pool;
    k_u32 input_pool;
    k_u32 channel;
    k_vb_blk_handle input_handle;
    k_u64 input_phys;
};

static void venc_fixture_init(struct venc_fixture *fixture)
{
    memset(fixture, 0, sizeof(*fixture));
    fixture->vb_fd = -1;
    fixture->owner_a = -1;
    fixture->owner_b = -1;
    fixture->nonai_2d_fd = -1;
    fixture->output_pool = VB_INVALID_POOLID;
    fixture->input_pool = VB_INVALID_POOLID;
    fixture->channel = VENC_MAX_CHN_NUMS;
    fixture->input_handle = VB_INVALID_HANDLE;
}

static int vb_create_private_pool(int fd, k_u32 block_count,
                                  k_u64 block_size, k_u32 *pool_id)
{
    k_vb_pool_info info;

    memset(&info, 0, sizeof(info));
    info.blk_cnt = block_count;
    info.blk_size = block_size;
    info.remap_mode = VB_REMAP_MODE_NOCACHE;
    if (ioctl(fd, K_VB_CreatePool, &info) != K_SUCCESS)
        return -1;

    *pool_id = info.pool_id;
    return 0;
}

static int setup_venc_fixture(struct venc_fixture *fixture)
{
    k_vb_config config;
    k_vb_blk_info block;
    venc_chn_pool_info pool_info;
    venc_chn_info channel_info;
    int requested_channel = -1;

    fixture->vb_fd = open(VB_DEVICE, O_RDWR);
    if (fixture->vb_fd < 0) {
        printf("FAIL: open VB device\n");
        return -1;
    }

    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = TEST_MAX_POOL_CNT;
    if (ioctl(fixture->vb_fd, K_VB_SetConfig, &config) != K_SUCCESS ||
        ioctl(fixture->vb_fd, K_VB_Init) != K_SUCCESS) {
        printf("FAIL: initialize VB for VENC\n");
        return -1;
    }
    fixture->vb_initialized = K_TRUE;

    if (vb_create_private_pool(fixture->vb_fd, TEST_VENC_OUTPUT_COUNT,
                               TEST_VENC_OUTPUT_SIZE,
                               &fixture->output_pool) != 0 ||
        vb_create_private_pool(fixture->vb_fd, 1, TEST_VENC_INPUT_SIZE,
                               &fixture->input_pool) != 0) {
        printf("FAIL: create VENC VB pools\n");
        return -1;
    }

    memset(&block, 0, sizeof(block));
    block.pool_id = fixture->input_pool;
    block.blk_size = TEST_VENC_INPUT_SIZE;
    if (ioctl(fixture->vb_fd, K_VB_GetBlock, &block) != K_SUCCESS) {
        printf("FAIL: get VENC input block\n");
        return -1;
    }
    fixture->input_handle = block.blk_handle;
    fixture->input_phys = fixture->input_handle;
    if (ioctl(fixture->vb_fd, K_VB_Handle2PhysAddr,
              &fixture->input_phys) != K_SUCCESS || !fixture->input_phys) {
        printf("FAIL: resolve VENC input physical address\n");
        return -1;
    }

    fixture->owner_a = open(VENC_DEVICE, O_RDWR);
    fixture->owner_b = open(VENC_DEVICE, O_RDWR);
    fixture->nonai_2d_fd = open(NONAI_2D_DEVICE, O_RDWR);
    if (fixture->owner_a < 0 || fixture->owner_b < 0 ||
        fixture->nonai_2d_fd < 0) {
        printf("FAIL: open VENC owners or non-AI 2D device\n");
        return -1;
    }
    printf("owners: A fd=%d, B fd=%d\n", fixture->owner_a,
           fixture->owner_b);

    if (ioctl(fixture->owner_a, K_IOC_VENC_RequestChn,
              &requested_channel) != K_SUCCESS || requested_channel < 0 ||
        requested_channel >= (int)VENC_MAX_CHN_NUMS) {
        printf("FAIL: owner A request VENC channel\n");
        return -1;
    }
    fixture->channel = (k_u32)requested_channel;
    fixture->channel_requested = K_TRUE;

    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.chn_num = fixture->channel;
    pool_info.pool_id = fixture->output_pool;
    pool_info.buff_num = TEST_VENC_OUTPUT_COUNT;
    if (ioctl(fixture->owner_a, K_IOC_VENC_AttachVbPool,
              &pool_info) != K_SUCCESS) {
        printf("FAIL: owner A attach VENC VB pool\n");
        return -1;
    }
    fixture->pool_attached = K_TRUE;

    memset(&channel_info, 0, sizeof(channel_info));
    channel_info.chn_num = fixture->channel;
    channel_info.attr.venc_attr.type = K_PT_JPEG;
    channel_info.attr.venc_attr.pic_width = TEST_VENC_WIDTH;
    channel_info.attr.venc_attr.pic_height = TEST_VENC_HEIGHT;
    channel_info.attr.rc_attr.rc_mode = K_VENC_RC_MODE_MJPEG_FIXQP;
    channel_info.attr.rc_attr.mjpeg_fixqp.src_frame_rate = 30;
    channel_info.attr.rc_attr.mjpeg_fixqp.dst_frame_rate = 30;
    channel_info.attr.rc_attr.mjpeg_fixqp.q_factor = 75;
    if (ioctl(fixture->owner_a, K_IOC_VENC_CreateChn,
              &channel_info) != K_SUCCESS) {
        printf("FAIL: owner A create VENC channel %u\n", fixture->channel);
        return -1;
    }
    fixture->channel_created = K_TRUE;

    if (ioctl(fixture->owner_a, K_IOC_VENC_StartChn,
              &fixture->channel) != K_SUCCESS) {
        printf("FAIL: owner A start VENC channel %u\n", fixture->channel);
        return -1;
    }
    fixture->channel_started = K_TRUE;
    return 0;
}

static int venc_encode_one_frame(struct venc_fixture *fixture, k_u64 pts,
                                 const char *stage)
{
    venc_frame input;
    venc_stream output;
    k_venc_pack packs[TEST_VENC_OUTPUT_COUNT];
    k_u32 i;
    k_u32 total_size = 0;
    int ret;

    memset(&input, 0, sizeof(input));
    input.chn_num = fixture->channel;
    input.milli_sec = 1000;
    input.frame.pool_id = fixture->input_pool;
    input.frame.mod_id = K_ID_VB;
    input.frame.v_frame.width = TEST_VENC_WIDTH;
    input.frame.v_frame.height = TEST_VENC_HEIGHT;
    input.frame.v_frame.pixel_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    input.frame.v_frame.stride[0] = TEST_VENC_WIDTH;
    input.frame.v_frame.stride[1] = TEST_VENC_WIDTH;
    input.frame.v_frame.phys_addr[0] = fixture->input_phys;
    input.frame.v_frame.phys_addr[1] =
        fixture->input_phys + TEST_VENC_Y_SIZE;
    input.frame.v_frame.pts = pts;

    ret = ioctl(fixture->owner_a, K_IOC_VENC_SendFrame, &input);
    if (ret != K_SUCCESS) {
        printf("FAIL: %s SendFrame returned 0x%x\n", stage,
               (unsigned int)ret);
        return -1;
    }

    memset(packs, 0, sizeof(packs));
    memset(&output, 0, sizeof(output));
    output.chn_num = fixture->channel;
    output.pack = packs;
    output.pack_cnt = TEST_VENC_OUTPUT_COUNT;
    output.milli_sec = 3000;
    ret = ioctl(fixture->owner_a, K_IOC_VENC_GetStream, &output);
    if (ret != K_SUCCESS || output.pack_cnt == 0 ||
        output.pack_cnt > TEST_VENC_OUTPUT_COUNT) {
        printf("FAIL: %s GetStream returned 0x%x with %u packs\n", stage,
               (unsigned int)ret, output.pack_cnt);
        return -1;
    }

    for (i = 0; i < output.pack_cnt; i++)
        total_size += packs[i].len;
    if (total_size == 0) {
        printf("FAIL: %s encoded stream is empty\n", stage);
        (void)ioctl(fixture->owner_a, K_IOC_VENC_RelStream, &output);
        return -1;
    }

    ret = ioctl(fixture->owner_a, K_IOC_VENC_RelStream, &output);
    if (ret != K_SUCCESS) {
        printf("FAIL: %s RelStream returned 0x%x\n", stage,
               (unsigned int)ret);
        return -1;
    }

    printf("PASS: %s encoded %u bytes in %u pack(s)\n", stage,
           total_size, output.pack_cnt);
    return 0;
}

static int cleanup_venc_fixture(struct venc_fixture *fixture)
{
    k_u32 value;
    int failures = 0;

    if (fixture->owner_a >= 0 && fixture->two_d_attached) {
        if (ioctl(fixture->owner_a, K_IOC_VENC_2D_Detach,
                  &fixture->channel) != K_SUCCESS)
            failures++;
        else
            fixture->two_d_attached = K_FALSE;
    }
    if (fixture->owner_a >= 0 && fixture->channel_started) {
        if (ioctl(fixture->owner_a, K_IOC_VENC_StopChn,
                  &fixture->channel) != K_SUCCESS)
            failures++;
        else
            fixture->channel_started = K_FALSE;
    }
    if (fixture->owner_a >= 0 && fixture->channel_created) {
        if (ioctl(fixture->owner_a, K_IOC_VENC_DestroyChn,
                  &fixture->channel) != K_SUCCESS)
            failures++;
        else
            fixture->channel_created = K_FALSE;
    }
    if (fixture->owner_a >= 0 && fixture->pool_attached) {
        if (ioctl(fixture->owner_a, K_IOC_VENC_DetachVbPool,
                  &fixture->channel) != K_SUCCESS)
            failures++;
        else
            fixture->pool_attached = K_FALSE;
    }
    if (fixture->owner_a >= 0 && fixture->channel_requested) {
        if (ioctl(fixture->owner_a, K_IOC_VENC_ReleaseChn,
                  &fixture->channel) != K_SUCCESS)
            failures++;
        else
            fixture->channel_requested = K_FALSE;
    }

    if (fixture->owner_b >= 0) {
        failures += close(fixture->owner_b) != 0;
        fixture->owner_b = -1;
    }
    if (fixture->owner_a >= 0) {
        failures += close(fixture->owner_a) != 0;
        fixture->owner_a = -1;
    }
    if (fixture->nonai_2d_fd >= 0) {
        failures += close(fixture->nonai_2d_fd) != 0;
        fixture->nonai_2d_fd = -1;
    }

    if (fixture->vb_fd >= 0 &&
        fixture->input_handle != VB_INVALID_HANDLE) {
        value = fixture->input_handle;
        if (ioctl(fixture->vb_fd, K_VB_ReleaseBlock, &value) != K_SUCCESS)
            failures++;
        else
            fixture->input_handle = VB_INVALID_HANDLE;
    }
    if (fixture->vb_fd >= 0 && fixture->input_pool != VB_INVALID_POOLID) {
        value = fixture->input_pool;
        if (ioctl(fixture->vb_fd, K_VB_DestoryPool, &value) != K_SUCCESS)
            failures++;
        else
            fixture->input_pool = VB_INVALID_POOLID;
    }
    if (fixture->vb_fd >= 0 && fixture->output_pool != VB_INVALID_POOLID) {
        value = fixture->output_pool;
        if (ioctl(fixture->vb_fd, K_VB_DestoryPool, &value) != K_SUCCESS)
            failures++;
        else
            fixture->output_pool = VB_INVALID_POOLID;
    }
    if (fixture->vb_fd >= 0 && fixture->vb_initialized) {
        if (ioctl(fixture->vb_fd, K_VB_Exit) != K_SUCCESS)
            failures++;
        else
            fixture->vb_initialized = K_FALSE;
    }
    if (fixture->vb_fd >= 0) {
        failures += close(fixture->vb_fd) != 0;
        fixture->vb_fd = -1;
    }
    return failures == 0 ? 0 : -1;
}

static int run_venc_owner_isolation(void)
{
    struct venc_fixture fixture;
    venc_chn_status status;
    int failures = 0;

    venc_fixture_init(&fixture);
    printf("M-04 VENC owner-isolation test\n");
    if (setup_venc_fixture(&fixture) != 0) {
        failures++;
        goto cleanup;
    }

    failures += venc_encode_one_frame(&fixture, 1,
                                      "baseline owner A frame") != 0;

    if (ioctl(fixture.owner_a, K_IOC_VENC_2D_Attach,
              &fixture.channel) != K_SUCCESS) {
        printf("FAIL: owner A attach VENC 2D\n");
        failures++;
        goto cleanup;
    }
    fixture.two_d_attached = K_TRUE;

    failures += ioctl_expect_error(fixture.owner_b, "B", "ReleaseChn",
                                   K_IOC_VENC_ReleaseChn,
                                   &fixture.channel) != 0;
    failures += ioctl_expect_error(fixture.owner_b, "B", "DestroyChn",
                                   K_IOC_VENC_DestroyChn,
                                   &fixture.channel) != 0;
    failures += ioctl_expect_error(fixture.owner_b, "B", "DetachVbPool",
                                   K_IOC_VENC_DetachVbPool,
                                   &fixture.channel) != 0;
    failures += ioctl_expect_error(fixture.owner_b, "B", "Detach2D",
                                   K_IOC_VENC_2D_Detach,
                                   &fixture.channel) != 0;

    memset(&status, 0, sizeof(status));
    status.chn_num = fixture.channel;
    if (ioctl(fixture.owner_a, K_IOC_VENC_QueryStatus, &status) != K_SUCCESS) {
        printf("FAIL: owner A channel unavailable after owner B requests\n");
        failures++;
    } else {
        printf("PASS: owner A channel remained available\n");
    }

    if (ioctl(fixture.owner_a, K_IOC_VENC_2D_Detach,
              &fixture.channel) != K_SUCCESS) {
        printf("FAIL: owner A detach VENC 2D\n");
        failures++;
    } else {
        fixture.two_d_attached = K_FALSE;
    }

    failures += venc_encode_one_frame(&fixture, 2,
                                      "post-isolation owner A frame") != 0;

cleanup:
    failures += cleanup_venc_fixture(&fixture) != 0;
    if (failures != 0) {
        printf("M-04 FAIL: %d failure(s)\n", failures);
        return 1;
    }

    printf("M-04 PASS\n");
    return 0;
}

static int test_channel_copyout_rollback(const char *name, const char *device,
                                         unsigned long request_command,
                                         unsigned long release_command,
                                         unsigned int max_channels)
{
    int channels[VENC_MAX_CHN_NUMS > VDEC_MAX_CHN_NUMS ?
                 VENC_MAX_CHN_NUMS : VDEC_MAX_CHN_NUMS];
    unsigned int seen_mask = 0;
    unsigned int acquired = 0;
    unsigned int i;
    int leaked_channel;
    int fd;
    int ret;
    int failures = 0;

    fd = open(device, O_RDWR);
    if (fd < 0) {
        printf("FAIL: open %s copyout probe\n", name);
        return 1;
    }

    ret = ioctl(fd, request_command,
                (void *)(uintptr_t)&readonly_channel_request);
    if (ret == K_SUCCESS) {
        leaked_channel = *(const volatile int *)&readonly_channel_request;
        printf("FAIL: %s RequestChn wrote read-only output, channel=%d\n",
               name, leaked_channel);
        failures++;
        if (leaked_channel >= 0 &&
            (unsigned int)leaked_channel < max_channels)
            (void)ioctl(fd, release_command, &leaked_channel);
    } else {
        printf("PASS: %s RequestChn copyout failed with 0x%x\n",
               name, (unsigned int)ret);
    }
    close(fd);

    fd = open(device, O_RDWR);
    if (fd < 0) {
        printf("FAIL: reopen %s after copyout probe\n", name);
        return failures + 1;
    }

    for (i = 0; i < max_channels; i++) {
        channels[i] = -1;
        ret = ioctl(fd, request_command, &channels[i]);
        if (ret != K_SUCCESS) {
            printf("FAIL: %s capacity reduced at channel %u, ret=0x%x\n",
                   name, i, (unsigned int)ret);
            failures++;
            break;
        }
        if (channels[i] < 0 ||
            (unsigned int)channels[i] >= max_channels ||
            (seen_mask & (1U << channels[i])) != 0) {
            printf("FAIL: %s returned invalid/duplicate channel %d\n",
                   name, channels[i]);
            failures++;
            break;
        }
        seen_mask |= 1U << channels[i];
        acquired++;
    }

    if (acquired == max_channels)
        printf("PASS: %s retained all %u channel slots\n",
               name, max_channels);

    leaked_channel = -1;
    ret = ioctl(fd, request_command, &leaked_channel);
    if (ret == K_SUCCESS) {
        printf("FAIL: %s allocated channel %d beyond capacity\n",
               name, leaked_channel);
        failures++;
        if (leaked_channel >= 0 &&
            (unsigned int)leaked_channel < max_channels)
            (void)ioctl(fd, release_command, &leaked_channel);
    } else if (acquired == max_channels) {
        printf("PASS: %s reports full capacity exhausted\n", name);
    }

    while (acquired > 0) {
        acquired--;
        if (ioctl(fd, release_command, &channels[acquired]) != K_SUCCESS) {
            printf("FAIL: %s release channel %d\n",
                   name, channels[acquired]);
            failures++;
        }
    }
    close(fd);
    return failures;
}

static int run_vpu_copyout_rollback(void)
{
    int failures = 0;

    printf("M-05 VENC/VDEC request-channel copyout rollback test\n");
    failures += test_channel_copyout_rollback(
        "VENC", VENC_DEVICE, K_IOC_VENC_RequestChn,
        K_IOC_VENC_ReleaseChn, VENC_MAX_CHN_NUMS);
    failures += test_channel_copyout_rollback(
        "VDEC", VDEC_DEVICE, K_IOC_VDEC_RequestChn,
        K_IOC_VDEC_ReleaseChn, VDEC_MAX_CHN_NUMS);

    if (failures != 0) {
        printf("M-05 FAIL: %d failure(s)\n", failures);
        return 1;
    }

    printf("M-05 PASS\n");
    return 0;
}

enum vpu_lifecycle_stage {
    VPU_STAGE_OPEN,
    VPU_STAGE_REQUEST,
    VPU_STAGE_ATTACH,
    VPU_STAGE_CREATE,
    VPU_STAGE_START,
    VPU_STAGE_BUSY_RETRY,
};

struct lifecycle_vb_fixture {
    int fd;
    k_bool initialized;
    k_u32 pool_id;
};

static const char *const lifecycle_stage_names[] = {
    "open", "request", "attach", "create", "start", "busy-retry",
};

static long lifetime_waitpid(pid_t pid, int *status)
{
    long ret;

    do {
        ret = syscall(_NRSYS_waitpid, (long)pid, (long)status, 0);
    } while ((ret == -1 && errno == EINTR) || ret == -EINTR);
    return ret;
}

static void lifecycle_vb_init(struct lifecycle_vb_fixture *fixture)
{
    fixture->fd = -1;
    fixture->initialized = K_FALSE;
    fixture->pool_id = VB_INVALID_POOLID;
}

static int setup_lifecycle_vb(struct lifecycle_vb_fixture *fixture,
                              k_u32 block_count, k_u64 block_size)
{
    k_vb_config config;

    fixture->fd = open(VB_DEVICE, O_RDWR);
    if (fixture->fd < 0)
        return -1;

    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = TEST_MAX_POOL_CNT;
    if (ioctl(fixture->fd, K_VB_SetConfig, &config) != K_SUCCESS ||
        ioctl(fixture->fd, K_VB_Init) != K_SUCCESS)
        return -1;
    fixture->initialized = K_TRUE;

    return vb_create_private_pool(fixture->fd, block_count, block_size,
                                  &fixture->pool_id);
}

static int cleanup_lifecycle_vb(struct lifecycle_vb_fixture *fixture)
{
    int failures = 0;

    if (fixture->fd >= 0 && fixture->pool_id != VB_INVALID_POOLID) {
        if (ioctl(fixture->fd, K_VB_DestoryPool,
                  &fixture->pool_id) != K_SUCCESS)
            failures++;
        else
            fixture->pool_id = VB_INVALID_POOLID;
    }
    if (fixture->fd >= 0 && fixture->initialized) {
        if (ioctl(fixture->fd, K_VB_Exit) != K_SUCCESS)
            failures++;
        else
            fixture->initialized = K_FALSE;
    }
    if (fixture->fd >= 0) {
        failures += close(fixture->fd) != 0;
        fixture->fd = -1;
    }
    return failures == 0 ? 0 : -1;
}

static int child_setup_venc(enum vpu_lifecycle_stage stage, k_u32 pool_id)
{
    venc_chn_pool_info pool_info;
    venc_chn_info channel_info;
    int channel = -1;
    int fd;

    fd = open(VENC_DEVICE, O_RDWR);
    if (fd < 0)
        return 1;
    if (stage == VPU_STAGE_OPEN)
        return 0;
    if (ioctl(fd, K_IOC_VENC_RequestChn, &channel) != K_SUCCESS)
        return 2;
    if (stage == VPU_STAGE_REQUEST)
        return 0;

    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.chn_num = (k_u32)channel;
    pool_info.pool_id = pool_id;
    pool_info.buff_num = TEST_VENC_OUTPUT_COUNT;
    if (ioctl(fd, K_IOC_VENC_AttachVbPool, &pool_info) != K_SUCCESS)
        return 3;
    if (stage == VPU_STAGE_ATTACH)
        return 0;

    memset(&channel_info, 0, sizeof(channel_info));
    channel_info.chn_num = (k_u32)channel;
    channel_info.attr.venc_attr.type = K_PT_JPEG;
    channel_info.attr.venc_attr.pic_width = TEST_VENC_WIDTH;
    channel_info.attr.venc_attr.pic_height = TEST_VENC_HEIGHT;
    channel_info.attr.rc_attr.rc_mode = K_VENC_RC_MODE_MJPEG_FIXQP;
    channel_info.attr.rc_attr.mjpeg_fixqp.src_frame_rate = 30;
    channel_info.attr.rc_attr.mjpeg_fixqp.dst_frame_rate = 30;
    channel_info.attr.rc_attr.mjpeg_fixqp.q_factor = 75;
    if (ioctl(fd, K_IOC_VENC_CreateChn, &channel_info) != K_SUCCESS)
        return 4;
    if (stage == VPU_STAGE_CREATE)
        return 0;
    if (ioctl(fd, K_IOC_VENC_StartChn, &channel) != K_SUCCESS)
        return 5;
    if (stage == VPU_STAGE_START)
        return 0;
    if (ioctl(fd, K_IOC_VENC_TEST_CLEANUP_BUSY_ONCE,
              &channel) != K_SUCCESS)
        return 6;
    return 0;
}

static int child_setup_vdec(enum vpu_lifecycle_stage stage, k_u32 pool_id)
{
    vdec_chn_pool_info pool_info;
    vdec_chn_info channel_info;
    int channel = -1;
    int fd;

    fd = open(VDEC_DEVICE, O_RDWR);
    if (fd < 0)
        return 1;
    if (stage == VPU_STAGE_OPEN)
        return 0;
    if (ioctl(fd, K_IOC_VDEC_RequestChn, &channel) != K_SUCCESS)
        return 2;
    if (stage == VPU_STAGE_REQUEST)
        return 0;

    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.chn_num = (k_u32)channel;
    pool_info.pool_id = pool_id;
    if (ioctl(fd, K_IOC_VDEC_AttachVbPool, &pool_info) != K_SUCCESS)
        return 3;
    if (stage == VPU_STAGE_ATTACH)
        return 0;

    memset(&channel_info, 0, sizeof(channel_info));
    channel_info.chn_num = (k_u32)channel;
    channel_info.attr.type = K_PT_JPEG;
    channel_info.attr.mode = K_VDEC_SEND_MODE_FRAME;
    channel_info.attr.pic_width = TEST_VENC_WIDTH;
    channel_info.attr.pic_height = TEST_VENC_HEIGHT;
    channel_info.attr.stream_buf_size = TEST_VDEC_OUTPUT_SIZE;
    if (ioctl(fd, K_IOC_VDEC_CreateChn, &channel_info) != K_SUCCESS)
        return 4;
    if (stage == VPU_STAGE_CREATE)
        return 0;
    if (ioctl(fd, K_IOC_VDEC_StartChn, &channel) != K_SUCCESS)
        return 5;
    if (stage == VPU_STAGE_START)
        return 0;
    if (ioctl(fd, K_IOC_VDEC_TEST_CLEANUP_BUSY_ONCE,
              &channel) != K_SUCCESS)
        return 6;
    return 0;
}

static int wait_child_ready(int fd, unsigned char *result)
{
    unsigned int i;
    int flags;

    flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0)
        return -1;

    for (i = 0; i < TEST_CHILD_READY_RETRIES; i++) {
        ssize_t count = read(fd, result, 1);

        if (count == 1)
            return 0;
        if (count == 0)
            return -1;
        if (errno != EAGAIN && errno != EINTR)
            return -1;
        usleep(100000);
    }
    return -1;
}

static int verify_lifecycle_capacity(const char *name, int fd,
                                     unsigned long request_command,
                                     unsigned long release_command,
                                     unsigned int max_channels)
{
    int channels[VENC_MAX_CHN_NUMS > VDEC_MAX_CHN_NUMS ?
                 VENC_MAX_CHN_NUMS : VDEC_MAX_CHN_NUMS];
    unsigned int acquired = 0;
    unsigned int seen_mask = 0;
    unsigned int i;
    int failures = 0;

    for (i = 0; i < max_channels; i++) {
        channels[i] = -1;
        if (ioctl(fd, request_command, &channels[i]) != K_SUCCESS ||
            channels[i] < 0 ||
            (unsigned int)channels[i] >= max_channels ||
            (seen_mask & (1U << channels[i])) != 0) {
            failures++;
            break;
        }
        seen_mask |= 1U << channels[i];
        acquired++;
    }

    while (acquired > 0) {
        acquired--;
        if (ioctl(fd, release_command, &channels[acquired]) != K_SUCCESS)
            failures++;
    }
    if (failures != 0)
        printf("FAIL: %s channel capacity after forced exit\n", name);
    return failures == 0 ? 0 : -1;
}

static int run_lifecycle_kill_stage(
    const char *name, enum vpu_lifecycle_stage stage, k_u32 pool_id,
    int guard_fd, unsigned long request_command,
    unsigned long release_command, unsigned int max_channels,
    int (*child_setup)(enum vpu_lifecycle_stage, k_u32))
{
    int pipe_fd[2];
    unsigned char child_result = 0xff;
    pid_t child;
    int status = 0;
    int ready;

    if (pipe(pipe_fd) != 0) {
        printf("FAIL: %s %s pipe\n", name, lifecycle_stage_names[stage]);
        return -1;
    }
    fflush(NULL);
    child = fork();
    if (child < 0) {
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        printf("FAIL: %s %s fork\n", name, lifecycle_stage_names[stage]);
        return -1;
    }
    if (child == 0) {
        close(pipe_fd[0]);
        child_result = (unsigned char)child_setup(stage, pool_id);
        (void)write(pipe_fd[1], &child_result, 1);
        close(pipe_fd[1]);
        if (child_result != 0)
            _Exit(1);
        for (;;)
            pause();
    }

    close(pipe_fd[1]);
    ready = wait_child_ready(pipe_fd[0], &child_result);
    close(pipe_fd[0]);
    if (ready != 0 || child_result != 0) {
        (void)kill(child, SIGKILL);
        (void)lifetime_waitpid(child, &status);
        printf("FAIL: %s %s child setup, step=%u\n", name,
               lifecycle_stage_names[stage], child_result);
        return -1;
    }

    if (kill(child, SIGKILL) != 0 ||
        lifetime_waitpid(child, &status) != child ||
        !WIFSIGNALED(status) || WTERMSIG(status) != SIGKILL) {
        printf("FAIL: %s %s SIGKILL/reap\n", name,
               lifecycle_stage_names[stage]);
        return -1;
    }
    if (verify_lifecycle_capacity(name, guard_fd, request_command,
                                  release_command, max_channels) != 0)
        return -1;

    printf("PASS: %s %-10s cleanup retained full capacity\n", name,
           lifecycle_stage_names[stage]);
    return 0;
}

static int run_module_lifecycle_kills(
    const char *name, const char *device, k_u32 pool_count, k_u64 pool_size,
    unsigned long request_command, unsigned long release_command,
    unsigned int max_channels,
    int (*child_setup)(enum vpu_lifecycle_stage, k_u32))
{
    struct lifecycle_vb_fixture vb;
    enum vpu_lifecycle_stage stage;
    int guard_fd = -1;
    int failures = 0;

    lifecycle_vb_init(&vb);
    if (setup_lifecycle_vb(&vb, pool_count, pool_size) != 0) {
        printf("FAIL: %s lifecycle VB setup\n", name);
        failures++;
        goto cleanup;
    }
    guard_fd = open(device, O_RDWR);
    if (guard_fd < 0) {
        printf("FAIL: %s guard open\n", name);
        failures++;
        goto cleanup;
    }

    for (stage = VPU_STAGE_OPEN; stage <= VPU_STAGE_BUSY_RETRY; stage++) {
        if (run_lifecycle_kill_stage(
                name, stage, vb.pool_id, guard_fd, request_command,
                release_command, max_channels, child_setup) != 0) {
            failures++;
            break;
        }
    }

cleanup:
    if (guard_fd >= 0 && close(guard_fd) != 0)
        failures++;
    if (cleanup_lifecycle_vb(&vb) != 0)
        failures++;
    return failures == 0 ? 0 : -1;
}

static int run_vpu_lifecycle_kills(void)
{
    int failures = 0;

    printf("M-06 VENC/VDEC forced-exit lifecycle test\n");
    failures += run_module_lifecycle_kills(
        "VENC", VENC_DEVICE, TEST_VENC_OUTPUT_COUNT,
        TEST_VENC_OUTPUT_SIZE, K_IOC_VENC_RequestChn,
        K_IOC_VENC_ReleaseChn, VENC_MAX_CHN_NUMS,
        child_setup_venc) != 0;
    if (failures == 0)
        failures += run_module_lifecycle_kills(
            "VDEC", VDEC_DEVICE, TEST_VDEC_OUTPUT_COUNT,
            TEST_VDEC_OUTPUT_SIZE, K_IOC_VDEC_RequestChn,
            K_IOC_VDEC_ReleaseChn, VDEC_MAX_CHN_NUMS,
            child_setup_vdec) != 0;

    if (failures != 0) {
        printf("M-06 FAIL: %d failure(s)\n", failures);
        return 1;
    }

    printf("M-06 PASS\n");
    return 0;
}

static int sys_bind_query_matches(int fd, const k_sys_bind_args *expected)
{
    k_sys_bind_args query;

    memset(&query, 0, sizeof(query));
    query.dest_chn = expected->dest_chn;
    if (ioctl(fd, K_IOC_SYS_GETBINDBYDEST, &query) != K_SUCCESS) {
        printf("FAIL: query binding\n");
        return -1;
    }
    if (memcmp(&query.src_chn, &expected->src_chn,
               sizeof(query.src_chn)) != 0) {
        printf("FAIL: queried source does not match owner A binding\n");
        return -1;
    }

    return 0;
}

static int run_sys_bind_owner_isolation(void)
{
    k_sys_bind_args binding = {
        .src_chn = { .mod_id = K_ID_VI, .dev_id = 0, .chn_id = 0 },
        .dest_chn = { .mod_id = K_ID_VENC, .dev_id = 0, .chn_id = 0 },
    };
    int owner_a = -1;
    int owner_b = -1;
    int owner_a_bound = 0;
    int owner_b_bound = 0;
    int ret;
    int failures = 0;

    printf("M-07 SYS bind owner-isolation test\n");
    owner_a = open(SYS_DEVICE, O_RDWR);
    owner_b = open(SYS_DEVICE, O_RDWR);
    if (owner_a < 0 || owner_b < 0) {
        printf("FAIL: open SYS owners A/B\n");
        failures++;
        goto cleanup;
    }
    printf("owners: A fd=%d, B fd=%d\n", owner_a, owner_b);

    ret = ioctl(owner_a, K_IOC_SYS_BIND_CTRL, &binding);
    if (ret != K_SUCCESS) {
        printf("FAIL: owner A bind returned 0x%x\n", (unsigned int)ret);
        failures++;
        goto cleanup;
    }
    owner_a_bound = 1;
    printf("PASS: owner A created VI(0,0) -> VENC(0,0) binding\n");

    ret = ioctl(owner_b, K_IOC_SYS_UNBIND_CTRL, &binding);
    if (ret != K_ERR_SYS_NOT_PERM) {
        printf("FAIL: owner B unbind returned 0x%x, expected 0x%x\n",
               (unsigned int)ret, (unsigned int)K_ERR_SYS_NOT_PERM);
        failures++;
        if (ret == K_SUCCESS)
            owner_a_bound = 0;
        goto cleanup;
    }
    printf("PASS: owner B unbind rejected with 0x%x\n",
           (unsigned int)ret);

    if (sys_bind_query_matches(owner_b, &binding) != 0) {
        failures++;
        goto cleanup;
    }
    printf("PASS: owner A binding remained active\n");

    ret = ioctl(owner_a, K_IOC_SYS_UNBIND_CTRL, &binding);
    if (ret != K_SUCCESS) {
        printf("FAIL: owner A unbind returned 0x%x\n", (unsigned int)ret);
        failures++;
        goto cleanup;
    }
    owner_a_bound = 0;
    printf("PASS: owner A unbound its binding\n");

    ret = ioctl(owner_a, K_IOC_SYS_BIND_CTRL, &binding);
    if (ret != K_SUCCESS) {
        printf("FAIL: owner A rebind returned 0x%x\n", (unsigned int)ret);
        failures++;
        goto cleanup;
    }
    owner_a_bound = 1;
    printf("PASS: owner A recreated its binding\n");

    close(owner_a);
    owner_a = -1;
    owner_a_bound = 0;

    ret = ioctl(owner_b, K_IOC_SYS_BIND_CTRL, &binding);
    if (ret != K_SUCCESS) {
        printf("FAIL: owner B bind after owner A close returned 0x%x\n",
               (unsigned int)ret);
        failures++;
        goto cleanup;
    }
    owner_b_bound = 1;
    printf("PASS: owner A close released its binding\n");

    ret = ioctl(owner_b, K_IOC_SYS_UNBIND_CTRL, &binding);
    if (ret != K_SUCCESS) {
        printf("FAIL: owner B cleanup unbind returned 0x%x\n",
               (unsigned int)ret);
        failures++;
        goto cleanup;
    }
    owner_b_bound = 0;
    printf("PASS: owner B unbound its own binding\n");

cleanup:
    if (owner_a_bound && owner_a >= 0)
        (void)ioctl(owner_a, K_IOC_SYS_UNBIND_CTRL, &binding);
    if (owner_b_bound && owner_b >= 0)
        (void)ioctl(owner_b, K_IOC_SYS_UNBIND_CTRL, &binding);
    if (owner_a >= 0)
        close(owner_a);
    if (owner_b >= 0)
        close(owner_b);

    if (failures != 0) {
        printf("M-07 FAIL: %d failure(s)\n", failures);
        return 1;
    }

    printf("M-07 PASS\n");
    return 0;
}

static int vicap_set_mclk(int fd, k_u8 enable)
{
    k_vicap_mclk mclk = {
        .id = VICAP_MCLK0,
        .mclk_sel = VICAP_PLL1_CLK_DIV4,
        .mclk_div = 25,
        .mclk_en = enable,
    };

    return ioctl(fd, K_IOC_VICAP_SetMclk, &mclk);
}

static int run_vicap_pm_mclk_ownership(void)
{
    k_u8 reset = 0;
    int owner_a = -1;
    int owner_b = -1;
    int owner_a_mclk = 0;
    int owner_b_mclk = 0;
    int ret;
    int failures = 0;

    printf("M-08 VICAP PM/MCLK ownership test\n");
    owner_a = open(VICAP_DEVICE, O_RDWR);
    owner_b = open(VICAP_DEVICE, O_RDWR);
    if (owner_a < 0 || owner_b < 0) {
        printf("FAIL: open VICAP owners A/B\n");
        failures++;
        goto cleanup;
    }
    printf("owners: A fd=%d, B fd=%d\n", owner_a, owner_b);

    ret = vicap_set_mclk(owner_b, K_FALSE);
    if (ret != K_ERR_VICAP_NOT_PERM) {
        printf("FAIL: owner B unmatched MCLK disable returned 0x%x, "
               "expected 0x%x\n", (unsigned int)ret,
               (unsigned int)K_ERR_VICAP_NOT_PERM);
        failures++;
        goto cleanup;
    }
    printf("PASS: owner B unmatched MCLK disable rejected with 0x%x\n",
           (unsigned int)ret);

    if (ioctl(owner_a, K_IOC_VICAP_DevReset, &reset) != K_SUCCESS ||
        ioctl(owner_a, K_IOC_VICAP_DevReset, &reset) != K_SUCCESS) {
        printf("FAIL: owner A repeated PM/reset acquisition\n");
        failures++;
        goto cleanup;
    }
    if (ioctl(owner_b, K_IOC_VICAP_DevReset, &reset) != K_SUCCESS) {
        printf("FAIL: owner B PM/reset acquisition\n");
        failures++;
        goto cleanup;
    }
    printf("PASS: owners A/B hold overlapping VICAP PM references\n");

    if (vicap_set_mclk(owner_a, K_TRUE) != K_SUCCESS) {
        printf("FAIL: owner A MCLK enable\n");
        failures++;
        goto cleanup;
    }
    owner_a_mclk++;
    if (vicap_set_mclk(owner_b, K_TRUE) != K_SUCCESS) {
        printf("FAIL: owner B MCLK enable\n");
        failures++;
        goto cleanup;
    }
    owner_b_mclk++;
    printf("PASS: owners A/B hold overlapping MCLK0 references\n");

    if (vicap_set_mclk(owner_a, K_FALSE) != K_SUCCESS) {
        printf("FAIL: owner A MCLK disable\n");
        failures++;
        goto cleanup;
    }
    owner_a_mclk--;

    ret = vicap_set_mclk(owner_a, K_FALSE);
    if (ret != K_ERR_VICAP_NOT_PERM) {
        printf("FAIL: owner A unmatched MCLK disable returned 0x%x, "
               "expected 0x%x\n", (unsigned int)ret,
               (unsigned int)K_ERR_VICAP_NOT_PERM);
        failures++;
        goto cleanup;
    }
    printf("PASS: owner A cannot consume owner B's MCLK reference\n");

    if (vicap_set_mclk(owner_a, K_TRUE) != K_SUCCESS) {
        printf("FAIL: owner A MCLK re-enable\n");
        failures++;
        goto cleanup;
    }
    owner_a_mclk++;

    if (close(owner_a) != 0) {
        printf("FAIL: close owner A\n");
        owner_a = -1;
        failures++;
        goto cleanup;
    }
    owner_a = -1;
    owner_a_mclk = 0;

    if (ioctl(owner_b, K_IOC_VICAP_DevReset, &reset) != K_SUCCESS) {
        printf("FAIL: owner B reset after owner A close\n");
        failures++;
        goto cleanup;
    }
    if (vicap_set_mclk(owner_b, K_FALSE) != K_SUCCESS) {
        printf("FAIL: owner B MCLK disable after owner A close\n");
        failures++;
        goto cleanup;
    }
    owner_b_mclk--;
    printf("PASS: owner B resources remained valid after owner A close\n");

    ret = vicap_set_mclk(owner_b, K_FALSE);
    if (ret != K_ERR_VICAP_NOT_PERM) {
        printf("FAIL: owner B second unmatched MCLK disable returned 0x%x, "
               "expected 0x%x\n", (unsigned int)ret,
               (unsigned int)K_ERR_VICAP_NOT_PERM);
        failures++;
        goto cleanup;
    }
    printf("PASS: owner B unmatched MCLK disable rejected with 0x%x\n",
           (unsigned int)ret);

    if (vicap_set_mclk(owner_b, K_TRUE) != K_SUCCESS) {
        printf("FAIL: owner B MCLK re-enable before close\n");
        failures++;
        goto cleanup;
    }
    owner_b_mclk++;

    if (close(owner_b) != 0) {
        printf("FAIL: close owner B\n");
        owner_b = -1;
        failures++;
        goto cleanup;
    }
    owner_b = -1;
    owner_b_mclk = 0;
    printf("PASS: owner close cleanup completed\n");

cleanup:
    while (owner_a_mclk > 0 && owner_a >= 0) {
        if (vicap_set_mclk(owner_a, K_FALSE) != K_SUCCESS)
            break;
        owner_a_mclk--;
    }
    while (owner_b_mclk > 0 && owner_b >= 0) {
        if (vicap_set_mclk(owner_b, K_FALSE) != K_SUCCESS)
            break;
        owner_b_mclk--;
    }
    if (owner_a >= 0)
        close(owner_a);
    if (owner_b >= 0)
        close(owner_b);

    if (failures != 0) {
        printf("M-08 FAIL: %d failure(s)\n", failures);
        return 1;
    }

    printf("M-08 PASS\n");
    return 0;
}

enum lifetime_module {
    LIFETIME_VENC,
    LIFETIME_VDEC,
    LIFETIME_VB,
    LIFETIME_VICAP,
    LIFETIME_MODULE_COUNT,
};

static k_u32 lifetime_random(k_u32 *state)
{
    k_u32 value = *state;

    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

static int lifetime_vpu_cycle(const char *name, const char *device,
                              unsigned long request_command,
                              unsigned long release_command,
                              unsigned int max_channels, k_u32 random_value)
{
    int invalid_channel = (int)max_channels;
    int channel = -1;
    int fd;
    int ret;
    int failed = 0;

    fd = open(device, O_RDWR);
    if (fd < 0) {
        printf("FAIL: M-11 open %s\n", name);
        return -1;
    }

    ret = ioctl(fd, release_command, &invalid_channel);
    if (ret == K_SUCCESS) {
        printf("FAIL: M-11 %s invalid release succeeded\n", name);
        failed = 1;
        goto out;
    }

    if ((random_value & 3U) == 0)
        goto out;

    ret = ioctl(fd, request_command, &channel);
    if (ret != K_SUCCESS || channel < 0 ||
        (unsigned int)channel >= max_channels) {
        printf("FAIL: M-11 %s request returned 0x%x, channel=%d\n",
               name, (unsigned int)ret, channel);
        failed = 1;
        goto out;
    }

    if (random_value & 4U) {
        ret = ioctl(fd, release_command, &channel);
        if (ret != K_SUCCESS) {
            printf("FAIL: M-11 %s explicit release returned 0x%x\n",
                   name, (unsigned int)ret);
            failed = 1;
            goto out;
        }
        ret = ioctl(fd, release_command, &channel);
        if (ret == K_SUCCESS) {
            printf("FAIL: M-11 %s duplicate release succeeded\n", name);
            failed = 1;
        }
    }

out:
    if (close(fd) != 0) {
        printf("FAIL: M-11 close %s\n", name);
        failed = 1;
    }
    return failed ? -1 : 0;
}

static int lifetime_vb_cycle(k_u32 random_value)
{
    k_vb_config config;
    k_vb_pool_info pool;
    k_vb_blk_info block;
    k_vb_blk_handle invalid_handle = VB_INVALID_HANDLE;
    k_vb_blk_handle block_handle = VB_INVALID_HANDLE;
    k_u32 pool_id = VB_INVALID_POOLID;
    k_bool initialized = K_FALSE;
    unsigned int sequence = random_value & 3U;
    int fd;
    int ret;
    int failed = 0;

    fd = open(VB_DEVICE, O_RDWR);
    if (fd < 0) {
        printf("FAIL: M-11 open VB\n");
        return -1;
    }

    if (ioctl(fd, K_VB_ReleaseBlock, &invalid_handle) == K_SUCCESS) {
        printf("FAIL: M-11 VB invalid block release succeeded\n");
        failed = 1;
        goto out;
    }
    if (sequence == 0)
        goto out;

    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = TEST_MAX_POOL_CNT;
    if (ioctl(fd, K_VB_SetConfig, &config) != K_SUCCESS ||
        ioctl(fd, K_VB_Init) != K_SUCCESS) {
        printf("FAIL: M-11 initialize VB\n");
        failed = 1;
        goto out;
    }
    initialized = K_TRUE;
    if (sequence == 1)
        goto cleanup_or_close;

    memset(&pool, 0, sizeof(pool));
    pool.blk_cnt = 1;
    pool.blk_size = TEST_BLOCK_SIZE;
    pool.remap_mode = VB_REMAP_MODE_NONE;
    if (ioctl(fd, K_VB_CreatePool, &pool) != K_SUCCESS) {
        printf("FAIL: M-11 create VB pool\n");
        failed = 1;
        goto out;
    }
    pool_id = pool.pool_id;
    if (sequence == 2)
        goto cleanup_or_close;

    memset(&block, 0, sizeof(block));
    block.pool_id = pool_id;
    block.blk_size = TEST_BLOCK_SIZE;
    if (ioctl(fd, K_VB_GetBlock, &block) != K_SUCCESS) {
        printf("FAIL: M-11 get VB block\n");
        failed = 1;
        goto out;
    }
    block_handle = block.blk_handle;

cleanup_or_close:
    if (!(random_value & 4U))
        goto out;

    if (block_handle != VB_INVALID_HANDLE) {
        ret = ioctl(fd, K_VB_ReleaseBlock, &block_handle);
        if (ret != K_SUCCESS) {
            printf("FAIL: M-11 release VB block returned 0x%x\n",
                   (unsigned int)ret);
            failed = 1;
            goto out;
        }
        block_handle = VB_INVALID_HANDLE;
    }
    if (pool_id != VB_INVALID_POOLID) {
        ret = ioctl(fd, K_VB_DestoryPool, &pool_id);
        if (ret != K_SUCCESS) {
            printf("FAIL: M-11 destroy VB pool returned 0x%x\n",
                   (unsigned int)ret);
            failed = 1;
            goto out;
        }
        pool_id = VB_INVALID_POOLID;
    }
    if (initialized) {
        ret = ioctl(fd, K_VB_Exit);
        if (ret != K_SUCCESS) {
            printf("FAIL: M-11 exit VB returned 0x%x\n",
                   (unsigned int)ret);
            failed = 1;
            goto out;
        }
        initialized = K_FALSE;
    }

out:
    if (close(fd) != 0) {
        printf("FAIL: M-11 close VB\n");
        failed = 1;
    }
    return failed ? -1 : 0;
}

static int lifetime_vicap_cycle(k_u32 random_value)
{
    k_u8 reset = 0;
    k_bool mclk_enabled = K_FALSE;
    unsigned int sequence = random_value & 3U;
    int fd;
    int ret;
    int failed = 0;

    fd = open(VICAP_DEVICE, O_RDWR);
    if (fd < 0) {
        printf("FAIL: M-11 open VICAP\n");
        return -1;
    }

    ret = vicap_set_mclk(fd, K_FALSE);
    if (ret != K_ERR_VICAP_NOT_PERM) {
        printf("FAIL: M-11 VICAP unmatched disable returned 0x%x\n",
               (unsigned int)ret);
        failed = 1;
        goto out;
    }

    if (sequence == 1 || sequence == 3) {
        ret = ioctl(fd, K_IOC_VICAP_DevReset, &reset);
        if (ret != K_SUCCESS) {
            printf("FAIL: M-11 VICAP reset returned 0x%x\n",
                   (unsigned int)ret);
            failed = 1;
            goto out;
        }
    }
    if (sequence == 2 || sequence == 3) {
        ret = vicap_set_mclk(fd, K_TRUE);
        if (ret != K_SUCCESS) {
            printf("FAIL: M-11 VICAP MCLK enable returned 0x%x\n",
                   (unsigned int)ret);
            failed = 1;
            goto out;
        }
        mclk_enabled = K_TRUE;
    }

    if (mclk_enabled && (random_value & 4U)) {
        ret = vicap_set_mclk(fd, K_FALSE);
        if (ret != K_SUCCESS) {
            printf("FAIL: M-11 VICAP MCLK disable returned 0x%x\n",
                   (unsigned int)ret);
            failed = 1;
            goto out;
        }
        mclk_enabled = K_FALSE;
        ret = vicap_set_mclk(fd, K_FALSE);
        if (ret != K_ERR_VICAP_NOT_PERM) {
            printf("FAIL: M-11 VICAP duplicate disable returned 0x%x\n",
                   (unsigned int)ret);
            failed = 1;
        }
    }

out:
    if (close(fd) != 0) {
        printf("FAIL: M-11 close VICAP\n");
        failed = 1;
    }
    return failed ? -1 : 0;
}

static int lifetime_verify_vb_capacity(void)
{
    k_vb_config config;
    k_vb_pool_info pool;
    k_u32 pools[TEST_MAX_POOL_CNT];
    unsigned int created = 0;
    unsigned int i;
    int fd;
    int failed = 0;

    fd = open(VB_DEVICE, O_RDWR);
    if (fd < 0)
        return -1;

    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = TEST_MAX_POOL_CNT;
    if (ioctl(fd, K_VB_SetConfig, &config) != K_SUCCESS ||
        ioctl(fd, K_VB_Init) != K_SUCCESS) {
        failed = 1;
        goto out;
    }

    for (i = 0; i < TEST_MAX_POOL_CNT; i++) {
        memset(&pool, 0, sizeof(pool));
        pool.blk_cnt = 1;
        pool.blk_size = TEST_BLOCK_SIZE;
        pool.remap_mode = VB_REMAP_MODE_NONE;
        if (ioctl(fd, K_VB_CreatePool, &pool) != K_SUCCESS) {
            failed = 1;
            break;
        }
        pools[created++] = pool.pool_id;
    }

    if (created == TEST_MAX_POOL_CNT) {
        int ret;

        memset(&pool, 0, sizeof(pool));
        pool.blk_cnt = 1;
        pool.blk_size = TEST_BLOCK_SIZE;
        pool.remap_mode = VB_REMAP_MODE_NONE;
        ret = ioctl(fd, K_VB_CreatePool, &pool);
        if (ret == K_SUCCESS) {
            k_u32 extra_pool = pool.pool_id;

            (void)ioctl(fd, K_VB_DestoryPool, &extra_pool);
            failed = 1;
        } else if (ret != K_ERR_VB_2MPOOLS) {
            printf("FAIL: VB pool exhaustion returned 0x%x, expected 0x%x\n",
                   (unsigned int)ret, (unsigned int)K_ERR_VB_2MPOOLS);
            failed = 1;
        }
    }

    while (created > 0) {
        created--;
        if (ioctl(fd, K_VB_DestoryPool, &pools[created]) != K_SUCCESS)
            failed = 1;
    }
    if (ioctl(fd, K_VB_Exit) != K_SUCCESS)
        failed = 1;

out:
    if (close(fd) != 0)
        failed = 1;
    return failed ? -1 : 0;
}

static int lifetime_verify_capacity(void)
{
    int fd;
    int failed = 0;

    fd = open(VENC_DEVICE, O_RDWR);
    if (fd < 0 || verify_lifecycle_capacity(
            "VENC", fd, K_IOC_VENC_RequestChn,
            K_IOC_VENC_ReleaseChn, VENC_MAX_CHN_NUMS) != 0)
        failed = 1;
    if (fd >= 0 && close(fd) != 0)
        failed = 1;

    fd = open(VDEC_DEVICE, O_RDWR);
    if (fd < 0 || verify_lifecycle_capacity(
            "VDEC", fd, K_IOC_VDEC_RequestChn,
            K_IOC_VDEC_ReleaseChn, VDEC_MAX_CHN_NUMS) != 0)
        failed = 1;
    if (fd >= 0 && close(fd) != 0)
        failed = 1;

    if (lifetime_verify_vb_capacity() != 0)
        failed = 1;
    if (lifetime_vicap_cycle(7U) != 0)
        failed = 1;
    return failed ? -1 : 0;
}

static int run_randomized_lifetime(void)
{
    static const char *const module_names[LIFETIME_MODULE_COUNT] = {
        "VENC", "VDEC", "VB", "VICAP",
    };
    unsigned int opens[LIFETIME_MODULE_COUNT] = { 0 };
    k_u32 random_state = TEST_LIFETIME_SEED;
    unsigned int iteration;

    printf("M-11 randomized media lifetime test: iterations=%u seed=0x%x\n",
           TEST_LIFETIME_ITERATIONS, TEST_LIFETIME_SEED);
    printf("DPU: SKIP (unavailable by user decision)\n");

    for (iteration = 0; iteration < TEST_LIFETIME_ITERATIONS; iteration++) {
        enum lifetime_module order[LIFETIME_MODULE_COUNT] = {
            LIFETIME_VENC, LIFETIME_VDEC, LIFETIME_VB, LIFETIME_VICAP,
        };
        unsigned int i;

        for (i = LIFETIME_MODULE_COUNT - 1; i > 0; i--) {
            unsigned int selected = lifetime_random(&random_state) % (i + 1);
            enum lifetime_module saved = order[i];

            order[i] = order[selected];
            order[selected] = saved;
        }

        for (i = 0; i < LIFETIME_MODULE_COUNT; i++) {
            enum lifetime_module module = order[i];
            k_u32 random_value = lifetime_random(&random_state);
            int ret;

            switch (module) {
            case LIFETIME_VENC:
                ret = lifetime_vpu_cycle(
                    "VENC", VENC_DEVICE, K_IOC_VENC_RequestChn,
                    K_IOC_VENC_ReleaseChn, VENC_MAX_CHN_NUMS, random_value);
                break;
            case LIFETIME_VDEC:
                ret = lifetime_vpu_cycle(
                    "VDEC", VDEC_DEVICE, K_IOC_VDEC_RequestChn,
                    K_IOC_VDEC_ReleaseChn, VDEC_MAX_CHN_NUMS, random_value);
                break;
            case LIFETIME_VB:
                ret = lifetime_vb_cycle(random_value);
                break;
            case LIFETIME_VICAP:
                ret = lifetime_vicap_cycle(random_value);
                break;
            default:
                ret = -1;
                break;
            }
            if (ret != 0) {
                printf("M-11 FAIL: cycle=%u module=%s\n", iteration + 1,
                       module_names[module]);
                return 1;
            }
            opens[module]++;
        }

        if ((iteration + 1) % TEST_LIFETIME_CAPACITY_INTERVAL == 0) {
            if (lifetime_verify_capacity() != 0) {
                printf("M-11 FAIL: capacity check after cycle %u\n",
                       iteration + 1);
                return 1;
            }
            printf("PASS: %u cycles; VENC/VDEC/VB capacity intact\n",
                   iteration + 1);
        }
    }

    printf("opens: VENC=%u VDEC=%u VB=%u VICAP=%u DPU=SKIP\n",
           opens[LIFETIME_VENC], opens[LIFETIME_VDEC], opens[LIFETIME_VB],
           opens[LIFETIME_VICAP]);
    printf("M-11 PASS\n");
    return 0;
}

static int run_invalid_user_pointer(void)
{
    static const struct {
        const char *name;
        unsigned long command;
    } requests[] = {
        { "SetConfig", K_VB_SetConfig },
        { "GetConfig", K_VB_GetConfig },
        { "CreatePool", K_VB_CreatePool },
        { "DestroyPool", K_VB_DestoryPool },
        { "GetBlock", K_VB_GetBlock },
        { "ReleaseBlock", K_VB_ReleaseBlock },
        { "PhysAddr2Handle", K_VB_PhysAddr2Handle },
        { "Handle2PhysAddr", K_VB_Handle2PhysAddr },
        { "Handle2PoolId", K_VB_Handle2PoolId },
        { "InquireUserCnt", K_VB_InquireUserCnt },
        { "GetSupplementAttr", K_VB_GetSupplementAttr },
        { "SetSupplementConfig", K_VB_SetSupplementConfig },
        { "GetSupplementConfig", K_VB_GetSupplementConfig },
        { "IncBlockRefCnt", K_VB_IncBlockRefCnt },
        { "DecBlockRefCnt", K_VB_DecBlockRefCnt },
        { "GetBlockRefCnt", K_VB_GetBlockRefCnt },
    };
    struct vb_fixture fixture;
    void *invalid_pointer = (void *)(uintptr_t)1;
    size_t i;
    int failures = 0;

    fixture_init(&fixture);
    if (open_owners(&fixture) != 0)
        return 1;

    for (i = 0; i < sizeof(requests) / sizeof(requests[0]); i++) {
        failures += ioctl_expect_error(fixture.owner_a, "A", requests[i].name,
                                       requests[i].command,
                                       invalid_pointer) != 0;
        failures += ioctl_expect_error(fixture.owner_b, "B", requests[i].name,
                                       requests[i].command,
                                       invalid_pointer) != 0;
    }

    failures += cleanup_vb_fixture(&fixture) != 0;
    if (failures != 0) {
        printf("invalid-user-pointer FAIL: %d failure(s)\n", failures);
        return 1;
    }

    printf("invalid-user-pointer PASS\n");
    return 0;
}

static void usage(const char *program)
{
    printf("usage: %s <vb-invalid-handles|vb-owner-isolation|"
           "vb-copyout-rollback|venc-owner-isolation|"
           "vpu-copyout-rollback|vpu-lifecycle-kills|"
           "sys-bind-owner-isolation|vicap-pm-mclk-ownership|"
           "randomized-lifetime|"
           "invalid-user-pointer>\n", program);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        usage(argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "vb-invalid-handles") == 0)
        return run_vb_invalid_handles();
    if (strcmp(argv[1], "vb-owner-isolation") == 0)
        return run_vb_owner_isolation();
    if (strcmp(argv[1], "vb-copyout-rollback") == 0)
        return run_vb_copyout_rollback();
    if (strcmp(argv[1], "venc-owner-isolation") == 0)
        return run_venc_owner_isolation();
    if (strcmp(argv[1], "vpu-copyout-rollback") == 0)
        return run_vpu_copyout_rollback();
    if (strcmp(argv[1], "vpu-lifecycle-kills") == 0)
        return run_vpu_lifecycle_kills();
    if (strcmp(argv[1], "sys-bind-owner-isolation") == 0)
        return run_sys_bind_owner_isolation();
    if (strcmp(argv[1], "vicap-pm-mclk-ownership") == 0)
        return run_vicap_pm_mclk_ownership();
    if (strcmp(argv[1], "randomized-lifetime") == 0)
        return run_randomized_lifetime();
    if (strcmp(argv[1], "invalid-user-pointer") == 0)
        return run_invalid_user_pointer();

    usage(argv[0]);
    return 2;
}
