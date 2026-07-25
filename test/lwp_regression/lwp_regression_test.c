/*
 * LWP memory-management, pthread, and kernel-interface regression test.
 *
 * Build this source as a normal RT-Smart userspace application and run it on
 * the target. It has a local Makefile but no Kconfig or SDK integration.
 */

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "canmv_misc.h"
#include "hal_syscall.h"

#define TLS_WORKAROUND_LENGTH_0 0x1b0
#define TLS_WORKAROUND_LENGTH_1 0xb70

#define MUTEX_WORKERS 4
#define MUTEX_ITERATIONS 2000
#define MAP_AVL_CHURN_COUNT 31
#define HEAP_PROBE_MAP_COUNT 48
#define MM_STRESS_WORKERS 4
#define MM_STRESS_ITERATIONS 32
#define DETACHED_WORKERS 8
#define WAIT_TIMEOUT_MS 1000
#define CUSTOM_STACK_SIZE (64U * 1024U)

/* PMUTEX is an RT-Smart-private pthread syscall interface. */
#define PMUTEX_INIT 0
#define PMUTEX_LOCK 1
#define PMUTEX_UNLOCK 2
#define PMUTEX_DESTROY 3

/* RT-Smart currently implements FUTEX_WAKE for its userspace futex syscall. */
#define FUTEX_WAKE 1

struct lwp_test_video_pool_cfg
{
    uint32_t buffer_size;
    uint32_t buffer_count;
};

struct lwp_test_video_buffer
{
    void *user_buffer;
    uint32_t buffer_size;
};

struct mutex_stress
{
    pthread_mutex_t mutex;
    unsigned int counter;
};

struct mutex_waiter
{
    pthread_mutex_t *mutex;
    volatile unsigned int entered;
    volatile unsigned int completed;
    int result;
};

struct lifecycle_state
{
    volatile unsigned int completed;
};

struct mmap_stress_state
{
    size_t page_size;
    volatile int failure;
};

static int fail(const char *test, int line, const char *message)
{
    printf("[FAIL] %s:%d: %s (errno=%d)\n", test, line, message, errno);
    return -1;
}

#define CHECK(test, condition, message) \
    do \
    { \
        if (!(condition)) \
        { \
            return fail((test), __LINE__, (message)); \
        } \
    } while (0)

static int has_expected_error(long result, int expected)
{
    return result == -expected || (result == -1 && errno == expected);
}

static void sleep_milliseconds(long milliseconds)
{
    struct timespec delay;

    delay.tv_sec = milliseconds / 1000;
    delay.tv_nsec = (milliseconds % 1000) * 1000000L;
    while (nanosleep(&delay, &delay) != 0 && errno == EINTR)
    {
    }
}

static int wait_for_count(volatile unsigned int *count, unsigned int expected)
{
    unsigned int elapsed;

    for (elapsed = 0; elapsed < WAIT_TIMEOUT_MS; elapsed++)
    {
        if (*count >= expected)
        {
            return 0;
        }
        sleep_milliseconds(1);
    }
    return -1;
}

static long pmutex_syscall(void *mutex, int operation, void *argument)
{
    return syscall(_NRSYS_pmutex, (long)mutex, (long)operation, (long)argument);
}

static long futex_wake_syscall(int *address)
{
    return syscall(_NRSYS_futex, (long)address, FUTEX_WAKE, 1, 0, 0, 0);
}

static long lwp_waitpid_syscall(pid_t pid, int *status, int options)
{
    return syscall(_NRSYS_waitpid, (long)pid, (long)status, (long)options);
}

static long lwp_shmget_syscall(key_t key, size_t size, int create)
{
    return syscall(_NRSYS_shmget, (long)key, (long)size, (long)create);
}

static void *lwp_shmat_syscall(int id, void *address)
{
    return (void *)syscall(_NRSYS_shmat, (long)id, (long)address);
}

static long lwp_shmdt_syscall(void *address)
{
    return syscall(_NRSYS_shmdt, (long)address);
}

static long lwp_shmrm_syscall(int id)
{
    return syscall(_NRSYS_shmrm, (long)id);
}

static int heap_info_is_consistent(const struct canmv_misc_dev_meminfo_t *info)
{
    return info->total_size > 0 &&
           info->used_size <= info->total_size &&
           info->free_size <= info->total_size &&
           info->used_size == info->total_size - info->free_size;
}

static int test_heap_info_current_usage(size_t page_size)
{
    const char *test = "heap-info-current-usage";
    struct canmv_misc_dev_meminfo_t before;
    struct canmv_misc_dev_meminfo_t during;
    struct canmv_misc_dev_meminfo_t after;
    void *mappings[HEAP_PROBE_MAP_COUNT] = { 0 };
    int i;

    CHECK(test, canmv_misc_get_sys_heap_size(&before) == 0,
          "initial heap-info query failed");
    CHECK(test, heap_info_is_consistent(&before),
          "initial heap-info values are inconsistent");

    /* Each anonymous map creates kernel map-area metadata from the heap. */
    for (i = 0; i < HEAP_PROBE_MAP_COUNT; i++)
    {
        mappings[i] = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(test, mappings[i] != MAP_FAILED, "heap probe mmap failed");
    }

    CHECK(test, canmv_misc_get_sys_heap_size(&during) == 0,
          "heap-info query during allocation failed");
    CHECK(test, heap_info_is_consistent(&during),
          "heap-info reports a historical high-water mark as current usage");

    for (i = 0; i < HEAP_PROBE_MAP_COUNT; i++)
    {
        CHECK(test, munmap(mappings[i], page_size) == 0,
              "heap probe munmap failed");
    }

    CHECK(test, canmv_misc_get_sys_heap_size(&after) == 0,
          "heap-info query after free failed");
    CHECK(test, heap_info_is_consistent(&after),
          "heap-info did not return to current usage after free");

    printf("[PASS] %s\n", test);
    return 0;
}

static int test_split_and_multi_area_munmap(size_t page_size)
{
    const char *test = "split-and-multi-area-munmap";
    volatile unsigned char *mapping;

    mapping = mmap(NULL, page_size * 3, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(test, mapping != MAP_FAILED, "initial mmap failed");

    mapping[0] = 0x3c;
    mapping[page_size * 2] = 0xc3;

    CHECK(test, munmap((void *)(mapping + page_size), page_size) == 0,
          "middle-page munmap failed");
    CHECK(test, mapping[0] == 0x3c, "mapping before the unmapped page was lost");
    CHECK(test, mapping[page_size * 2] == 0xc3,
          "mapping after the unmapped page was lost");

    /* This range contains two map areas separated by the page just unmapped. */
    CHECK(test, munmap((void *)mapping, page_size * 3) == 0,
          "munmap spanning multiple map areas failed");

    printf("[PASS] %s\n", test);
    return 0;
}

static int test_partial_boundary_munmap(size_t page_size)
{
    const char *test = "partial-boundary-munmap";
    volatile unsigned char *mapping;

    mapping = mmap(NULL, page_size * 3, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(test, mapping != MAP_FAILED, "initial mmap failed");
    mapping[page_size] = 0x7b;
    mapping[page_size * 2] = 0xb7;

    /* Trim the front, then the tail, so the same AVL node is rekeyed and shrunk. */
    CHECK(test, munmap((void *)mapping, page_size) == 0,
          "front-page munmap failed");
    CHECK(test, mapping[page_size] == 0x7b && mapping[page_size * 2] == 0xb7,
          "front-page munmap altered the retained range");
    CHECK(test, munmap((void *)(mapping + page_size * 2), page_size) == 0,
          "tail-page munmap failed");
    CHECK(test, mapping[page_size] == 0x7b,
          "tail-page munmap altered the retained range");
    CHECK(test, munmap((void *)(mapping + page_size), page_size) == 0,
          "final retained-page munmap failed");

    printf("[PASS] %s\n", test);
    return 0;
}

static int test_map_area_avl_churn(size_t page_size)
{
    const char *test = "map-area-avl-churn";
    volatile unsigned int *mappings[MAP_AVL_CHURN_COUNT] = { 0 };
    int i;

    for (i = 0; i < MAP_AVL_CHURN_COUNT; i++)
    {
        mappings[i] = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(test, mappings[i] != MAP_FAILED, "mmap for AVL churn failed");
        mappings[i][0] = 0x4c575000U + (unsigned int)i;
    }

    /* Remove nodes in a non-tree-order sequence to exercise AVL rebalancing. */
    for (i = 0; i < MAP_AVL_CHURN_COUNT; i++)
    {
        int index = (i * 7) % MAP_AVL_CHURN_COUNT;

        CHECK(test, mappings[index][0] == 0x4c575000U + (unsigned int)index,
              "AVL node lookup returned the wrong map area");
        CHECK(test, munmap((void *)mappings[index], page_size) == 0,
              "AVL node removal failed");
    }

    printf("[PASS] %s\n", test);
    return 0;
}

static int test_munmap_validation(size_t page_size)
{
    const char *test = "munmap-validation";
    volatile unsigned char *mapping;

    mapping = mmap(NULL, page_size * 2, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(test, mapping != MAP_FAILED, "initial mmap failed");
    mapping[0] = 0x5a;

    errno = 0;
    CHECK(test, has_expected_error(munmap(NULL, page_size), EINVAL),
          "munmap(NULL) did not fail with EINVAL");
    errno = 0;
    CHECK(test, has_expected_error(munmap((void *)(mapping + 1), page_size), EINVAL),
          "unaligned munmap did not fail with EINVAL");
    errno = 0;
    CHECK(test, has_expected_error(munmap((void *)mapping, 0), EINVAL),
          "zero-length munmap did not fail with EINVAL");
    errno = 0;
    CHECK(test, has_expected_error(munmap((void *)(uintptr_t)UINTPTR_MAX, page_size), EINVAL),
          "out-of-range munmap did not fail with EINVAL");
    errno = 0;
    CHECK(test, has_expected_error(munmap((void *)mapping, SIZE_MAX), EINVAL),
          "overflowing munmap length did not fail with EINVAL");
    CHECK(test, mapping[0] == 0x5a,
          "an invalid munmap altered a valid mapping");
    CHECK(test, munmap((void *)mapping, page_size * 2) == 0,
          "valid munmap after validation failures failed");

    printf("[PASS] %s\n", test);
    return 0;
}

static int test_tls_munmap_metadata_case(size_t page_size, size_t requested_length,
                                         size_t unmap_length)
{
    const char *test = "tls-munmap-metadata";
    void *mapping;
    void *replacement;
    void *hidden_page;

    mapping = mmap(NULL, requested_length, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(test, mapping != MAP_FAILED, "TLS-workaround mmap failed");

    memset(mapping, 0xa5, requested_length);
    hidden_page = (void *)((char *)mapping - page_size);

    CHECK(test, munmap(mapping, unmap_length) == 0,
          "TLS-workaround munmap failed");

    replacement = mmap(hidden_page, page_size * 2, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(test, replacement != MAP_FAILED,
          "TLS-workaround backing range was not fully released");
    CHECK(test, replacement == hidden_page,
          "replacement mmap did not use the released backing range");
    CHECK(test, munmap(replacement, page_size * 2) == 0,
          "replacement munmap failed");

    printf("[PASS] %s-0x%zx\n", test, requested_length);
    return 0;
}

static int test_tls_munmap_metadata(size_t page_size)
{
    const char *test = "tls-metadata-after-head-trim";
    void *mapping;
    void *hidden_page;
    void *replacement;

    if (test_tls_munmap_metadata_case(page_size, TLS_WORKAROUND_LENGTH_0,
                                      page_size) != 0)
    {
        return -1;
    }
    if (test_tls_munmap_metadata_case(page_size, TLS_WORKAROUND_LENGTH_1, 1) != 0)
    {
        return -1;
    }

    mapping = mmap(NULL, TLS_WORKAROUND_LENGTH_0, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(test, mapping != MAP_FAILED, "TLS-workaround mmap failed");
    hidden_page = (void *)((char *)mapping - page_size);
    CHECK(test, munmap(hidden_page, page_size) == 0,
          "TLS backing-page munmap failed");
    CHECK(test, munmap(mapping, page_size) == 0,
          "TLS returned-page munmap failed after backing-page trim");
    replacement = mmap(hidden_page, page_size * 2, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(test, replacement == hidden_page,
          "TLS metadata was not retained after map-area rekeying");
    CHECK(test, munmap(replacement, page_size * 2) == 0,
          "TLS replacement munmap failed after map-area rekeying");

    printf("[PASS] %s\n", test);
    return 0;
}

static int tls_metadata_child_check(void *mapping, void *hidden_page, size_t page_size)
{
    pthread_mutex_t mutex;
    void *replacement;

    memset(&mutex, 0, sizeof(mutex));
    if (pthread_mutex_init(&mutex, NULL) != 0 ||
        pthread_mutex_lock(&mutex) != 0 ||
        pthread_mutex_unlock(&mutex) != 0 ||
        pthread_mutex_destroy(&mutex) != 0)
    {
        return 1;
    }

    if (munmap(mapping, page_size) != 0)
    {
        return 2;
    }
    replacement = mmap(hidden_page, page_size * 2, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (replacement != hidden_page)
    {
        return 3;
    }
    if (munmap(replacement, page_size * 2) != 0)
    {
        return 4;
    }
    return 0;
}

static int test_tls_metadata_after_fork(size_t page_size)
{
    const char *test = "tls-metadata-after-fork";
    void *mapping;
    void *hidden_page;
    pid_t child;
    int status;

    mapping = mmap(NULL, TLS_WORKAROUND_LENGTH_1, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(test, mapping != MAP_FAILED, "TLS-workaround mmap failed");
    hidden_page = (void *)((char *)mapping - page_size);

    child = fork();
    CHECK(test, child >= 0, "fork failed");
    if (child == 0)
    {
        _Exit(tls_metadata_child_check(mapping, hidden_page, page_size));
    }

    /* The RT-Smart musl port exposes process reaping via _NRSYS_waitpid. */
    CHECK(test, lwp_waitpid_syscall(child, &status, 0) == child,
          "RT-Smart waitpid syscall failed");
    CHECK(test, WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "child did not preserve TLS metadata or PMUTEX state");
    CHECK(test, munmap(mapping, page_size) == 0,
          "parent TLS-workaround munmap failed after fork");

    printf("[PASS] %s\n", test);
    return 0;
}

static int test_shared_memory_partial_munmap(size_t page_size)
{
    const char *test = "shared-memory-partial-munmap";
    key_t key = (key_t)(0x4c575000U ^ (unsigned int)getpid());
    volatile unsigned char *mapping;
    int id;

    /* The RT-Smart musl port exposes System V shared memory via _NRSYS_*. */
    id = (int)lwp_shmget_syscall(key, page_size * 2, IPC_CREAT | 0600);
    CHECK(test, id >= 0, "shmget failed");
    mapping = lwp_shmat_syscall(id, NULL);
    CHECK(test, mapping != NULL && mapping != (void *)-1, "shmat failed");

    mapping[0] = 0x13;
    mapping[page_size] = 0x31;
    errno = 0;
    CHECK(test, has_expected_error(munmap((void *)(mapping + page_size), page_size), EINVAL),
          "partial shared-memory munmap did not fail with EINVAL");
    CHECK(test, mapping[0] == 0x13 && mapping[page_size] == 0x31,
          "failed partial shared-memory munmap altered the mapping");
    CHECK(test, lwp_shmdt_syscall((void *)mapping) == 0, "shmdt failed");
    CHECK(test, lwp_shmrm_syscall(id) == 0, "shared-memory removal failed");

    printf("[PASS] %s\n", test);
    return 0;
}

static int test_user_pointer_validation(void)
{
    const char *test = "user-pointer-validation";
    pthread_mutex_t mutex;
    void *invalid = (void *)(uintptr_t)UINTPTR_MAX;
    int pipefd[2];
    char byte = 'L';
    char received = 0;

    CHECK(test, pipe(pipefd) == 0, "pipe creation failed");
    CHECK(test, write(pipefd[1], &byte, 1) == 1, "pipe write failed");
    errno = 0;
    CHECK(test, has_expected_error(read(pipefd[0], invalid, 1), EFAULT),
          "invalid read destination did not fail with EFAULT");
    CHECK(test, read(pipefd[0], &received, 1) == 1 && received == byte,
          "invalid read consumed data or damaged the pipe");
    errno = 0;
    CHECK(test, has_expected_error(write(pipefd[1], invalid, 1), EFAULT),
          "invalid write source did not fail with EFAULT");
    CHECK(test, close(pipefd[0]) == 0 && close(pipefd[1]) == 0,
          "pipe close failed");

    CHECK(test, pmutex_syscall(NULL, PMUTEX_INIT, NULL) == EINVAL,
          "NULL PMUTEX pointer did not return positive EINVAL");
    CHECK(test, pmutex_syscall(invalid, PMUTEX_INIT, NULL) == EINVAL,
          "out-of-range PMUTEX pointer did not return positive EINVAL");

    memset(&mutex, 0, sizeof(mutex));
    CHECK(test, pthread_mutex_init(&mutex, NULL) == 0, "pthread_mutex_init failed");
    CHECK(test, pmutex_syscall(&mutex, PMUTEX_LOCK, invalid) == EINVAL,
          "invalid PMUTEX timeout pointer did not return positive EINVAL");
    CHECK(test, pthread_mutex_lock(&mutex) == 0, "mutex lock after invalid timeout failed");
    CHECK(test, pthread_mutex_unlock(&mutex) == 0, "mutex unlock after invalid timeout failed");
    CHECK(test, pthread_mutex_destroy(&mutex) == 0, "mutex destroy failed");

    printf("[PASS] %s\n", test);
    return 0;
}

static int test_pmutex_futex_tree_separation(void)
{
    const char *test = "pmutex-futex-tree-separation";
    pthread_mutex_t mutex;

    memset(&mutex, 0, sizeof(mutex));
    CHECK(test, pthread_mutex_init(&mutex, NULL) == 0, "pthread_mutex_init failed");

    /* A futex at the same user address must not find a PMUTEX AVL node. */
    CHECK(test, futex_wake_syscall((int *)(void *)&mutex) == 0,
          "FUTEX_WAKE at a PMUTEX address failed");
    CHECK(test, pthread_mutex_lock(&mutex) == 0,
          "pthread_mutex_lock failed after FUTEX_WAKE collision probe");
    CHECK(test, pmutex_syscall(&mutex, PMUTEX_DESTROY, NULL) == EBUSY,
          "raw PMUTEX destroy did not return positive EBUSY");
    CHECK(test, pthread_mutex_destroy(&mutex) == EBUSY,
          "pthread_mutex_destroy did not return EBUSY");
    CHECK(test, pthread_mutex_unlock(&mutex) == 0,
          "pthread_mutex_unlock failed after collision probe");
    CHECK(test, pthread_mutex_destroy(&mutex) == 0,
          "pthread_mutex_destroy failed after unlock");

    printf("[PASS] %s\n", test);
    return 0;
}

static void *mutex_worker(void *parameter)
{
    struct mutex_stress *stress = (struct mutex_stress *)parameter;
    int i;

    for (i = 0; i < MUTEX_ITERATIONS; i++)
    {
        int ret = pthread_mutex_lock(&stress->mutex);

        if (ret != 0)
        {
            return (void *)(intptr_t)ret;
        }
        stress->counter++;
        ret = pthread_mutex_unlock(&stress->mutex);
        if (ret != 0)
        {
            return (void *)(intptr_t)ret;
        }
    }

    return NULL;
}

static void *mutex_waiter_worker(void *parameter)
{
    struct mutex_waiter *waiter = (struct mutex_waiter *)parameter;

    __sync_lock_test_and_set(&waiter->entered, 1);
    waiter->result = pthread_mutex_lock(waiter->mutex);
    if (waiter->result == 0)
    {
        waiter->result = pthread_mutex_unlock(waiter->mutex);
    }
    __sync_lock_test_and_set(&waiter->completed, 1);
    return (void *)(intptr_t)waiter->result;
}

static int test_pmutex_lifetime(size_t page_size)
{
    const char *test = "pmutex-lifetime";
    struct mutex_stress *stress;
    struct mutex_waiter waiter;
    pthread_t workers[MUTEX_WORKERS];
    pthread_t waiter_thread;
    int created = 0;
    int i;
    int ret;

    CHECK(test, sizeof(*stress) <= page_size, "stress state does not fit in one page");
    stress = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(test, stress != MAP_FAILED, "mmap-backed mutex allocation failed");
    memset(stress, 0, sizeof(*stress));

    ret = pthread_mutex_init(&stress->mutex, NULL);
    CHECK(test, ret == 0, "pthread_mutex_init failed");

    for (i = 0; i < MUTEX_WORKERS; i++)
    {
        ret = pthread_create(&workers[i], NULL, mutex_worker, stress);
        if (ret != 0)
        {
            printf("[FAIL] %s: pthread_create returned %d\n", test, ret);
            return -1;
        }
        created++;
    }

    for (i = 0; i < created; i++)
    {
        void *worker_result = NULL;

        ret = pthread_join(workers[i], &worker_result);
        if (ret != 0 || worker_result != NULL)
        {
            printf("[FAIL] %s: worker %d failed (join=%d worker=%ld)\n",
                   test, i, ret, (long)(intptr_t)worker_result);
            return -1;
        }
    }

    CHECK(test, stress->counter == MUTEX_WORKERS * MUTEX_ITERATIONS,
          "mutex workers lost increments");
    CHECK(test, pthread_mutex_lock(&stress->mutex) == 0,
          "pthread_mutex_lock before destroy race failed");

    memset(&waiter, 0, sizeof(waiter));
    waiter.mutex = &stress->mutex;
    CHECK(test, pthread_create(&waiter_thread, NULL, mutex_waiter_worker, &waiter) == 0,
          "pthread_create for blocked mutex waiter failed");
    CHECK(test, wait_for_count(&waiter.entered, 1) == 0,
          "mutex waiter did not start");
    sleep_milliseconds(10);

    CHECK(test, pmutex_syscall(&stress->mutex, PMUTEX_DESTROY, NULL) == EBUSY,
          "raw PMUTEX destroy did not reject an in-use mutex");
    CHECK(test, pthread_mutex_destroy(&stress->mutex) == EBUSY,
          "pthread_mutex_destroy did not return EBUSY for an in-use mutex");
    CHECK(test, pthread_mutex_unlock(&stress->mutex) == 0,
          "pthread_mutex_unlock failed after destroy rejection");

    {
        void *waiter_result = NULL;

        CHECK(test, pthread_join(waiter_thread, &waiter_result) == 0,
              "pthread_join for mutex waiter failed");
        CHECK(test, waiter.completed != 0 && waiter.result == 0 && waiter_result == NULL,
              "mutex waiter did not complete cleanly after destroy rejection");
    }

    CHECK(test, pthread_mutex_destroy(&stress->mutex) == 0,
          "pthread_mutex_destroy failed after all users left");
    CHECK(test, munmap(stress, page_size) == 0,
          "mmap-backed mutex storage munmap failed");

    printf("[PASS] %s\n", test);
    return 0;
}

static void *lifecycle_worker(void *parameter)
{
    struct lifecycle_state *state = (struct lifecycle_state *)parameter;

    __sync_fetch_and_add(&state->completed, 1);
    return state;
}

static int test_pthread_reaper_lifecycles(void)
{
    const char *test = "pthread-reaper-lifecycles";
    struct lifecycle_state joinable = { 0 };
    struct lifecycle_state custom_stack = { 0 };
    struct lifecycle_state detached = { 0 };
    pthread_t thread;
    pthread_t detached_threads[DETACHED_WORKERS];
    pthread_attr_t attr;
    void *stack;
    void *result = NULL;
    int i;

    CHECK(test, pthread_create(&thread, NULL, lifecycle_worker, &joinable) == 0,
          "joinable pthread_create failed");
    CHECK(test, wait_for_count(&joinable.completed, 1) == 0,
          "joinable thread did not complete");
    sleep_milliseconds(20);
    CHECK(test, pthread_join(thread, &result) == 0 && result == &joinable,
          "joinable thread failed after reaper cleanup");

    stack = mmap(NULL, CUSTOM_STACK_SIZE, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(test, stack != MAP_FAILED, "custom pthread stack mmap failed");
    CHECK(test, pthread_attr_init(&attr) == 0, "pthread_attr_init failed");
    CHECK(test, pthread_attr_setstack(&attr, stack, CUSTOM_STACK_SIZE) == 0,
          "pthread_attr_setstack failed");
    CHECK(test, pthread_create(&thread, &attr, lifecycle_worker, &custom_stack) == 0,
          "custom-stack pthread_create failed");
    CHECK(test, wait_for_count(&custom_stack.completed, 1) == 0,
          "custom-stack thread did not complete");
    sleep_milliseconds(20);
    result = NULL;
    CHECK(test, pthread_join(thread, &result) == 0 && result == &custom_stack,
          "custom-stack thread failed after reaper cleanup");
    ((volatile unsigned char *)stack)[0] = 0x6c;
    ((volatile unsigned char *)stack)[CUSTOM_STACK_SIZE - 1] = 0xc6;
    CHECK(test, pthread_attr_destroy(&attr) == 0, "pthread_attr_destroy failed");
    CHECK(test, munmap(stack, CUSTOM_STACK_SIZE) == 0,
          "caller-owned pthread stack was not safe to unmap");

    CHECK(test, pthread_attr_init(&attr) == 0, "detached pthread_attr_init failed");
    CHECK(test, pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED) == 0,
          "pthread_attr_setdetachstate failed");
    for (i = 0; i < DETACHED_WORKERS; i++)
    {
        CHECK(test, pthread_create(&detached_threads[i], &attr, lifecycle_worker, &detached) == 0,
              "detached pthread_create failed");
    }
    CHECK(test, wait_for_count(&detached.completed, DETACHED_WORKERS) == 0,
          "detached threads did not complete");
    sleep_milliseconds(50);
    CHECK(test, pthread_attr_destroy(&attr) == 0,
          "detached pthread_attr_destroy failed");

    printf("[PASS] %s\n", test);
    return 0;
}

static void *mmap_stress_worker(void *parameter)
{
    struct mmap_stress_state *state = (struct mmap_stress_state *)parameter;
    int i;

    for (i = 0; i < MM_STRESS_ITERATIONS; i++)
    {
        volatile unsigned char *mapping;

        mapping = mmap(NULL, state->page_size * 3, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED)
        {
            __sync_lock_test_and_set(&state->failure, 1);
            return (void *)(intptr_t)1;
        }
        mapping[0] = (unsigned char)i;
        mapping[state->page_size * 2] = (unsigned char)(i ^ 0xff);
        if (munmap((void *)(mapping + state->page_size), state->page_size) != 0 ||
            munmap((void *)mapping, state->page_size * 3) != 0)
        {
            __sync_lock_test_and_set(&state->failure, 1);
            return (void *)(intptr_t)1;
        }
    }
    return NULL;
}

static int test_mm_lock_concurrent_mmap(size_t page_size)
{
    const char *test = "mm-lock-concurrent-mmap";
    struct mmap_stress_state state;
    pthread_t workers[MM_STRESS_WORKERS];
    int i;

    memset(&state, 0, sizeof(state));
    state.page_size = page_size;
    for (i = 0; i < MM_STRESS_WORKERS; i++)
    {
        CHECK(test, pthread_create(&workers[i], NULL, mmap_stress_worker, &state) == 0,
              "mmap stress pthread_create failed");
    }
    for (i = 0; i < MM_STRESS_WORKERS; i++)
    {
        void *result = NULL;

        CHECK(test, pthread_join(workers[i], &result) == 0 && result == NULL,
              "mmap stress worker failed");
    }
    CHECK(test, state.failure == 0, "concurrent mmap/munmap reported a failure");

    /* Reaching this test also proves the board-time mm_lock initialization completed. */
    printf("[PASS] %s\n", test);
    return 0;
}

int main(void)
{
    long page_size = sysconf(_SC_PAGESIZE);

    if (page_size <= 0)
    {
        printf("[FAIL] cannot determine page size\n");
        return EXIT_FAILURE;
    }

    if (test_heap_info_current_usage((size_t)page_size) != 0 ||
        test_split_and_multi_area_munmap((size_t)page_size) != 0 ||
        test_partial_boundary_munmap((size_t)page_size) != 0 ||
        test_map_area_avl_churn((size_t)page_size) != 0 ||
        test_munmap_validation((size_t)page_size) != 0 ||
        test_tls_munmap_metadata((size_t)page_size) != 0 ||
        test_tls_metadata_after_fork((size_t)page_size) != 0 ||
        test_shared_memory_partial_munmap((size_t)page_size) != 0 ||
        test_user_pointer_validation() != 0 ||
        test_pmutex_futex_tree_separation() != 0 ||
        test_pmutex_lifetime((size_t)page_size) != 0 ||
        test_pthread_reaper_lifecycles() != 0 ||
        test_mm_lock_concurrent_mmap((size_t)page_size) != 0)
    {
        return EXIT_FAILURE;
    }

    printf("LWP regression test passed\n");
    return EXIT_SUCCESS;
}
