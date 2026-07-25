/* RT-Smart musl pthread/futex regression tests. */

#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "canmv_misc.h"
#include "hal_syscall.h"

#define FUTEX_WAIT          0
#define FUTEX_WAKE          1
#define FUTEX_PRIVATE_FLAG  128

#define FUTEX_WAKE_CHURN_COUNT 256
#define CONDITION_WAITERS      4
#define CONDITION_TIMEOUT_MS   1000

struct condition_test_state
{
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    unsigned int ready;
    unsigned int completed;
    int release;
    int worker_error;
};

static int unwaited_futexes[FUTEX_WAKE_CHURN_COUNT];

static int pthread_futex_fail(const char *test, int line, const char *message)
{
    printf("[FAIL] %s:%d: %s (errno=%d)\n", test, line, message, errno);
    return -1;
}

#define FUTEX_CHECK(test, condition, message) \
    do \
    { \
        if (!(condition)) \
        { \
            return pthread_futex_fail((test), __LINE__, (message)); \
        } \
    } while (0)

static int futex_has_error(long result, int expected)
{
    return result == -expected || (result == -1 && errno == expected);
}

static long raw_futex(int *address, int operation, int value,
                      const struct timespec *timeout)
{
    return syscall(_NRSYS_futex, (long)address, operation, value,
                   (long)timeout, 0, 0);
}

static int64_t elapsed_nanoseconds(const struct timespec *start,
                                   const struct timespec *end)
{
    return (int64_t)(end->tv_sec - start->tv_sec) * 1000000000LL +
           end->tv_nsec - start->tv_nsec;
}

static int test_private_futex_wait_errors(void)
{
    const char *test = "private-futex-wait-errors";
    struct timespec timeout = { 0, 1 };
    struct timespec invalid_timeout = { 0, 1000000000L };
    struct timespec start;
    struct timespec end;
    int futex_value = 1;
    long result;

    errno = 0;
    result = raw_futex(&futex_value, FUTEX_WAIT | FUTEX_PRIVATE_FLAG, 0, NULL);
    FUTEX_CHECK(test, futex_has_error(result, EAGAIN),
                "private FUTEX_WAIT value mismatch did not return EAGAIN");

    errno = 0;
    result = raw_futex(&futex_value, FUTEX_WAIT | FUTEX_PRIVATE_FLAG,
                       futex_value, &invalid_timeout);
    FUTEX_CHECK(test, futex_has_error(result, EINVAL),
                "invalid private FUTEX_WAIT timeout did not return EINVAL");

    FUTEX_CHECK(test, clock_gettime(CLOCK_MONOTONIC, &start) == 0,
                "clock_gettime before futex wait failed");
    errno = 0;
    result = raw_futex(&futex_value, FUTEX_WAIT | FUTEX_PRIVATE_FLAG,
                       futex_value, &timeout);
    FUTEX_CHECK(test, clock_gettime(CLOCK_MONOTONIC, &end) == 0,
                "clock_gettime after futex wait failed");
    FUTEX_CHECK(test, futex_has_error(result, ETIMEDOUT),
                "private FUTEX_WAIT did not return ETIMEDOUT");
    FUTEX_CHECK(test, elapsed_nanoseconds(&start, &end) >= 100000,
                "sub-tick futex timeout expired immediately");

    printf("[PASS] %s\n", test);
    return 0;
}

static int test_unwaited_futex_wake_does_not_allocate(void)
{
    const char *test = "unwaited-futex-wake-no-allocation";
    struct canmv_misc_dev_meminfo_t warmup;
    struct canmv_misc_dev_meminfo_t before;
    struct canmv_misc_dev_meminfo_t after;
    int i;

    FUTEX_CHECK(test, canmv_misc_get_sys_heap_size(&warmup) == 0,
                "heap-info warmup failed");
    FUTEX_CHECK(test, canmv_misc_get_sys_heap_size(&before) == 0,
                "heap-info query before FUTEX_WAKE churn failed");

    for (i = 0; i < FUTEX_WAKE_CHURN_COUNT; i++)
    {
        FUTEX_CHECK(test, raw_futex(&unwaited_futexes[i], FUTEX_WAKE, 1, NULL) == 0,
                    "FUTEX_WAKE without a waiter failed");
        FUTEX_CHECK(test,
                    raw_futex(&unwaited_futexes[i],
                              FUTEX_WAKE | FUTEX_PRIVATE_FLAG, 1, NULL) == 0,
                    "private FUTEX_WAKE without a waiter failed");
    }

    FUTEX_CHECK(test, canmv_misc_get_sys_heap_size(&after) == 0,
                "heap-info query after FUTEX_WAKE churn failed");
    FUTEX_CHECK(test, after.used_size <= before.used_size,
                "unwaited FUTEX_WAKE calls leaked kernel heap objects");

    printf("[PASS] %s\n", test);
    return 0;
}

static void *condition_waiter(void *parameter)
{
    struct condition_test_state *state = parameter;
    int result;

    result = pthread_mutex_lock(&state->mutex);
    if (result != 0)
    {
        state->worker_error = result;
        return (void *)(intptr_t)result;
    }

    state->ready++;
    while (!state->release)
    {
        result = pthread_cond_wait(&state->condition, &state->mutex);
        if (result != 0)
        {
            state->worker_error = result;
            pthread_mutex_unlock(&state->mutex);
            return (void *)(intptr_t)result;
        }
    }
    state->completed++;

    result = pthread_mutex_unlock(&state->mutex);
    if (result != 0)
    {
        state->worker_error = result;
        return (void *)(intptr_t)result;
    }
    return NULL;
}

static int wait_for_condition_waiters(struct condition_test_state *state)
{
    struct timespec delay = { 0, 1000000L };
    int elapsed;

    for (elapsed = 0; elapsed < CONDITION_TIMEOUT_MS; elapsed++)
    {
        unsigned int ready;

        if (pthread_mutex_lock(&state->mutex) != 0)
        {
            return -1;
        }
        ready = state->ready;
        if (pthread_mutex_unlock(&state->mutex) != 0)
        {
            return -1;
        }
        if (ready == CONDITION_WAITERS)
        {
            return 0;
        }
        nanosleep(&delay, NULL);
    }
    return -1;
}

static int test_private_condition_broadcast(void)
{
    const char *test = "private-condition-broadcast";
    struct condition_test_state state;
    pthread_t threads[CONDITION_WAITERS];
    int i;

    memset(&state, 0, sizeof(state));
    FUTEX_CHECK(test, pthread_mutex_init(&state.mutex, NULL) == 0,
                "pthread_mutex_init failed");
    FUTEX_CHECK(test, pthread_cond_init(&state.condition, NULL) == 0,
                "pthread_cond_init failed");

    for (i = 0; i < CONDITION_WAITERS; i++)
    {
        FUTEX_CHECK(test,
                    pthread_create(&threads[i], NULL, condition_waiter, &state) == 0,
                    "pthread_create for condition waiter failed");
    }
    FUTEX_CHECK(test, wait_for_condition_waiters(&state) == 0,
                "condition waiters did not become ready");

    FUTEX_CHECK(test, pthread_mutex_lock(&state.mutex) == 0,
                "pthread_mutex_lock before broadcast failed");
    state.release = 1;
    FUTEX_CHECK(test, pthread_cond_broadcast(&state.condition) == 0,
                "pthread_cond_broadcast failed");
    FUTEX_CHECK(test, pthread_mutex_unlock(&state.mutex) == 0,
                "pthread_mutex_unlock after broadcast failed");

    for (i = 0; i < CONDITION_WAITERS; i++)
    {
        struct timespec deadline;
        void *thread_result = NULL;

        FUTEX_CHECK(test, clock_gettime(CLOCK_REALTIME, &deadline) == 0,
                    "clock_gettime before pthread join failed");
        deadline.tv_sec += 1;
        FUTEX_CHECK(test,
                    pthread_timedjoin_np(threads[i], &thread_result, &deadline) == 0 &&
                    thread_result == NULL,
                    "condition waiter timed out or did not exit cleanly");
    }
    FUTEX_CHECK(test, state.worker_error == 0,
                "a condition waiter reported an error");
    FUTEX_CHECK(test, state.completed == CONDITION_WAITERS,
                "pthread_cond_broadcast did not release every waiter");
    FUTEX_CHECK(test, pthread_cond_destroy(&state.condition) == 0,
                "pthread_cond_destroy failed");
    FUTEX_CHECK(test, pthread_mutex_destroy(&state.mutex) == 0,
                "pthread_mutex_destroy failed");

    printf("[PASS] %s\n", test);
    return 0;
}

static int test_scheduler_validation(void)
{
    const char *test = "scheduler-validation";
    struct sched_param parameter = { 0 };
    long tid;
    long result;

    tid = syscall(_NRSYS_gettid);
    FUTEX_CHECK(test, tid > 0, "gettid failed");

    errno = 0;
    result = syscall(_NRSYS_sched_setscheduler, tid, INT_MAX, &parameter);
    FUTEX_CHECK(test, futex_has_error(result, EINVAL),
                "invalid scheduling policy did not return EINVAL");

    parameter.sched_priority = INT_MAX;
    errno = 0;
    result = syscall(_NRSYS_sched_setscheduler, tid, SCHED_OTHER, &parameter);
    FUTEX_CHECK(test, futex_has_error(result, EINVAL),
                "out-of-range scheduling priority did not return EINVAL");

    parameter.sched_priority = 0;
    errno = 0;
    result = syscall(_NRSYS_sched_setscheduler, INT_MAX, SCHED_OTHER, &parameter);
    FUTEX_CHECK(test, futex_has_error(result, ESRCH),
                "unknown thread id did not return ESRCH");

    printf("[PASS] %s\n", test);
    return 0;
}

int main(void)
{
    if (test_private_futex_wait_errors() != 0 ||
        test_unwaited_futex_wake_does_not_allocate() != 0 ||
        test_private_condition_broadcast() != 0 ||
        test_scheduler_validation() != 0)
    {
        return -1;
    }

    printf("pthread/futex regression test passed\n");
    return 0;
}
