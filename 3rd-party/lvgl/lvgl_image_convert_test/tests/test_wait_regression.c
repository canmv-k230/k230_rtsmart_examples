/* Exercise the real regression control flow without touching VO or WBC. */
#define _GNU_SOURCE
#include <signal.h>
static int fixture_sigaction(int number, const struct sigaction *action, struct sigaction *previous);
#define sigaction(...) fixture_sigaction(__VA_ARGS__)
#define kd_mpi_vo_wait_frame_release mock_wait_frame_release
#define kd_mpi_vo_set_wbc_attr mock_set_wbc_attr
#define kd_mpi_vo_enable_wbc mock_enable_wbc
#define kd_mpi_vo_disable_wbc mock_disable_wbc
#define kd_mpi_wbc_dump_frame mock_dump_frame
#define kd_mpi_wbc_dump_release mock_dump_release
#include "../lvgl_display_regression.c"
#undef sigaction

#include <assert.h>
#include <semaphore.h>

static sem_t control;
static sem_t held_frame;
static unsigned dump_calls;
static unsigned registered_waiters;
static unsigned blocker_delay_us;
static bool broken_cleanup;
static __thread volatile sig_atomic_t interrupted;

static void fixture_interrupt(int number)
{
    (void)number;
    interrupted = 1;
}

static int fixture_sigaction(int number, const struct sigaction *action, struct sigaction *previous)
{
    struct sigaction copy = *action;
    if (copy.sa_handler == interrupt_wait) copy.sa_handler = fixture_interrupt;
    return sigaction(number, &copy, previous);
}

static int take(sem_t *sem, int timeout_ms)
{
    uint64_t deadline = time_us() + (uint64_t)(unsigned)timeout_ms * 1000u;
    for (;;) {
        int ret = sem_trywait(sem);
        if (ret == 0) return 0;
        assert(errno == EAGAIN);
        if (timeout_ms == 0) { errno = EBUSY; return -1; }
        if (interrupted) { errno = EINTR; return -1; }
        if (timeout_ms > 0 && time_us() >= deadline) { errno = ETIMEDOUT; return -1; }
        usleep(1000);
    }
}

k_s32 mock_wait_frame_release(k_vo_layer_id layer, const k_video_frame_info *frame,
                              k_s32 timeout_ms)
{
    assert(layer == TEST_LAYER);
    interrupted = 0;
    if (timeout_ms == INT32_MAX || timeout_ms < -1) {
        errno = EINVAL;
        return -1;
    }
    if (take(&control, timeout_ms)) return -1;
    bool held = frame->v_frame.phys_addr[0] == 1;
    if (held) {
        int policy;
        struct sched_param priority;
        assert(pthread_getschedparam(pthread_self(), &policy, &priority) == 0);
        assert(priority.sched_priority == 10);
        __atomic_fetch_add(&registered_waiters, 1, __ATOMIC_RELAXED);
    }
    assert(sem_post(&control) == 0);
    if (!held) return 0;

    int ret = take(&held_frame, timeout_ms);
    int error = errno;
    if (broken_cleanup) {
        while (sem_wait(&control) != 0) assert(errno == EINTR);
        assert(sem_post(&control) == 0);
    }
    __atomic_fetch_sub(&registered_waiters, 1, __ATOMIC_RELAXED);
    errno = error;
    return ret;
}

k_s32 mock_set_wbc_attr(k_vo_wbc_attr *attr)
{
    assert(attr->blk_cnt == 2);
    dump_calls = 0;
    return 0;
}

k_s32 mock_enable_wbc(void) { return 0; }
k_s32 mock_disable_wbc(void) { return 0; }
k_s32 mock_dump_release(const k_video_frame_info *frame) { (void)frame; return 0; }

k_s32 mock_dump_frame(k_video_frame_info *frame, k_u32 timeout_ms)
{
    (void)frame;
    if (++dump_calls == 1) return 0;
    /* A delayed blocker must not be mistaken for a contended control mutex. */
    usleep(blocker_delay_us);
    assert(sem_wait(&control) == 0);
    if (dump_calls == 3) assert(__atomic_load_n(&registered_waiters, __ATOMIC_RELAXED) == 2);
    usleep(timeout_ms * 1000u);
    assert(sem_post(&control) == 0);
    return -1;
}

int main(void)
{
    alarm(15);
    /* Keep this isolated test ahead of unrelated applications on the board. */
    struct sched_param priority = {.sched_priority = 15};
    assert(pthread_setschedparam(pthread_self(), SCHED_OTHER, &priority) == 0);
    assert(sem_init(&control, 0, 1) == 0);
    assert(sem_init(&held_frame, 0, 0) == 0);
    k_video_frame_info held = {0}, free_frame = {0};
    held.v_frame.phys_addr[0] = 1;
    assert(test_release_timeout_limits(&held, &free_frame));
    assert(test_release_contention(&held, &free_frame));
    blocker_delay_us = 50000;
    assert(test_release_contention(&held, &free_frame));
    broken_cleanup = true;
    assert(!test_release_contention(&held, &free_frame));
    assert(__atomic_load_n(&registered_waiters, __ATOMIC_RELAXED) == 0);
    assert(sem_destroy(&held_frame) == 0);
    assert(sem_destroy(&control) == 0);
    puts("VO regression fixture: PASS (normal, delayed blocker, broken cleanup detected)");
    return 0;
}
