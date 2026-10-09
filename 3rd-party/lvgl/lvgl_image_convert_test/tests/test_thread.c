/* Run on the host: cc -std=gnu99 -Wall -Wextra -Werror -pthread
 * tests/test_thread.c -o /tmp/lvgl-test-thread && /tmp/lvgl-test-thread
 */
#define _GNU_SOURCE
#ifndef __riscv
#define LVGL_TEST_THREAD_TIMEOUT_MS 200
#endif
#include "../lvgl_test_thread.h"

#include <assert.h>
#include <semaphore.h>
#include <string.h>
#include <sys/wait.h>

static int signals;
static bool allow_wait;
static sem_t gate;
static int wait_result;
static int wait_error;

static void on_signal(int number)
{
    (void)number;
    __atomic_fetch_add(&signals, 1, __ATOMIC_RELAXED);
}

static void *delayed_wait(void *arg)
{
    (void)arg;
    while (!__atomic_load_n(&allow_wait, __ATOMIC_ACQUIRE)) sched_yield();
    wait_result = sem_wait(&gate);
    wait_error = errno;
    return NULL;
}

static void *blocked_forever(void *arg)
{
    (void)arg;
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGUSR1);
    assert(pthread_sigmask(SIG_BLOCK, &set, NULL) == 0);
    for (;;) pause();
    return NULL;
}

static void check_timeout(bool interrupt)
{
    uint64_t start = time_us();
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        test_thread_t thread;
        assert(test_thread_start(&thread, NULL, blocked_forever, NULL));
        test_thread_join(&thread, interrupt);
        _Exit(99);
    }
    int status;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE);
    assert(time_us() - start >= LVGL_TEST_THREAD_TIMEOUT_MS * 1000u);
}

int main(int argc, char **argv)
{
    alarm(10);
    struct sigaction action = {0};
    action.sa_handler = on_signal;
    sigemptyset(&action.sa_mask);
    assert(sigaction(SIGUSR1, &action, NULL) == 0);
    if (argc == 2 && (strcmp(argv[1], "--timeout") == 0 ||
                      strcmp(argv[1], "--timeout-interrupt") == 0)) {
        test_thread_t thread;
        assert(test_thread_start(&thread, NULL, blocked_forever, NULL));
        test_thread_join(&thread, strcmp(argv[1], "--timeout-interrupt") == 0);
        return 99;
    }
    assert(sem_init(&gate, 0, 0) == 0);

    test_thread_t thread;
    assert(test_thread_start(&thread, NULL, delayed_wait, NULL));
    assert(!test_thread_done(&thread));
    assert(pthread_kill(thread.id, SIGUSR1) == 0);
    while (__atomic_load_n(&signals, __ATOMIC_RELAXED) == 0) sched_yield();
    /* The first interrupt is deliberately consumed before entering sem_wait. */
    __atomic_store_n(&allow_wait, true, __ATOMIC_RELEASE);
    test_thread_join(&thread, true);
    assert(wait_result == -1 && wait_error == EINTR);
    assert(__atomic_load_n(&signals, __ATOMIC_RELAXED) >= 2);
    assert(test_thread_done(&thread));
    puts("lost interrupt before syscall: PASS");

    assert(test_thread_start(&thread, NULL, delayed_wait, NULL));
    assert(sem_post(&gate) == 0);
    test_thread_join(&thread, false);
    assert(wait_result == 0);
    assert(sem_destroy(&gate) == 0);
    puts("normal completion and worker reuse: PASS");
    fflush(NULL);

    if (argc == 2 && strcmp(argv[1], "--worker-check") == 0) return 0;
    check_timeout(false);
    check_timeout(true);
    puts("bounded joins with and without interrupts: PASS");
    return 0;
}
