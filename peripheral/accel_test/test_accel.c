#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#define GNNE_DEVICE "/dev/gnne_device"
#define AI2D_DEVICE "/dev/ai_2d_device"

#define GNNE_CMD_LOCK    0
#define GNNE_CMD_TRYLOCK 1
#define GNNE_CMD_UNLOCK  2

static unsigned int passed;
static unsigned int failed;

static void check_result(int condition, const char *name)
{
    if (condition) {
        passed++;
        printf("[PASS] %s\n", name);
    } else {
        failed++;
        printf("[FAIL] %s (errno=%d: %s)\n", name, errno,
               strerror(errno));
    }
}

static uint64_t monotonic_ms(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0;
    return (uint64_t)now.tv_sec * 1000U + (uint64_t)now.tv_nsec / 1000000U;
}

static void test_open_close(const char *path, const char *name)
{
    unsigned int cycle;
    int fd;

    for (cycle = 0; cycle < 100; cycle++) {
        fd = open(path, O_RDWR);
        if (fd < 0 || close(fd) != 0)
            break;
    }
    check_result(cycle == 100, name);
}

static void test_poll_timeout(const char *path, const char *name)
{
    struct pollfd pfd;
    uint64_t start_ms;
    uint64_t elapsed_ms;
    int fd;
    int ret;

    fd = open(path, O_RDWR);
    check_result(fd >= 0, name);
    if (fd < 0)
        return;

    pfd.fd = fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    start_ms = monotonic_ms();
    ret = poll(&pfd, 1, 25);
    elapsed_ms = monotonic_ms() - start_ms;
    check_result(ret == 0 && elapsed_ms < 500,
                 strcmp(path, GNNE_DEVICE) == 0 ?
                     "GNNE poll timeout is bounded" :
                     "AI2D poll timeout is bounded");
    close(fd);
}

static void test_resource_exhaustion(const char *path, const char *name)
{
    struct rlimit limit;
    size_t capacity = 64;
    size_t count = 0;
    int *fds;
    int recovery_fd;

    if (getrlimit(RLIMIT_NOFILE, &limit) == 0 && limit.rlim_cur <= 256)
        capacity = (size_t)limit.rlim_cur;
    if (capacity < 8)
        capacity = 8;

    fds = calloc(capacity, sizeof(*fds));
    check_result(fds != NULL, name);
    if (fds == NULL)
        return;

    while (count < capacity) {
        fds[count] = open(path, O_RDWR);
        if (fds[count] < 0)
            break;
        count++;
    }

    check_result(count > 0 && count < capacity,
                 strcmp(path, GNNE_DEVICE) == 0 ?
                     "GNNE descriptor exhaustion is bounded" :
                     "AI2D descriptor exhaustion is bounded");
    while (count > 0)
        close(fds[--count]);
    free(fds);

    recovery_fd = open(path, O_RDWR);
    check_result(recovery_fd >= 0,
                 strcmp(path, GNNE_DEVICE) == 0 ?
                     "GNNE recovers after exhaustion" :
                     "AI2D recovers after exhaustion");
    if (recovery_fd >= 0)
        close(recovery_fd);
}

static void test_gnne_locking(void)
{
    int owner = open(GNNE_DEVICE, O_RDWR);
    int waiter = open(GNNE_DEVICE, O_RDWR);

    check_result(owner >= 0 && waiter >= 0, "GNNE concurrent open");
    if (owner < 0 || waiter < 0)
        goto cleanup;

    check_result(ioctl(owner, GNNE_CMD_LOCK, NULL) == 0,
                 "GNNE owner lock");
    check_result(ioctl(waiter, GNNE_CMD_TRYLOCK, NULL) < 0,
                 "GNNE competing trylock rejected");
    check_result(ioctl(owner, 0x7fffffff, NULL) < 0,
                 "GNNE invalid ioctl rejected");

    close(owner);
    owner = -1;
    check_result(ioctl(waiter, GNNE_CMD_TRYLOCK, NULL) == 0,
                 "GNNE close releases owner lock");
    check_result(ioctl(waiter, GNNE_CMD_UNLOCK, NULL) == 0,
                 "GNNE unlock");

cleanup:
    if (owner >= 0)
        close(owner);
    if (waiter >= 0)
        close(waiter);
}

static void test_ai2d_invalid_ioctl(void)
{
    int fd = open(AI2D_DEVICE, O_RDWR);

    check_result(fd >= 0, "AI2D open for invalid ioctl");
    if (fd < 0)
        return;
    check_result(ioctl(fd, 0x7fffffff, NULL) < 0,
                 "AI2D invalid ioctl rejected");
    close(fd);
}

static int wait_for_interrupt(const char *device, unsigned int timeout_ms)
{
    struct pollfd pfd;
    int fd;
    int ret;

    fd = open(device, O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    printf("ACCEL_WAIT_READY device=%s timeout_ms=%u\n", device,
           timeout_ms);
    pfd.fd = fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    ret = poll(&pfd, 1, (int)timeout_ms);
    printf("ACCEL_WAIT_DONE ret=%d errno=%d revents=0x%x\n", ret, errno,
           pfd.revents);
    close(fd);
    return ret < 0 && errno != EINTR;
}

static int parse_wait_args(int argc, char **argv)
{
    const char *device;
    char *end;
    unsigned long timeout_ms;

    if (argc != 4 || strcmp(argv[1], "--wait") != 0)
        return -1;
    if (strcmp(argv[2], "gnne") == 0)
        device = GNNE_DEVICE;
    else if (strcmp(argv[2], "ai2d") == 0)
        device = AI2D_DEVICE;
    else
        return -1;

    timeout_ms = strtoul(argv[3], &end, 10);
    if (end == argv[3] || *end != '\0' || timeout_ms == 0 ||
        timeout_ms > 60000)
        return -1;
    return wait_for_interrupt(device, (unsigned int)timeout_ms);
}

int main(int argc, char **argv)
{
    int wait_result;

    if (argc > 1) {
        wait_result = parse_wait_args(argc, argv);
        if (wait_result >= 0)
            return wait_result;
        fprintf(stderr, "Usage: %s [--wait <gnne|ai2d> <timeout_ms>]\n",
                argv[0]);
        return 1;
    }

    test_open_close(GNNE_DEVICE, "GNNE 100 open/close cycles");
    test_open_close(AI2D_DEVICE, "AI2D 100 open/close cycles");
    test_poll_timeout(GNNE_DEVICE, "GNNE open for poll");
    test_poll_timeout(AI2D_DEVICE, "AI2D open for poll");
    test_gnne_locking();
    test_ai2d_invalid_ioctl();
    test_resource_exhaustion(GNNE_DEVICE, "GNNE exhaustion allocation");
    test_resource_exhaustion(AI2D_DEVICE, "AI2D exhaustion allocation");

    printf("ACCEL_TEST_RESULT passed=%u failed=%u total=%u\n", passed,
           failed, passed + failed);
    return failed ? 1 : 0;
}
