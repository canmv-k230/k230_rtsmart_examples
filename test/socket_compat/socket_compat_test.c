#define _GNU_SOURCE

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/uio.h>
#include <unistd.h>

#include "socket_test_protocol.h"

#define DEFAULT_PORT             5202
#define CONNECT_TIMEOUT_MS       3000
#define EVENT_TIMEOUT_MS         10000
#define SOCKET_IO_TIMEOUT_SEC    10
#define UDP_RETRY_TIMEOUT_MS     3000
#define UDP_RETRY_COUNT          3

static int passed;
static int failed;
static int skipped;
static struct sockaddr_in peer_address;

#define PASS(name) do { printf("PASS %-28s\n", name); passed++; } while (0)
#define FAIL(name) do { printf("FAIL %-28s errno=%d (%s)\n", \
        name, errno, strerror(errno)); failed++; } while (0)
#define SKIP(name, reason) do { printf("SKIP %-28s %s\n", name, reason); skipped++; } while (0)
#define CHECK(name, condition) do { if (condition) PASS(name); else FAIL(name); } while (0)

static int write_full(int fd, const void *buffer, size_t length)
{
    size_t done = 0;

    while (done < length)
    {
        ssize_t result = send(fd, (const char *)buffer + done, length - done,
                MSG_NOSIGNAL);
        if (result < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return -1;
        }
        done += (size_t)result;
    }
    return 0;
}

static int read_full(int fd, void *buffer, size_t length)
{
    size_t done = 0;

    while (done < length)
    {
        ssize_t result = recv(fd, (char *)buffer + done, length - done, 0);
        if (result <= 0)
        {
            if (result < 0 && errno == EINTR)
            {
                continue;
            }
            return -1;
        }
        done += (size_t)result;
    }
    return 0;
}

static int send_request(int fd, uint32_t operation, uint32_t length)
{
    struct socket_test_request request;

    request.magic = htonl(SOCKET_TEST_MAGIC);
    request.operation = htonl(operation);
    request.length = htonl(length);
    return write_full(fd, &request, sizeof(request));
}

static int wait_for_event(int fd, short events, int timeout_ms)
{
    struct pollfd pfd = { fd, events, 0 };
    int result;

    do
    {
        result = poll(&pfd, 1, timeout_ms);
    } while (result < 0 && errno == EINTR);

    if (result == 0)
    {
        errno = ETIMEDOUT;
        return -1;
    }
    if (result < 0)
    {
        return -1;
    }
    if (pfd.revents & events)
    {
        return 0;
    }

    errno = EIO;
    return -1;
}

static void set_socket_timeouts(int fd)
{
    struct timeval timeout = { SOCKET_IO_TIMEOUT_SEC, 0 };

    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

static int make_udp_socket(void)
{
    int fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

    if (fd >= 0)
    {
        set_socket_timeouts(fd);
    }
    return fd;
}

static int connect_tcp(void)
{
    int fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    int flags;
    int result;

    if (fd < 0)
    {
        return -1;
    }
    flags = fcntl(fd, F_GETFL);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
    {
        close(fd);
        return -1;
    }

    result = connect(fd, (const struct sockaddr *)&peer_address,
            sizeof(peer_address));
    if (result < 0 && (errno == EINPROGRESS || errno == EAGAIN))
    {
        struct pollfd pfd = { fd, POLLOUT, 0 };
        int socket_error = 0;
        socklen_t error_length = sizeof(socket_error);

        result = poll(&pfd, 1, CONNECT_TIMEOUT_MS);
        if (result == 1 && getsockopt(fd, SOL_SOCKET, SO_ERROR,
                &socket_error, &error_length) == 0 && socket_error == 0)
        {
            result = 0;
        }
        else
        {
            errno = result == 0 ? ETIMEDOUT :
                    (socket_error ? socket_error : errno);
            result = -1;
        }
    }
    if (result < 0 || fcntl(fd, F_SETFL, flags & ~O_NONBLOCK) < 0)
    {
        close(fd);
        return -1;
    }

    set_socket_timeouts(fd);
    return fd;
}

static void test_creation_and_errors(void)
{
    int fd;
    int type = 0;
    socklen_t length = sizeof(type);

    errno = 0;
    fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, IPPROTO_UDP);
    CHECK("socket SOCK_NONBLOCK", fd >= 0 && (fcntl(fd, F_GETFL) & O_NONBLOCK));
    if (fd >= 0)
    {
        CHECK("getsockopt SO_TYPE", getsockopt(fd, SOL_SOCKET, SO_TYPE,
                &type, &length) == 0 && type == SOCK_DGRAM);
        CHECK("socket close", close(fd) == 0);
        errno = 0;
        CHECK("closed fd", close(fd) < 0 && errno == EBADF);
    }

#if defined(__riscv)
    errno = 0;
    fd = socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP);
    CHECK("IPv6 policy", fd < 0 && errno == EAFNOSUPPORT);
    if (fd >= 0)
    {
        close(fd);
    }
#else
    SKIP("IPv6 policy", "host kernel policy differs from RT-Smart");
#endif

    errno = 0;
    CHECK("invalid fd", recv(-1, &type, sizeof(type), 0) < 0 && errno == EBADF);

    {
        int pair[2];
#if defined(__riscv)
        errno = 0;
        CHECK("socketpair policy", socketpair(AF_UNIX, SOCK_STREAM, 0, pair) < 0 &&
                errno == EAFNOSUPPORT);
#else
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0)
        {
            close(pair[0]);
            close(pair[1]);
        }
        SKIP("socketpair policy", "host kernel policy differs from RT-Smart");
#endif
    }
}

static void test_socket_options(void)
{
    int fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    int broadcast_fd;
    int one = 1;
    int value;
    socklen_t length;
    struct timeval timeout = { 0, 250000 };
    struct linger linger = { 1, 1 };

    if (fd < 0)
    {
        FAIL("socket options setup");
        return;
    }

#define TEST_INT_OPTION(label, level, option, input) do { \
    value = (input); \
    length = sizeof(value); \
    errno = 0; \
    CHECK(label, setsockopt(fd, level, option, &value, sizeof(value)) == 0 && \
            getsockopt(fd, level, option, &value, &length) == 0); \
} while (0)

    TEST_INT_OPTION("SO_REUSEADDR", SOL_SOCKET, SO_REUSEADDR, one);
    TEST_INT_OPTION("SO_KEEPALIVE", SOL_SOCKET, SO_KEEPALIVE, one);
    TEST_INT_OPTION("SO_SNDBUF", SOL_SOCKET, SO_SNDBUF, 4096);
    TEST_INT_OPTION("SO_RCVBUF", SOL_SOCKET, SO_RCVBUF, 8192);
    TEST_INT_OPTION("IP_TTL", IPPROTO_IP, IP_TTL, 32);
    TEST_INT_OPTION("IP_TOS", IPPROTO_IP, IP_TOS, 0x10);
    TEST_INT_OPTION("TCP_NODELAY", IPPROTO_TCP, TCP_NODELAY, one);
    TEST_INT_OPTION("TCP_KEEPIDLE", IPPROTO_TCP, TCP_KEEPIDLE, 20);
    TEST_INT_OPTION("TCP_KEEPINTVL", IPPROTO_TCP, TCP_KEEPINTVL, 5);
    TEST_INT_OPTION("TCP_KEEPCNT", IPPROTO_TCP, TCP_KEEPCNT, 3);

    broadcast_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    value = one;
    length = sizeof(value);
    errno = 0;
    CHECK("SO_BROADCAST", broadcast_fd >= 0 &&
            setsockopt(broadcast_fd, SOL_SOCKET, SO_BROADCAST,
                    &value, sizeof(value)) == 0 &&
            getsockopt(broadcast_fd, SOL_SOCKET, SO_BROADCAST,
                    &value, &length) == 0 && value != 0);
    if (broadcast_fd >= 0)
    {
        close(broadcast_fd);
    }

    CHECK("SO_LINGER", setsockopt(fd, SOL_SOCKET, SO_LINGER,
            &linger, sizeof(linger)) == 0);
    CHECK("SO_RCVTIMEO", setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
            &timeout, sizeof(timeout)) == 0);
    CHECK("SO_SNDTIMEO", setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO,
            &timeout, sizeof(timeout)) == 0);

    errno = 0;
    CHECK("invalid socket option", setsockopt(fd, SOL_SOCKET, 0x7fffffff,
            &one, sizeof(one)) < 0 && errno == ENOPROTOOPT);
    close(fd);
#undef TEST_INT_OPTION
}

static void test_tcp_echo(void)
{
    static const char expected[] = "sendmsg-scatter-gather";
    char part1[8] = { 0 };
    char part2[sizeof(expected) - sizeof(part1)] = { 0 };
    struct iovec send_iov[4];
    struct iovec recv_iov[3];
    struct msghdr empty_message;
    struct msghdr send_message;
    struct msghdr recv_message;
    struct sockaddr_in local;
    struct sockaddr_in remote;
    struct sockaddr_storage recv_name;
    socklen_t address_len;
    fd_set writefds;
    struct timeval timeout = { 1, 0 };
    int fd = connect_tcp();

    if (fd < 0)
    {
        FAIL("TCP connect");
        return;
    }
    PASS("TCP connect");

    address_len = sizeof(local);
    CHECK("getsockname", getsockname(fd, (struct sockaddr *)&local, &address_len) == 0 &&
            local.sin_family == AF_INET);
    address_len = sizeof(remote);
    CHECK("getpeername", getpeername(fd, (struct sockaddr *)&remote, &address_len) == 0 &&
            remote.sin_port == peer_address.sin_port);

    FD_ZERO(&writefds);
    FD_SET(fd, &writefds);
    CHECK("select writable", select(fd + 1, NULL, &writefds, NULL, &timeout) == 1 &&
            FD_ISSET(fd, &writefds));

    memset(&empty_message, 0, sizeof(empty_message));
    CHECK("TCP sendmsg zero iov", sendmsg(fd, &empty_message, MSG_NOSIGNAL) == 0);

    if (send_request(fd, SOCKET_TEST_TCP_ECHO, sizeof(expected) - 1) < 0)
    {
        FAIL("TCP request");
        close(fd);
        return;
    }

    memset(&send_message, 0, sizeof(send_message));
    send_iov[0].iov_base = NULL;
    send_iov[0].iov_len = 0;
    send_iov[1].iov_base = (void *)"sendmsg-";
    send_iov[1].iov_len = 8;
    send_iov[2].iov_base = (void *)"scatter-";
    send_iov[2].iov_len = 8;
    send_iov[3].iov_base = (void *)"gather";
    send_iov[3].iov_len = 6;
    send_message.msg_iov = send_iov;
    send_message.msg_iovlen = 4;
    CHECK("TCP sendmsg", sendmsg(fd, &send_message, MSG_NOSIGNAL) == 22);

    memset(&empty_message, 0, sizeof(empty_message));
    CHECK("TCP recvmsg zero iov", recvmsg(fd, &empty_message, 0) == 0);

    memset(&recv_message, 0, sizeof(recv_message));
    recv_iov[0].iov_base = NULL;
    recv_iov[0].iov_len = 0;
    recv_iov[1].iov_base = part1;
    recv_iov[1].iov_len = sizeof(part1);
    recv_iov[2].iov_base = part2;
    recv_iov[2].iov_len = sizeof(part2) - 1;
    recv_message.msg_iov = recv_iov;
    recv_message.msg_iovlen = 3;
    memset(&recv_name, 0xa5, sizeof(recv_name));
    recv_message.msg_name = &recv_name;
    recv_message.msg_namelen = sizeof(recv_name);
    CHECK("TCP recvmsg", recvmsg(fd, &recv_message, 0) == 22 &&
            memcmp(part1, expected, sizeof(part1)) == 0 &&
            memcmp(part2, expected + sizeof(part1), sizeof(part2) - 1) == 0 &&
            recv_message.msg_namelen == 0);
    close(fd);
}

static void test_posix_io(void)
{
    static const char expected[] = "writev-read";
    char receive_a[6] = { 0 };
    char receive_b[sizeof(expected) - sizeof(receive_a)] = { 0 };
    char read_buffer[sizeof(expected)] = { 0 };
    struct iovec send_iov[2];
    struct iovec receive_iov[2];
    int fd = connect_tcp();

    if (fd < 0 || send_request(fd, SOCKET_TEST_TCP_ECHO, sizeof(expected) - 1) < 0)
    {
        FAIL("POSIX I/O setup");
        if (fd >= 0) close(fd);
        return;
    }

    send_iov[0].iov_base = (void *)"writev-";
    send_iov[0].iov_len = 7;
    send_iov[1].iov_base = (void *)"read";
    send_iov[1].iov_len = 4;
    CHECK("writev socket", writev(fd, send_iov, 2) == 11);

    receive_iov[0].iov_base = receive_a;
    receive_iov[0].iov_len = sizeof(receive_a);
    receive_iov[1].iov_base = receive_b;
    receive_iov[1].iov_len = sizeof(receive_b) - 1;
    CHECK("readv socket", readv(fd, receive_iov, 2) == 11 &&
            memcmp(receive_a, expected, sizeof(receive_a)) == 0 &&
            memcmp(receive_b, expected + sizeof(receive_a), sizeof(receive_b) - 1) == 0);
    close(fd);

    fd = connect_tcp();
    if (fd < 0 || send_request(fd, SOCKET_TEST_TCP_ECHO, sizeof(expected) - 1) < 0)
    {
        FAIL("read/write setup");
        if (fd >= 0) close(fd);
        return;
    }
    CHECK("write socket", write(fd, expected, sizeof(expected) - 1) ==
            (ssize_t)(sizeof(expected) - 1));
    CHECK("read socket", read_full(fd, read_buffer, sizeof(expected) - 1) == 0 &&
            memcmp(read_buffer, expected, sizeof(expected) - 1) == 0);
    close(fd);
}

static void test_peek_poll_ioctl(void)
{
    char peek[9] = { 0 };
    char receive[9] = { 0 };
    ssize_t peeked;
    int available = 0;
    int peek_ok;
    int fd = connect_tcp();

    if (fd < 0 || send_request(fd, SOCKET_TEST_TCP_PEEK, 0) < 0)
    {
        FAIL("peek setup");
        if (fd >= 0) close(fd);
        return;
    }

    if (wait_for_event(fd, POLLIN, EVENT_TIMEOUT_MS) < 0)
    {
        FAIL("poll readable");
        close(fd);
        return;
    }
    PASS("poll readable");

    errno = 0;
    CHECK("ioctl FIONREAD", ioctl(fd, FIONREAD, &available) == 0 && available > 0);

    peeked = recv(fd, peek, sizeof(peek), MSG_PEEK);
    peek_ok = peeked > 0;
    if (peek_ok)
    {
        peek_ok = read_full(fd, receive, (size_t)peeked) == 0 &&
                memcmp(peek, receive, (size_t)peeked) == 0;
    }
    if (peek_ok && (size_t)peeked < sizeof(receive))
    {
        peek_ok = read_full(fd, receive + peeked,
                sizeof(receive) - (size_t)peeked) == 0;
    }
    CHECK("recv MSG_PEEK", peek_ok &&
            memcmp(receive, "peek-data", sizeof(receive)) == 0);

    CHECK("fcntl O_NONBLOCK", fcntl(fd, F_SETFL, O_NONBLOCK) == 0 &&
            (fcntl(fd, F_GETFL) & O_NONBLOCK));
    errno = 0;
    CHECK("recv MSG_DONTWAIT", recv(fd, receive, 1, MSG_DONTWAIT) <= 0 &&
            (errno == EAGAIN || errno == EWOULDBLOCK || errno == 0));
    close(fd);
}

static void test_shutdown(void)
{
    char response[13];
    int fd = connect_tcp();

    if (fd < 0 || send_request(fd, SOCKET_TEST_TCP_HALF_CLOSE, 0) < 0)
    {
        FAIL("shutdown setup");
        if (fd >= 0) close(fd);
        return;
    }
    CHECK("shutdown SHUT_WR", shutdown(fd, SHUT_WR) == 0);
    CHECK("half-close receive", read_full(fd, response, sizeof(response)) == 0 &&
            memcmp(response, "half-close-ok", sizeof(response)) == 0);
    close(fd);
}

static void test_accept4(void)
{
    struct sockaddr_in address;
    struct sockaddr_in from;
    socklen_t length;
    char payload[8];
    int listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    int control = -1;
    int accepted = -1;
    int accepting = 0;

    if (listener < 0)
    {
        FAIL("accept4 setup");
        return;
    }
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = 0;
    if (bind(listener, (const struct sockaddr *)&address, sizeof(address)) < 0 ||
            listen(listener, 1) < 0)
    {
        FAIL("accept4 setup");
        close(listener);
        return;
    }
    length = sizeof(accepting);
    CHECK("SO_ACCEPTCONN", getsockopt(listener, SOL_SOCKET, SO_ACCEPTCONN,
            &accepting, &length) == 0 && accepting != 0);
    length = sizeof(address);
    if (getsockname(listener, (struct sockaddr *)&address, &length) < 0 ||
            (control = connect_tcp()) < 0 ||
            send_request(control, SOCKET_TEST_CALLBACK, ntohs(address.sin_port)) < 0)
    {
        FAIL("accept4 callback");
        goto out;
    }

    length = sizeof(from);
    {
        int ready = wait_for_event(listener, POLLIN, EVENT_TIMEOUT_MS);

        if (ready < 0)
        {
            FAIL("accept4 callback");
            goto out;
        }
    }
    accepted = accept4(listener, (struct sockaddr *)&from, &length, SOCK_NONBLOCK);
    CHECK("accept4 SOCK_NONBLOCK", accepted >= 0 &&
            (fcntl(accepted, F_GETFL) & O_NONBLOCK));
    if (accepted >= 0)
    {
        int flags = fcntl(accepted, F_GETFL);
        int result = wait_for_event(accepted, POLLIN, EVENT_TIMEOUT_MS);

        if (result == 0 && flags >= 0)
        {
            result = fcntl(accepted, F_SETFL, flags & ~O_NONBLOCK);
        }
        CHECK("accept4 payload", result == 0 &&
                read_full(accepted, payload, sizeof(payload)) == 0 &&
                memcmp(payload, "callback", sizeof(payload)) == 0);
    }

#if defined(__riscv)
    errno = 0;
    CHECK("accept4 CLOEXEC policy", accept4(listener, NULL, NULL, SOCK_CLOEXEC) < 0 &&
            errno == EOPNOTSUPP);
#else
    SKIP("accept4 CLOEXEC policy", "host kernel supports close-on-exec");
#endif
out:
    if (accepted >= 0) close(accepted);
    if (control >= 0) close(control);
    close(listener);
}

static void prepare_udp_message(struct mmsghdr *message, struct iovec *iov,
        void *buffer, size_t length, struct sockaddr_in *address)
{
    memset(message, 0, sizeof(*message));
    iov->iov_base = buffer;
    iov->iov_len = length;
    message->msg_hdr.msg_iov = iov;
    message->msg_hdr.msg_iovlen = 1;
    message->msg_hdr.msg_name = address;
    message->msg_hdr.msg_namelen = sizeof(*address);
}

static void test_udp(void)
{
    static const char datagram[] = "udp-sendmsg-scatter";
    char receive_a[4];
    char receive_b[32];
    char small[4];
    struct sockaddr_in from;
    struct iovec send_iov[3];
    struct iovec receive_iov[2];
    struct msghdr send_message;
    struct msghdr receive_message;
    socklen_t from_len;
    int attempt;
    int ready;
    int fd = make_udp_socket();
    ssize_t result;

    if (fd < 0)
    {
        FAIL("UDP socket");
        return;
    }
    PASS("UDP socket");

    memset(&send_message, 0, sizeof(send_message));
    send_iov[0].iov_base = (void *)"udp-";
    send_iov[0].iov_len = 4;
    send_iov[1].iov_base = (void *)"sendmsg-";
    send_iov[1].iov_len = 8;
    send_iov[2].iov_base = (void *)"scatter";
    send_iov[2].iov_len = 7;
    send_message.msg_name = &peer_address;
    send_message.msg_namelen = sizeof(peer_address);
    send_message.msg_iov = send_iov;
    send_message.msg_iovlen = 3;
    result = sendmsg(fd, &send_message, 0);
    CHECK("UDP sendmsg", result == 19);
    if (result != 19)
    {
        close(fd);
        return;
    }

    ready = wait_for_event(fd, POLLIN, UDP_RETRY_TIMEOUT_MS);
    for (attempt = 1; ready < 0 && attempt < UDP_RETRY_COUNT; attempt++)
    {
        if (sendmsg(fd, &send_message, 0) != 19)
        {
            break;
        }
        ready = wait_for_event(fd, POLLIN, UDP_RETRY_TIMEOUT_MS);
    }
    if (ready < 0)
    {
        FAIL("UDP recvmsg");
        close(fd);
        return;
    }

    memset(&receive_message, 0, sizeof(receive_message));
    receive_iov[0].iov_base = receive_a;
    receive_iov[0].iov_len = sizeof(receive_a);
    receive_iov[1].iov_base = receive_b;
    receive_iov[1].iov_len = sizeof(receive_b);
    receive_message.msg_name = &from;
    receive_message.msg_namelen = sizeof(from);
    receive_message.msg_iov = receive_iov;
    receive_message.msg_iovlen = 2;
    CHECK("UDP recvmsg", recvmsg(fd, &receive_message, 0) == 19 &&
            memcmp(receive_a, datagram, sizeof(receive_a)) == 0 &&
            memcmp(receive_b, datagram + sizeof(receive_a), 15) == 0 &&
            from.sin_port == peer_address.sin_port &&
            receive_message.msg_namelen == sizeof(from));

    close(fd);
    fd = make_udp_socket();
    if (fd < 0)
    {
        FAIL("UDP sendto setup");
        return;
    }

    result = sendto(fd, datagram, sizeof(datagram) - 1, 0,
            (const struct sockaddr *)&peer_address, sizeof(peer_address));
    CHECK("UDP sendto", result == (ssize_t)(sizeof(datagram) - 1));
    if (result != (ssize_t)(sizeof(datagram) - 1))
    {
        close(fd);
        return;
    }
    ready = wait_for_event(fd, POLLIN, UDP_RETRY_TIMEOUT_MS);
    for (attempt = 1; ready < 0 && attempt < UDP_RETRY_COUNT; attempt++)
    {
        if (sendto(fd, datagram, sizeof(datagram) - 1, 0,
                (const struct sockaddr *)&peer_address,
                sizeof(peer_address)) != (ssize_t)(sizeof(datagram) - 1))
        {
            break;
        }
        ready = wait_for_event(fd, POLLIN, UDP_RETRY_TIMEOUT_MS);
    }
    if (ready < 0)
    {
        FAIL("UDP MSG_TRUNC");
        close(fd);
        return;
    }
    memset(&receive_message, 0, sizeof(receive_message));
    receive_iov[0].iov_base = small;
    receive_iov[0].iov_len = sizeof(small);
    receive_message.msg_name = &from;
    receive_message.msg_namelen = sizeof(from.sin_family);
    receive_message.msg_iov = receive_iov;
    receive_message.msg_iovlen = 1;
    CHECK("UDP MSG_TRUNC", recvmsg(fd, &receive_message, MSG_TRUNC) ==
            (ssize_t)(sizeof(datagram) - 1) &&
            (receive_message.msg_flags & MSG_TRUNC) &&
            receive_message.msg_namelen == sizeof(from));

    close(fd);
    fd = make_udp_socket();
    if (fd < 0)
    {
        FAIL("UDP connect setup");
        return;
    }

    result = connect(fd, (const struct sockaddr *)&peer_address,
            sizeof(peer_address));
    if (result == 0)
    {
        result = send(fd, "connected", 9, 0);
    }
    CHECK("UDP connect", result == 9);
    if (result != 9)
    {
        close(fd);
        return;
    }
    ready = wait_for_event(fd, POLLIN, UDP_RETRY_TIMEOUT_MS);
    for (attempt = 1; ready < 0 && attempt < UDP_RETRY_COUNT; attempt++)
    {
        if (send(fd, "connected", 9, 0) != 9)
        {
            break;
        }
        ready = wait_for_event(fd, POLLIN, UDP_RETRY_TIMEOUT_MS);
    }
    if (ready < 0)
    {
        FAIL("UDP recvfrom");
        close(fd);
        return;
    }
    from_len = sizeof(from);
    CHECK("UDP recvfrom", recvfrom(fd, receive_b, sizeof(receive_b), 0,
            (struct sockaddr *)&from, &from_len) == 9);
    close(fd);
}

static void test_mmsg(void)
{
    char send_a[] = "mmsg-one";
    char send_b[] = "mmsg-two";
    char recv_a[16] = { 0 };
    char recv_b[16] = { 0 };
    struct sockaddr_in from[2];
    struct iovec send_iov[2];
    struct iovec receive_iov[2];
    struct mmsghdr send_messages[2];
    struct mmsghdr receive_messages[2];
    struct timespec timeout = { 3, 0 };
    int attempt;
    int fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    int ready;
    int result;
    int valid;

    if (fd < 0)
    {
        FAIL("mmsg setup");
        return;
    }
    errno = 0;
    CHECK("sendmmsg vlen zero", sendmmsg(fd, NULL, 0, 0) == 0);
    errno = 0;
    CHECK("recvmmsg vlen zero", recvmmsg(fd, NULL, 0, 0, NULL) < 0 &&
            errno == EINVAL);
    errno = 0;
    CHECK("sendmmsg NULL vector", sendmmsg(fd, NULL, 1, 0) < 0 &&
            errno == EFAULT);
    errno = 0;
    CHECK("recvmmsg NULL vector", recvmmsg(fd, NULL, 1, 0, NULL) < 0 &&
            errno == EFAULT);
    errno = 0;
    CHECK("recvmmsg vlen limit", recvmmsg(fd, receive_messages, IOV_MAX + 1U,
            MSG_DONTWAIT, NULL) < 0 && errno == EINVAL);
    prepare_udp_message(&send_messages[0], &send_iov[0], send_a, 8, &peer_address);
    prepare_udp_message(&send_messages[1], &send_iov[1], send_b, 8, &peer_address);
    result = sendmmsg(fd, send_messages, 2, 0);
    CHECK("sendmmsg", result == 2 &&
            send_messages[0].msg_len == 8 && send_messages[1].msg_len == 8);
    if (result != 2)
    {
        close(fd);
        return;
    }

    prepare_udp_message(&receive_messages[0], &receive_iov[0], recv_a, sizeof(recv_a), &from[0]);
    prepare_udp_message(&receive_messages[1], &receive_iov[1], recv_b, sizeof(recv_b), &from[1]);
    ready = wait_for_event(fd, POLLIN, UDP_RETRY_TIMEOUT_MS);
    for (attempt = 1; ready < 0 && attempt < UDP_RETRY_COUNT; attempt++)
    {
        if (sendmmsg(fd, send_messages, 2, 0) != 2)
        {
            break;
        }
        ready = wait_for_event(fd, POLLIN, UDP_RETRY_TIMEOUT_MS);
    }
    if (ready < 0)
    {
        FAIL("recvmmsg");
        close(fd);
        return;
    }

    result = recvmmsg(fd, receive_messages, 2, MSG_WAITFORONE, &timeout);
    valid = result > 0 && result <= 2;
    for (attempt = 0; valid && attempt < result; attempt++)
    {
        const char *received = attempt == 0 ? recv_a : recv_b;

        valid = receive_messages[attempt].msg_len == 8 &&
                (memcmp(received, send_a, 8) == 0 ||
                 memcmp(received, send_b, 8) == 0);
    }
    CHECK("recvmmsg", valid);
    close(fd);

    fd = make_udp_socket();
    if (fd < 0)
    {
        FAIL("recvmmsg MSG_WAITFORONE setup");
        return;
    }
    result = sendto(fd, send_a, 8, 0, (const struct sockaddr *)&peer_address,
            sizeof(peer_address));
    ready = result == 8 ? wait_for_event(fd, POLLIN, UDP_RETRY_TIMEOUT_MS) : -1;
    if (ready < 0)
    {
        FAIL("recvmmsg MSG_WAITFORONE setup");
    }
    else
    {
        memset(recv_a, 0, sizeof(recv_a));
        memset(recv_b, 0, sizeof(recv_b));
        prepare_udp_message(&receive_messages[0], &receive_iov[0], recv_a,
                sizeof(recv_a), &from[0]);
        prepare_udp_message(&receive_messages[1], &receive_iov[1], recv_b,
                sizeof(recv_b), &from[1]);
        timeout.tv_sec = 3;
        timeout.tv_nsec = 0;
        result = recvmmsg(fd, receive_messages, 2, MSG_WAITFORONE, &timeout);
        CHECK("recvmmsg MSG_WAITFORONE", result == 1 &&
                (timeout.tv_sec > 0 || timeout.tv_nsec > 0));
    }
    close(fd);

    fd = make_udp_socket();
    if (fd < 0)
    {
        FAIL("recvmmsg MSG_DONTWAIT setup");
        return;
    }
    prepare_udp_message(&receive_messages[0], &receive_iov[0], recv_a,
            sizeof(recv_a), &from[0]);
    timeout.tv_sec = 3;
    timeout.tv_nsec = 0;
    errno = 0;
    result = recvmmsg(fd, receive_messages, 1, MSG_DONTWAIT, &timeout);
    CHECK("recvmmsg MSG_DONTWAIT", result < 0 &&
            (errno == EAGAIN || errno == EWOULDBLOCK) && timeout.tv_sec == 3 &&
            timeout.tv_nsec == 0);
    close(fd);
}

static void test_optional_features(void)
{
    int fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    int one = 1;
    int ancillary_result;
    struct timeval timeout = { SOCKET_IO_TIMEOUT_SEC, 0 };

    if (fd < 0)
    {
        FAIL("optional setup");
        return;
    }
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#ifdef IP_PKTINFO
    errno = 0;
    if (setsockopt(fd, IPPROTO_IP, IP_PKTINFO, &one, sizeof(one)) == 0)
    {
        char byte = 'p';
        char received = 0;
        char control[CMSG_SPACE(sizeof(struct in_pktinfo))] = { 0 };
        struct sockaddr_in bind_address;
        struct sockaddr_in destination;
        struct iovec iov = { &received, sizeof(received) };
        struct msghdr message;
        struct cmsghdr *cmsg;
        struct in_pktinfo *pktinfo = NULL;
        socklen_t destination_len = sizeof(destination);

        memset(&bind_address, 0, sizeof(bind_address));
        bind_address.sin_family = AF_INET;
        bind_address.sin_addr.s_addr = htonl(INADDR_ANY);
        if (bind(fd, (const struct sockaddr *)&bind_address, sizeof(bind_address)) == 0 &&
                getsockname(fd, (struct sockaddr *)&destination, &destination_len) == 0)
        {
            destination.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        }
        (void)sendto(fd, &byte, sizeof(byte), 0,
                (const struct sockaddr *)&destination, sizeof(destination));
        memset(&message, 0, sizeof(message));
        message.msg_iov = &iov;
        message.msg_iovlen = 1;
        message.msg_control = control;
        message.msg_controllen = sizeof(control);
        if (recvmsg(fd, &message, 0) == 1)
        {
            cmsg = CMSG_FIRSTHDR(&message);
            if (cmsg && cmsg->cmsg_level == IPPROTO_IP && cmsg->cmsg_type == IP_PKTINFO &&
                    cmsg->cmsg_len >= CMSG_LEN(sizeof(*pktinfo)))
            {
                pktinfo = (struct in_pktinfo *)CMSG_DATA(cmsg);
            }
        }
        CHECK("IP_PKTINFO receive", pktinfo && pktinfo->ipi_ifindex > 0 &&
                pktinfo->ipi_addr.s_addr == htonl(INADDR_LOOPBACK));
    }
    else if (errno == ENOPROTOOPT)
    {
        SKIP("IP_PKTINFO enable", "LWIP_NETBUF_RECVINFO is disabled");
    }
    else
    {
        FAIL("IP_PKTINFO enable");
    }
#else
    SKIP("IP_PKTINFO enable", "not in libc headers");
#endif

#if defined(__riscv)
    errno = 0;
    {
        char control[CMSG_SPACE(sizeof(int))] = { 0 };
        char byte = 'x';
        struct iovec iov = { &byte, 1 };
        struct msghdr message;
        memset(&message, 0, sizeof(message));
        message.msg_name = &peer_address;
        message.msg_namelen = sizeof(peer_address);
        message.msg_iov = &iov;
        message.msg_iovlen = 1;
        message.msg_control = control;
        message.msg_controllen = sizeof(control);
        ancillary_result = sendmsg(fd, &message, 0);
    }
    CHECK("sendmsg ancillary policy", ancillary_result < 0 && errno == EOPNOTSUPP);
#else
    (void)ancillary_result;
    SKIP("sendmsg ancillary policy", "host kernel behavior differs from lwIP");
#endif
    close(fd);
}

static int resolve_peer(const char *host, uint16_t port)
{
    struct addrinfo hints;
    struct addrinfo *result = NULL;
    char service[8];
    int error;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    snprintf(service, sizeof(service), "%u", port);
    error = getaddrinfo(host, service, &hints, &result);
    if (error != 0 || !result)
    {
        fprintf(stderr, "getaddrinfo(%s): %s\n", host, gai_strerror(error));
        return -1;
    }
    memcpy(&peer_address, result->ai_addr, sizeof(peer_address));
    freeaddrinfo(result);
    return 0;
}

int main(int argc, char **argv)
{
    uint16_t port = DEFAULT_PORT;

    if (argc < 2 || argc > 3)
    {
        fprintf(stderr, "usage: %s <pc-ip-or-name> [port]\n", argv[0]);
        return 2;
    }
    if (argc == 3)
    {
        long parsed = strtol(argv[2], NULL, 10);
        if (parsed <= 0 || parsed > 65535)
        {
            fprintf(stderr, "invalid port\n");
            return 2;
        }
        port = (uint16_t)parsed;
    }
    if (resolve_peer(argv[1], port) < 0)
    {
        return 2;
    }

    test_creation_and_errors();
    test_socket_options();
    test_tcp_echo();
    test_posix_io();
    test_peek_poll_ioctl();
    test_shutdown();
    test_accept4();
    test_udp();
    test_mmsg();
    test_optional_features();

    printf("\nSUMMARY pass=%d fail=%d skip=%d\n", passed, failed, skipped);
    return failed ? 1 : 0;
}
