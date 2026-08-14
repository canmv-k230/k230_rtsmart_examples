#define _GNU_SOURCE

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "socket_test_protocol.h"

#define DEFAULT_PORT 5202
#define MAX_PAYLOAD  65535U
#define CLIENT_IO_TIMEOUT_SEC 5

static volatile sig_atomic_t running = 1;

static void stop_peer(int signal_number)
{
    (void)signal_number;
    running = 0;
}

static int read_full(int fd, void *buffer, size_t length)
{
    size_t done = 0;

    while (running && done < length)
    {
        ssize_t result = recv(fd, (char *)buffer + done, length - done, 0);
        if (result == 0)
        {
            return -1;
        }
        if (result < 0)
        {
            if (errno == EINTR)
            {
                if (running)
                {
                    continue;
                }
            }
            return -1;
        }
        done += (size_t)result;
    }
    return done == length ? 0 : -1;
}

static int write_full(int fd, const void *buffer, size_t length)
{
    size_t done = 0;

    while (running && done < length)
    {
        ssize_t result = send(fd, (const char *)buffer + done, length - done,
                MSG_NOSIGNAL);
        if (result < 0)
        {
            if (errno == EINTR)
            {
                if (running)
                {
                    continue;
                }
            }
            return -1;
        }
        if (result == 0)
        {
            return -1;
        }
        done += (size_t)result;
    }
    return done == length ? 0 : -1;
}

static int connect_callback(const struct sockaddr_in *client, uint16_t port)
{
    struct sockaddr_in address = *client;
    int fd;

    address.sin_port = htons(port);
    fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd < 0)
    {
        return -1;
    }
    if (connect(fd, (const struct sockaddr *)&address, sizeof(address)) < 0 ||
            write_full(fd, "callback", 8) < 0)
    {
        close(fd);
        return -1;
    }
    close(fd);
    return 0;
}

static void handle_tcp(int fd, const struct sockaddr_in *client)
{
    struct socket_test_request wire;
    uint32_t operation;
    uint32_t length;
    char *payload = NULL;

    if (read_full(fd, &wire, sizeof(wire)) < 0 ||
            ntohl(wire.magic) != SOCKET_TEST_MAGIC)
    {
        return;
    }
    operation = ntohl(wire.operation);
    length = ntohl(wire.length);
    if (length > MAX_PAYLOAD)
    {
        return;
    }

    switch (operation)
    {
    case SOCKET_TEST_TCP_ECHO:
        payload = malloc(length ? length : 1);
        if (payload && read_full(fd, payload, length) == 0)
        {
            (void)write_full(fd, payload, length);
        }
        break;
    case SOCKET_TEST_TCP_PEEK:
        (void)write_full(fd, "peek-data", 9);
        break;
    case SOCKET_TEST_TCP_HALF_CLOSE:
        {
            char buffer[64];
            ssize_t result;
            do
            {
                result = recv(fd, buffer, sizeof(buffer), 0);
            } while (running && (result > 0 || (result < 0 && errno == EINTR)));
            if (result == 0)
            {
                (void)write_full(fd, "half-close-ok", 13);
            }
        }
        break;
    case SOCKET_TEST_CALLBACK:
        if (length > 0 && length <= 65535U)
        {
            (void)connect_callback(client, (uint16_t)length);
        }
        break;
    default:
        break;
    }

    free(payload);
}

static int make_bound_socket(int type, uint16_t port)
{
    struct sockaddr_in address;
    int one = 1;
    int fd = socket(AF_INET, type, 0);

    if (fd < 0)
    {
        return -1;
    }
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port);
    if (bind(fd, (const struct sockaddr *)&address, sizeof(address)) < 0)
    {
        close(fd);
        return -1;
    }
    return fd;
}

static void set_client_timeouts(int fd)
{
    struct timeval timeout = { CLIENT_IO_TIMEOUT_SEC, 0 };

    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

int main(int argc, char **argv)
{
    struct sigaction action;
    uint16_t port = DEFAULT_PORT;
    int tcp_fd;
    int udp_fd;

    if (argc > 2)
    {
        fprintf(stderr, "usage: %s [port]\n", argv[0]);
        return 2;
    }
    if (argc == 2)
    {
        long parsed = strtol(argv[1], NULL, 10);
        if (parsed <= 0 || parsed > 65535)
        {
            fprintf(stderr, "invalid port\n");
            return 2;
        }
        port = (uint16_t)parsed;
    }

    memset(&action, 0, sizeof(action));
    action.sa_handler = stop_peer;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);
    signal(SIGPIPE, SIG_IGN);

    tcp_fd = make_bound_socket(SOCK_STREAM, port);
    udp_fd = make_bound_socket(SOCK_DGRAM, port);
    if (tcp_fd < 0 || udp_fd < 0 || listen(tcp_fd, 8) < 0)
    {
        perror("socket setup");
        return 1;
    }

    printf("socket peer listening on TCP/UDP %u\n", port);
    fflush(stdout);

    while (running)
    {
        fd_set readfds;
        struct timeval timeout = { 1, 0 };
        int maxfd = tcp_fd > udp_fd ? tcp_fd : udp_fd;
        int ready;

        FD_ZERO(&readfds);
        FD_SET(tcp_fd, &readfds);
        FD_SET(udp_fd, &readfds);
        ready = select(maxfd + 1, &readfds, NULL, NULL, &timeout);
        if (ready < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            perror("select");
            break;
        }
        if (FD_ISSET(udp_fd, &readfds))
        {
            struct sockaddr_storage from;
            socklen_t fromlen = sizeof(from);
            char buffer[MAX_PAYLOAD];
            ssize_t length = recvfrom(udp_fd, buffer, sizeof(buffer), 0,
                    (struct sockaddr *)&from, &fromlen);
            if (length >= 0)
            {
                (void)sendto(udp_fd, buffer, (size_t)length, 0,
                        (const struct sockaddr *)&from, fromlen);
            }
        }
        if (FD_ISSET(tcp_fd, &readfds))
        {
            struct sockaddr_in client;
            socklen_t client_len = sizeof(client);
            int client_fd = accept(tcp_fd, (struct sockaddr *)&client, &client_len);
            if (client_fd >= 0)
            {
                set_client_timeouts(client_fd);
                handle_tcp(client_fd, &client);
                close(client_fd);
            }
        }
    }

    close(udp_fd);
    close(tcp_fd);
    return 0;
}
