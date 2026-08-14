#ifndef SOCKET_TEST_PROTOCOL_H
#define SOCKET_TEST_PROTOCOL_H

#include <stdint.h>

#define SOCKET_TEST_MAGIC 0x534f434bU

enum socket_test_operation
{
    SOCKET_TEST_TCP_ECHO = 1,
    SOCKET_TEST_TCP_PEEK = 2,
    SOCKET_TEST_TCP_HALF_CLOSE = 3,
    SOCKET_TEST_CALLBACK = 4,
};

struct socket_test_request
{
    uint32_t magic;
    uint32_t operation;
    uint32_t length;
};

#endif
