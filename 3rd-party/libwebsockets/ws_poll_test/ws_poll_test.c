/*
 * ws_poll_test - isolation test for non-blocking connect() + poll(POLLOUT)
 *                on RT-Smart / SAL / lwIP.
 *
 * Usage:
 *   ws_poll_test.elf <host> <port>            raw non-blocking connect
 *   ws_poll_test.elf <host> <port> -l         lws-emulation mode
 *   ws_poll_test.elf <host> <port> -b         blocking connect (sanity)
 *   ws_poll_test.elf <host> <port> -l -b      both -l and -b combined
 *   ws_poll_test.elf <host> <port> -t <ms>    poll timeout (default 10000)
 *
 * The -l flag applies the exact same socket options as lws's
 * lws_plat_set_socket_options(): fcntl(FD_CLOEXEC) + setsockopt(TCP_NODELAY)
 * via SOL_TCP(=IPPROTO_TCP=6).  If the raw test passes but -l hangs, the
 * root cause of ws_client's "stuck in LRS_WAITING_CONNECT" is setsockopt
 * corrupting the lwIP socket state before connect().
 *
 * Exit code: 0 = PASS, 1 = FAIL.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

static int parse_positive_int(const char *s, int *out)
{
	char *end = NULL;
	long v = strtol(s, &end, 10);
	if (!s[0] || (end && *end) || v < 1 || v > 65535)
		return -1;
	*out = (int)v;
	return 0;
}

static const char *revents_str(short re, char *buf, size_t n)
{
	buf[0] = 0;
	if (re & POLLIN)  strncat(buf, "IN ", n - strlen(buf) - 1);
	if (re & POLLOUT) strncat(buf, "OUT ", n - strlen(buf) - 1);
	if (re & POLLERR) strncat(buf, "ERR ", n - strlen(buf) - 1);
	if (re & POLLHUP) strncat(buf, "HUP ", n - strlen(buf) - 1);
	if (re & POLLNVAL) strncat(buf, "NVAL ", n - strlen(buf) - 1);
	if (!buf[0]) strncat(buf, "(none)", n - strlen(buf) - 1);
	return buf;
}

static int do_blocking_connect(const char *host, int port, int lws_opts)
{
	struct sockaddr_in addr;
	int fd, r, optval;
	char rbuf[256];

	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_port = htons((uint16_t)port);
	if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
		printf("invalid host '%s' (use an IPv4 literal)\n", host);
		return 1;
	}

	fd = socket(AF_INET, SOCK_STREAM, 0);
	if (fd < 0) {
		printf("socket() failed: %d (%s)\n", errno, strerror(errno));
		return 1;
	}

	if (lws_opts) {
		optval = 1;
		(void)fcntl(fd, F_SETFD, 1 /* FD_CLOEXEC */);
		printf("[lws] fcntl(F_SETFD, FD_CLOEXEC) done\n");
		r = setsockopt(fd, 6 /* SOL_TCP -> IPPROTO_TCP */,
			       0x01 /* TCP_NODELAY */, &optval, sizeof(optval));
		printf("[lws] setsockopt(SOL_TCP, TCP_NODELAY, 1) = %d,"
		       " errno=%d (%s)\n", r, errno, strerror(errno));
	}

	printf("[blocking] connecting to %s:%d ...\n", host, port);
	r = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
	printf("[blocking] connect() returned %d, errno=%d (%s)\n",
	       r, errno, strerror(errno));
	if (r < 0) { close(fd); return 1; }

	const char *req = "GET / HTTP/1.0\r\nHost: %s\r\n\r\n";
	char sendbuf[128];
	snprintf(sendbuf, sizeof(sendbuf), req, host);
	r = (int)send(fd, sendbuf, strlen(sendbuf), 0);
	printf("[blocking] sent %d bytes of HTTP request\n", r);

	r = (int)recv(fd, rbuf, sizeof(rbuf) - 1, 0);
	if (r > 0) {
		rbuf[r] = 0;
		printf("[blocking] recv %d bytes:\n%.120s\n", r, rbuf);
	} else {
		printf("[blocking] recv returned %d, errno=%d (%s)\n",
		       r, errno, strerror(errno));
	}
	close(fd);
	return 0;
}

int main(int argc, char **argv)
{
	const char *host = NULL;
	int port = 0, lws_opts = 0, blocking = 0, timeout_ms = 10000;
	struct sockaddr_in addr;
	struct pollfd pfd;
	char rbuf[256], rebuf[32];
	int fd, r, soerr, opt, optval;
	socklen_t sl;
	struct timespec t0, t1;
	int i;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-l")) lws_opts = 1;
		else if (!strcmp(argv[i], "-b")) blocking = 1;
		else if (!strcmp(argv[i], "-t")) {
			if (++i >= argc || parse_positive_int(argv[i], &timeout_ms)) {
				printf("invalid -t timeout\n"); return 1;
			}
		} else if (!host) host = argv[i];
		else if (!port) {
			if (parse_positive_int(argv[i], &port)) {
				printf("invalid port: %s\n", argv[i]); return 1;
			}
		} else { printf("unexpected arg: %s\n", argv[i]); return 1; }
	}

	if (!host || !port) {
		printf("Usage: %s <host> <port> [-l] [-b] [-t <ms>]\n", argv[0]);
		return 1;
	}

	if (blocking)
		return do_blocking_connect(host, port, lws_opts);

	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_port = htons((uint16_t)port);
	if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
		printf("invalid host '%s' (use an IPv4 literal)\n", host);
		return 1;
	}

	fd = socket(AF_INET, SOCK_STREAM, 0);
	if (fd < 0) {
		printf("socket() failed: %d (%s)\n", errno, strerror(errno));
		return 1;
	}
	printf("socket() = %d\n", fd);

	/*
	 * Step A: lws-emulation mode: apply the exact same socket options
	 * as lws_plat_set_socket_options() before connect.
	 *
	 * lws does:  fcntl(FD_CLOEXEC)
	 *            setsockopt(SOL_TCP, TCP_NODELAY, 1)   where SOL_TCP = IPPROTO_TCP = 6
	 *            fcntl(O_NONBLOCK)
	 *
	 * We must do this BEFORE fcntl(O_NONBLOCK) to match lws order exactly.
	 */
	if (lws_opts) {
		optval = 1;
		(void)fcntl(fd, F_SETFD, 1);  /* FD_CLOEXEC = 1 */
		printf("[lws] fcntl(F_SETFD, FD_CLOEXEC) done\n");

		r = setsockopt(fd, 6 /* SOL_TCP=IPPROTO_TCP */,
			       0x01 /* TCP_NODELAY */,
			       &optval, sizeof(optval));
		printf("[lws] setsockopt(SOL_TCP, TCP_NODELAY, 1) = %d,"
		       " errno=%d (%s)\n",
		       r, errno, strerror(errno));

		/* Now fcntl(O_NONBLOCK) — lws calls this inside
		 * lws_plat_set_nonblocking(), which is the LAST thing
		 * lws_plat_set_socket_options() does. */
		r = fcntl(fd, F_SETFL, O_NONBLOCK);
		printf("[lws] fcntl(F_SETFL, O_NONBLOCK) = %d, errno=%d (%s)\n",
		       r, errno, strerror(errno));
		opt = fcntl(fd, F_GETFL);
		printf("[lws] fcntl(F_GETFL) = 0x%x (O_NONBLOCK %sSET)\n",
		       opt, (opt & O_NONBLOCK) ? "" : "NOT ");
		if (r < 0 || !(opt & O_NONBLOCK)) {
			printf("FAIL: could not set O_NONBLOCK - connect() will block\n");
			close(fd);
			return 1;
		}

		goto do_connect;
	}

	/* Step 1 (raw): set non-blocking. */
	r = fcntl(fd, F_SETFL, O_NONBLOCK);
	printf("fcntl(F_SETFL, O_NONBLOCK) = %d, errno=%d (%s)\n",
	       r, errno, strerror(errno));
	opt = fcntl(fd, F_GETFL);
	printf("fcntl(F_GETFL) = 0x%x (O_NONBLOCK %sSET)\n",
	       opt, (opt & O_NONBLOCK) ? "" : "NOT ");
	if (r < 0 || !(opt & O_NONBLOCK)) {
		printf("FAIL: could not set O_NONBLOCK - connect() will block\n");
		close(fd);
		return 1;
	}

do_connect:
	/* Step 2: non-blocking connect. */
	printf("connect() to %s:%d ...\n", host, port);
	r = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
	printf("connect() = %d, errno=%d (%s)\n", r, errno, strerror(errno));
	if (r == 0) {
		printf("NOTE: connect() returned 0 immediately.\n");
	} else if (r < 0 && errno != EINPROGRESS && errno != EALREADY &&
		   errno != EWOULDBLOCK) {
		printf("FAIL: connect() failed fatally\n");
		close(fd);
		return 1;
	}

	/* Step 3: poll for POLLOUT. */
	pfd.fd = fd;
	pfd.events = POLLOUT;
	pfd.revents = 0;

	printf("poll(POLLOUT, %d ms) ...\n", timeout_ms);
	clock_gettime(CLOCK_MONOTONIC, &t0);
	r = poll(&pfd, 1, timeout_ms);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	printf("poll() = %d, revents=[%s], elapsed=%ld ms\n",
	       r, revents_str(pfd.revents, rebuf, sizeof(rebuf)),
	       (long)((t1.tv_sec - t0.tv_sec) * 1000 +
		      (t1.tv_nsec - t0.tv_nsec) / 1000000));

	if (r <= 0) {
		printf("FAIL: poll returned %d - POLLOUT never delivered for "
		       "connect completion%s\n", r,
		       r == 0 ? " (timeout)" : " (error)");
		printf("  => This is the root cause of the ws_client hang.\n");
		close(fd);
		return 1;
	}

	if (!(pfd.revents & POLLOUT)) {
		printf("FAIL: poll returned but POLLOUT not set: [%s]\n",
		       revents_str(pfd.revents, rebuf, sizeof(rebuf)));
		close(fd);
		return 1;
	}

	/* Step 4: confirm connect complete via SO_ERROR. */
	soerr = 0;
	sl = sizeof(soerr);
	r = getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &sl);
	printf("getsockopt(SO_ERROR) = %d, soerr=%d (%s)\n", r, soerr,
	       strerror(soerr));
	if (soerr != 0) {
		printf("FAIL: connect completed with error %d (%s)\n",
		       soerr, strerror(soerr));
		close(fd);
		return 1;
	}

	printf("PASS: non-blocking connect completion was reported as POLLOUT.\n");

	/* Step 5: minimal HTTP round-trip to prove socket is usable. */
	const char *req = "GET / HTTP/1.0\r\nHost: %s\r\n\r\n";
	char sendbuf[128];
	snprintf(sendbuf, sizeof(sendbuf), req, host);
	r = (int)send(fd, sendbuf, strlen(sendbuf), 0);
	printf("send() = %d bytes\n", r);

	pfd.events = POLLIN;
	pfd.revents = 0;
	r = poll(&pfd, 1, 5000);
	printf("poll(POLLIN) = %d, revents=[%s]\n", r,
	       revents_str(pfd.revents, rebuf, sizeof(rebuf)));
	if (r > 0 && (pfd.revents & POLLIN)) {
		r = (int)recv(fd, rbuf, sizeof(rbuf) - 1, 0);
		if (r > 0) {
			rbuf[r] = 0;
			printf("recv() = %d bytes:\n%.200s\n", r, rbuf);
			printf("PASS: full non-blocking HTTP round-trip OK.\n");
			close(fd);
			return 0;
		}
		/* recv() == 0 means peer closed (WS-only server rejects plain
		 * HTTP) — the connect+send path still worked. */
		if (r == 0) {
			printf("PASS: connect+send OK (peer closed after"
			       " receiving; expected against a WS-only server).\n");
			close(fd);
			return 0;
		}
		printf("recv() = %d, errno=%d (%s)\n", r, errno, strerror(errno));
	}

	printf("FAIL: no HTTP response received\n");
	close(fd);
	return 1;
}
