/*
 * libwebsockets WebSocket Client Example for RT-Smart
 *
 * Connects to a websocket server, sends a text message, and prints any
 * received messages.
 *
 * Uses lws's internal service loop (lws_service()), matching ws_server.c.
 * Do not use the external-poll pattern here unless the library is built with
 * LWS_WITH_EXTERNAL_POLL - without that flag the ADD/DEL/CHANGE_POLL_FD
 * callbacks are compiled out and the poll set stays empty.
 */

#include <libwebsockets.h>
#include <string.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define EXAMPLE_SERVER "127.0.0.1"
#define EXAMPLE_PORT 7681
#define EXAMPLE_PATH "/"
#define EXAMPLE_MESSAGE "Hello from RT-Smart WebSocket client!"
#define EXAMPLE_TIMEOUT_SECS 10
#define EXAMPLE_SERVICE_MS 50

static int interrupted;
static int connected;
static int message_sent;
static const char *server_address = EXAMPLE_SERVER;
static const char *server_path = EXAMPLE_PATH;
static const char *client_message = EXAMPLE_MESSAGE;
static int server_port = EXAMPLE_PORT;
static int timeout_secs = EXAMPLE_TIMEOUT_SECS;

static void
example_lws_log(int level, const char *line)
{
	(void)level;
	fputs(line, stderr);
}

static void
print_usage(const char *prog)
{
	printf("Usage:\n");
	printf("  %s [host] [port]\n", prog);
	printf("  %s -a <host> [-p <port>] [-u <path>] [-m <message>] [-t <timeout_secs>]\n", prog);
	printf("\nDefaults: host=%s port=%d path=%s timeout=%d\n", EXAMPLE_SERVER,
	       EXAMPLE_PORT, EXAMPLE_PATH, EXAMPLE_TIMEOUT_SECS);
}

static int
parse_positive_int(const char *arg, int min, int max, int *value_out)
{
	char *end = NULL;
	long value;

	value = strtol(arg, &end, 10);
	if (!arg[0] || (end && *end) || value < min || value > max)
		return -1;

	*value_out = (int)value;
	return 0;
}

static int
parse_args(int argc, char **argv)
{
	int positional = 0;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-help") || !strcmp(argv[i], "--help")) {
			print_usage(argv[0]);
			return 1;
		} else if (!strcmp(argv[i], "-a") || !strcmp(argv[i], "--address")) {
			if (++i >= argc) {
				printf("missing host after %s\n", argv[i - 1]);
				return -1;
			}
			server_address = argv[i];
		} else if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--port")) {
			if (++i >= argc || parse_positive_int(argv[i], 1, 65535, &server_port)) {
				printf("invalid port\n");
				return -1;
			}
		} else if (!strcmp(argv[i], "-u") || !strcmp(argv[i], "--path")) {
			if (++i >= argc || argv[i][0] != '/') {
				printf("path must start with /\n");
				return -1;
			}
			server_path = argv[i];
		} else if (!strcmp(argv[i], "-m") || !strcmp(argv[i], "--message")) {
			if (++i >= argc) {
				printf("missing message after %s\n", argv[i - 1]);
				return -1;
			}
			client_message = argv[i];
		} else if (!strcmp(argv[i], "-t") || !strcmp(argv[i], "--timeout")) {
			if (++i >= argc || parse_positive_int(argv[i], 1, 3600, &timeout_secs)) {
				printf("invalid timeout\n");
				return -1;
			}
		} else if (argv[i][0] == '-') {
			printf("unknown option: %s\n", argv[i]);
			return -1;
		} else if (positional == 0) {
			server_address = argv[i];
			positional++;
		} else if (positional == 1) {
			if (parse_positive_int(argv[i], 1, 65535, &server_port)) {
				printf("invalid port: %s\n", argv[i]);
				return -1;
			}
			positional++;
		} else {
			printf("unexpected argument: %s\n", argv[i]);
			return -1;
		}
	}

	return 0;
}

static int
callback_ws(struct lws *wsi, enum lws_callback_reasons reason,
	    void *user, void *in, size_t len)
{
	(void)user;

	switch (reason) {
	case LWS_CALLBACK_CLIENT_ESTABLISHED:
		connected = 1;
		lwsl_user("WS Client: connected to server\n");
		lws_callback_on_writable(wsi);
		break;

	case LWS_CALLBACK_CLIENT_RECEIVE:
		lwsl_user("WS Client: received %zu bytes: %.*s\n", len, (int)len,
			  (const char *)in);
		interrupted = 1;
		break;

	case LWS_CALLBACK_CLIENT_WRITEABLE: {
		size_t msg_len = strlen(client_message);
		unsigned char buf[LWS_PRE + 256];
		int written;

		if (message_sent)
			break;

		if (msg_len > sizeof(buf) - LWS_PRE) {
			lwsl_err("WS Client: message too large\n");
			return -1;
		}

		memcpy(&buf[LWS_PRE], client_message, msg_len);
		written = lws_write(wsi, &buf[LWS_PRE], msg_len, LWS_WRITE_TEXT);
		if (written < 0 || (size_t)written != msg_len) {
			lwsl_err("WS Client: lws_write failed: %d\n", written);
			return -1;
		}

		message_sent = 1;
		lwsl_user("WS Client: sent: %s\n", client_message);
		break;
	}

	case LWS_CALLBACK_CLIENT_CLOSED:
	case LWS_CALLBACK_CLOSED:
		lwsl_user("WS Client: connection closed\n");
		interrupted = 1;
		break;

	case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
		lwsl_err("WS Client: connection error: %s\n",
			 in ? (const char *)in : "unknown");
		interrupted = 1;
		break;

	default:
		break;
	}

	return 0;
}

static struct lws_protocols protocols[] = {
	{
		.name = "echo-protocol",
		.callback = callback_ws,
		.rx_buffer_size = 4096,
	},
	{ 0 } /* terminator */
};

static void
signal_handler(int sig)
{
	(void)sig;
	interrupted = 1;
}

/*
 * Use a monotonic clock for the timeout deadline.  The wall clock (time())
 * can jump forward by years when NTP first syncs during the test, which would
 * make a wall-clock deadline appear instantly expired and abort the client
 * before the echo round-trip completes.
 */
static time_t
monotonic_secs(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec;
}

int main(int argc, char **argv)
{
	struct lws_context_creation_info info;
	struct lws_client_connect_info ci;
	struct lws_context *context;
	struct lws *wsi;
	time_t deadline;
	int timed_out = 0;
	int ret = 0;

	ret = parse_args(argc, argv);
	if (ret > 0)
		return 0;
	if (ret < 0) {
		print_usage(argv[0]);
		return 1;
	}

	lws_set_log_level(LLL_USER | LLL_ERR | LLL_WARN | LLL_INFO |
			  LLL_DEBUG, example_lws_log);

	memset(&info, 0, sizeof(info));
	info.protocols = protocols;
	info.options = LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
	info.port = CONTEXT_PORT_NO_LISTEN;
	info.timeout_secs = (unsigned int)timeout_secs;
	info.connect_timeout_secs = (unsigned int)timeout_secs;
	info.gid = -1;
	info.uid = -1;

	context = lws_create_context(&info);
	if (!context) {
		lwsl_err("WS Client: failed to create context\n");
		return 1;
	}

	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	memset(&ci, 0, sizeof(ci));
	ci.context = context;
	ci.address = server_address;
	ci.port = server_port;
	ci.path = server_path;
	ci.host = ci.address;
	ci.origin = ci.address;
	ci.protocol = protocols[0].name;

	wsi = lws_client_connect_via_info(&ci);
	if (!wsi) {
		lwsl_err("WS Client: failed to connect\n");
		lws_context_destroy(context);
		return 1;
	}

	deadline = monotonic_secs() + timeout_secs;
	lwsl_user("WS Client: connecting to ws://%s:%d%s timeout=%ds\n",
		  server_address, server_port, server_path, timeout_secs);

	while (!interrupted) {
		ret = lws_service(context, EXAMPLE_SERVICE_MS);
		if (ret < 0) {
			lwsl_err("WS Client: service error %d\n", ret);
			break;
		}

		if (monotonic_secs() >= deadline) {
			lwsl_err("WS Client: timeout waiting for %s\n",
				 connected ? "echo" : "connection");
			timed_out = 1;
			break;
		}
	}

	lws_context_destroy(context);

	return (ret < 0 || timed_out) ? 1 : 0;
}
