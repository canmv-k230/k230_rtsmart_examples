/*
 * libwebsockets WebSocket Server Example for RT-Smart
 *
 * Simple echo server that listens on a port and echoes back any text messages
 * it receives.
 */

#include <libwebsockets.h>
#include <string.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#define EXAMPLE_PORT 7681
#define EXAMPLE_RX_BUFFER_SIZE 4096

struct session_data {
	unsigned char buf[LWS_PRE + EXAMPLE_RX_BUFFER_SIZE];
	size_t len;
	int pending;
};

static int interrupted;
static int listen_port = EXAMPLE_PORT;

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
	printf("  %s [port]\n", prog);
	printf("  %s -p <port>\n", prog);
	printf("\nDefault: port=%d\n", EXAMPLE_PORT);
}

static int
parse_port(const char *arg, int *port)
{
	char *end = NULL;
	long value;

	value = strtol(arg, &end, 10);
	if (!arg[0] || (end && *end) || value <= 0 || value > 65535)
		return -1;

	*port = (int)value;
	return 0;
}

static int
parse_args(int argc, char **argv)
{
	int got_positional = 0;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-help") || !strcmp(argv[i], "--help")) {
			print_usage(argv[0]);
			return 1;
		} else if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--port")) {
			if (++i >= argc || parse_port(argv[i], &listen_port)) {
				printf("invalid port\n");
				return -1;
			}
		} else if (argv[i][0] == '-') {
			printf("unknown option: %s\n", argv[i]);
			return -1;
		} else if (!got_positional) {
			if (parse_port(argv[i], &listen_port)) {
				printf("invalid port: %s\n", argv[i]);
				return -1;
			}
			got_positional = 1;
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
	struct session_data *pss = (struct session_data *)user;

	switch (reason) {
	case LWS_CALLBACK_ESTABLISHED:
		lwsl_user("WS Server: client connected\n");
		break;

	case LWS_CALLBACK_RECEIVE:
		lwsl_user("WS Server: received %zu bytes: %.*s\n", len, (int)len,
			  (const char *)in);

		if (len > EXAMPLE_RX_BUFFER_SIZE) {
			lwsl_err("WS Server: message too large: %zu\n", len);
			return -1;
		}

		if (pss->pending) {
			lwsl_err("WS Server: previous echo still pending\n");
			return -1;
		}

		memcpy(&pss->buf[LWS_PRE], in, len);
		pss->len = len;
		pss->pending = 1;
		lws_callback_on_writable(wsi);
		break;

	case LWS_CALLBACK_SERVER_WRITEABLE:
		if (pss->pending) {
			int written = lws_write(wsi, &pss->buf[LWS_PRE], pss->len,
						LWS_WRITE_TEXT);

			if (written < 0 || (size_t)written != pss->len) {
				lwsl_err("WS Server: lws_write failed: %d\n", written);
				return -1;
			}

			pss->pending = 0;
		}
		break;

	case LWS_CALLBACK_CLOSED:
		lwsl_user("WS Server: client disconnected\n");
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
		.per_session_data_size = sizeof(struct session_data),
		.rx_buffer_size = EXAMPLE_RX_BUFFER_SIZE,
	},
	{ 0 } /* terminator */
};

static void
signal_handler(int sig)
{
	(void)sig;
	interrupted = 1;
}

int main(int argc, char **argv)
{
	struct lws_context_creation_info info;
	struct lws_context *context;
	int ret = 0;

	ret = parse_args(argc, argv);
	if (ret > 0)
		return 0;
	if (ret < 0) {
		print_usage(argv[0]);
		return 1;
	}

	lws_set_log_level(LLL_USER | LLL_ERR | LLL_WARN,
			  example_lws_log);

	memset(&info, 0, sizeof(info));
	info.port = listen_port;
	info.protocols = protocols;
	info.options = LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
	info.gid = -1;
	info.uid = -1;

	context = lws_create_context(&info);
	if (!context) {
		lwsl_err("WS Server: failed to create context\n");
		return 1;
	}

	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	lwsl_user("WS Server: listening on port %d\n", listen_port);

	while (!interrupted) {
		ret = lws_service(context, 1000);
		if (ret < 0) {
			lwsl_err("WS Server: lws_service error %d\n", ret);
			break;
		}
	}

	lwsl_user("WS Server: shutting down\n");
	lws_context_destroy(context);

	return 0;
}
