#include "xiaozhi_transport.h"

#include <libwebsockets.h>

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum xiaozhi_tx_kind {
	XIAOZHI_TX_CONTROL,
	XIAOZHI_TX_AUDIO,
};

struct xiaozhi_tx_item {
	struct xiaozhi_tx_item *next;
	enum xiaozhi_tx_kind kind;
	size_t len;
	unsigned char data[];
};

struct xiaozhi_transport {
	char address[XIAOZHI_MAX_ADDRESS];
	char path[XIAOZHI_MAX_PATH];
	char auth_header[XIAOZHI_MAX_AUTH_HEADER];
	char device_id[XIAOZHI_MAX_DEVICE_ID];
	char client_id[XIAOZHI_MAX_CLIENT_ID];
	int port;
	int use_ssl;
	int allow_insecure;
	int timeout_secs;
	int log_level;

	struct xiaozhi_transport_events events;
	struct lws_context *context;
	struct lws *wsi;

	/*
	 * lws_service() and its callbacks run on the transport thread. Audio
	 * capture and other producers may enqueue messages concurrently. This
	 * mutex protects both the queues and the shared connection state; lws
	 * writes themselves remain on the transport thread.
	 */
	pthread_mutex_t tx_lock;
	struct xiaozhi_tx_item *control_head;
	struct xiaozhi_tx_item *control_tail;
	struct xiaozhi_tx_item *audio_head;
	struct xiaozhi_tx_item *audio_tail;
	int control_count;
	int audio_count;

	int connected;
	int ready;
	int attempt_finished;
	int failed;
	int stopping;
	int stop_requested;
	uint64_t connected_at_ms;
	uint64_t ready_at_ms;

	int rx_active;
	int rx_binary;
	size_t rx_len;
	unsigned char rx_buffer[XIAOZHI_MAX_RX_FRAME];
};

static uint64_t monotonic_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000ULL +
	       (uint64_t)ts.tv_nsec / 1000000ULL;
}

static int copy_string(char *dst, size_t dst_size, const char *src)
{
	int written;

	written = snprintf(dst, dst_size, "%s", src ? src : "");
	return written < 0 || (size_t)written >= dst_size ? -1 : 0;
}

static void free_queue_locked(struct xiaozhi_transport *transport)
{
	struct xiaozhi_tx_item *item;

	while (transport->control_head) {
		item = transport->control_head;
		transport->control_head = item->next;
		free(item);
	}
	while (transport->audio_head) {
		item = transport->audio_head;
		transport->audio_head = item->next;
		free(item);
	}
	transport->control_tail = NULL;
	transport->audio_tail = NULL;
	transport->control_count = 0;
	transport->audio_count = 0;
}

static void free_queue(struct xiaozhi_transport *transport)
{
	pthread_mutex_lock(&transport->tx_lock);
	free_queue_locked(transport);
	pthread_mutex_unlock(&transport->tx_lock);
}

static void get_transport_state(struct xiaozhi_transport *transport,
				int *connected, int *ready, int *attempt_finished,
				int *stop_requested, uint64_t *connected_at_ms,
				uint64_t *ready_at_ms)
{
	pthread_mutex_lock(&transport->tx_lock);
	*connected = transport->connected;
	*ready = transport->ready;
	*attempt_finished = transport->attempt_finished;
	*stop_requested = transport->stop_requested;
	*connected_at_ms = transport->connected_at_ms;
	*ready_at_ms = transport->ready_at_ms;
	pthread_mutex_unlock(&transport->tx_lock);
}

static void drop_oldest_audio_locked(struct xiaozhi_transport *transport)
{
	struct xiaozhi_tx_item *item = transport->audio_head;

	if (!item)
		return;
	transport->audio_head = item->next;
	if (!transport->audio_head)
		transport->audio_tail = NULL;
	transport->audio_count--;
	free(item);
}

static int queue_message(struct xiaozhi_transport *transport,
			 enum xiaozhi_tx_kind kind, const void *data, size_t len)
{
	struct xiaozhi_tx_item *item;
	struct lws *wsi;
	size_t allocation_size;

	if (!transport || !data || !len || len > XIAOZHI_MAX_JSON)
		return -1;

	allocation_size = sizeof(*item) + LWS_PRE + len;
	item = (struct xiaozhi_tx_item *)malloc(allocation_size);
	if (!item)
		return -1;

	memset(item, 0, sizeof(*item));
	item->kind = kind;
	item->len = len;
	memcpy(item->data + LWS_PRE, data, len);

	pthread_mutex_lock(&transport->tx_lock);
	if (!transport->connected || transport->stopping) {
		pthread_mutex_unlock(&transport->tx_lock);
		free(item);
		return -1;
	}
	if (kind == XIAOZHI_TX_AUDIO) {
		if (transport->audio_count >= XIAOZHI_TX_AUDIO_QUEUE_DEPTH)
			drop_oldest_audio_locked(transport);
		if (!transport->audio_tail)
			transport->audio_head = item;
		else
			transport->audio_tail->next = item;
		transport->audio_tail = item;
		transport->audio_count++;
	} else {
		if (transport->control_count >= XIAOZHI_TX_CONTROL_QUEUE_DEPTH) {
			pthread_mutex_unlock(&transport->tx_lock);
			free(item);
			return -1;
		}
		if (!transport->control_tail)
			transport->control_head = item;
		else
			transport->control_tail->next = item;
		transport->control_tail = item;
		transport->control_count++;
	}
	wsi = transport->wsi;
	if (wsi)
		lws_callback_on_writable(wsi);
	pthread_mutex_unlock(&transport->tx_lock);
	return 0;
}

static struct xiaozhi_tx_item *take_next_message(
	struct xiaozhi_transport *transport)
{
	struct xiaozhi_tx_item *item;

	pthread_mutex_lock(&transport->tx_lock);
	if (transport->control_head) {
		item = transport->control_head;
		transport->control_head = item->next;
		if (!transport->control_head)
			transport->control_tail = NULL;
		transport->control_count--;
	} else {
		item = transport->audio_head;
		if (item) {
			transport->audio_head = item->next;
			if (!transport->audio_head)
				transport->audio_tail = NULL;
			transport->audio_count--;
		}
	}
	pthread_mutex_unlock(&transport->tx_lock);
	return item;
}

static int has_messages(struct xiaozhi_transport *transport)
{
	int has;

	pthread_mutex_lock(&transport->tx_lock);
	has = transport->control_head || transport->audio_head;
	pthread_mutex_unlock(&transport->tx_lock);
	return has;
}

static struct xiaozhi_transport *transport_from_wsi(struct lws *wsi)
{
	if (!wsi)
		return NULL;
	return (struct xiaozhi_transport *)lws_get_opaque_user_data(wsi);
}

static int append_header(struct lws *wsi, const char *name, const char *value,
			 unsigned char **p, unsigned char *end)
{
	if (!value || !value[0])
		return 0;
	return lws_add_http_header_by_name(
		wsi, (const unsigned char *)name, (const unsigned char *)value,
		(int)strlen(value), p, end);
}

static int send_next_message(struct lws *wsi,
				 struct xiaozhi_transport *transport)
{
	struct xiaozhi_tx_item *item;
	enum lws_write_protocol protocol;
	int written;

	item = take_next_message(transport);
	if (!item)
		return 0;

	protocol = item->kind == XIAOZHI_TX_AUDIO ? LWS_WRITE_BINARY :
								 LWS_WRITE_TEXT;
	written = lws_write(wsi, item->data + LWS_PRE, item->len, protocol);
	if (written < 0 || (size_t)written < item->len) {
		free(item);
		return -1;
	}
	free(item);

	if (has_messages(transport))
		lws_callback_on_writable(wsi);
	return 0;
}

static void notify_error(struct xiaozhi_transport *transport,
			 const char *reason)
{
	pthread_mutex_lock(&transport->tx_lock);
	if (transport->failed) {
		pthread_mutex_unlock(&transport->tx_lock);
		return;
	}
	transport->failed = 1;
	transport->wsi = NULL;
	transport->connected = 0;
	transport->attempt_finished = 1;
	pthread_mutex_unlock(&transport->tx_lock);
	if (transport->events.on_error)
		transport->events.on_error(transport->events.opaque, reason);
}

static int callback_xiaozhi(struct lws *wsi, enum lws_callback_reasons reason,
				void *user, void *in, size_t len)
{
	struct xiaozhi_transport *transport;
	int final;
	int first;
	int ret;

	(void)user;
	if (reason == LWS_CALLBACK_PROTOCOL_INIT ||
	    reason == LWS_CALLBACK_PROTOCOL_DESTROY)
		return 0;

	transport = transport_from_wsi(wsi);
	if (!transport) {
		if (reason == LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER)
			return -1;
		return 0;
	}

	switch (reason) {
	case LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER: {
		unsigned char **p = (unsigned char **)in;
		unsigned char *end = *p + len;

		if (append_header(wsi, "Authorization:", transport->auth_header,
				  p, end) ||
		    append_header(wsi, "Protocol-Version:", "1", p, end) ||
		    append_header(wsi, "Device-Id:", transport->device_id, p, end) ||
		    append_header(wsi, "Client-Id:", transport->client_id, p, end))
			return -1;
		break;
	}

	case LWS_CALLBACK_CLIENT_ESTABLISHED:
		pthread_mutex_lock(&transport->tx_lock);
		transport->wsi = wsi;
		transport->connected = 1;
		transport->connected_at_ms = monotonic_ms();
		pthread_mutex_unlock(&transport->tx_lock);
		lwsl_user("xiaozhi: WebSocket connected to %s:%d%s%s\n",
			  transport->address, transport->port,
			  transport->use_ssl ? " (TLS)" : "", transport->path);
		if (transport->events.on_connected) {
			ret = transport->events.on_connected(transport->events.opaque);
			if (ret) {
				notify_error(transport, "connection setup failed");
				return -1;
			}
		}
		lws_callback_on_writable(wsi);
		break;

	case LWS_CALLBACK_CLIENT_WRITEABLE:
		if (send_next_message(wsi, transport)) {
			notify_error(transport, "WebSocket write failed");
			return -1;
		}
		break;

	case LWS_CALLBACK_CLIENT_RECEIVE:
		ret = 0;
		first = lws_is_first_fragment(wsi);
		final = lws_is_final_fragment(wsi);
		if (first) {
			transport->rx_active = 1;
			transport->rx_binary = lws_frame_is_binary(wsi) ? 1 : 0;
			transport->rx_len = 0;
		}
		if (!transport->rx_active ||
		    len > sizeof(transport->rx_buffer) - transport->rx_len) {
			notify_error(transport, "WebSocket frame exceeds receive buffer");
			return -1;
		}
		memcpy(transport->rx_buffer + transport->rx_len, in, len);
		transport->rx_len += len;
		if (final) {
			if (transport->rx_binary) {
				if (transport->events.on_binary)
					ret = transport->events.on_binary(
						transport->events.opaque,
						transport->rx_buffer, transport->rx_len);
			} else if (transport->events.on_text) {
				ret = transport->events.on_text(transport->events.opaque,
						(const char *)transport->rx_buffer,
						transport->rx_len);
			}
			if (ret) {
				notify_error(transport, "application rejected incoming message");
				return -1;
			}
			transport->rx_active = 0;
			transport->rx_binary = 0;
			transport->rx_len = 0;
		}
		break;

	case LWS_CALLBACK_CLIENT_CLOSED:
	case LWS_CALLBACK_CLOSED:
		{
			int stopping;

		pthread_mutex_lock(&transport->tx_lock);
		transport->wsi = NULL;
		transport->connected = 0;
		stopping = transport->stopping;
		if (!stopping)
			transport->attempt_finished = 1;
		pthread_mutex_unlock(&transport->tx_lock);
		if (!stopping && transport->events.on_closed)
			transport->events.on_closed(transport->events.opaque);
		}
		break;

	case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
		notify_error(transport,
			in ? (const char *)in : "WebSocket connection error");
		break;

	default:
		break;
	}

	return 0;
}

static struct lws_protocols protocols[] = {
	{
		.name = "xiaozhi-client",
		.callback = callback_xiaozhi,
		.rx_buffer_size = XIAOZHI_MAX_RX_FRAME,
	},
	{ 0 }
};

static void lws_log(int level, const char *line)
{
	(void)level;
	fputs(line, stderr);
}

static int lws_log_mask(int log_level)
{
	switch (log_level) {
	case XIAOZHI_LOG_LEVEL_ERROR:
		return LLL_ERR;
	case XIAOZHI_LOG_LEVEL_WARN:
		return LLL_ERR | LLL_WARN;
	case XIAOZHI_LOG_LEVEL_DEBUG:
		return LLL_ERR | LLL_WARN | LLL_NOTICE | LLL_INFO |
			LLL_DEBUG | LLL_PARSER | LLL_HEADER | LLL_EXT |
			LLL_CLIENT | LLL_LATENCY | LLL_USER | LLL_THREAD;
	case XIAOZHI_LOG_LEVEL_INFO:
	default:
		return LLL_USER | LLL_ERR | LLL_WARN | LLL_INFO;
	}
}

static void destroy_context(struct lws_context **context)
{
	int attempt;

	for (attempt = 0;
	     *context && attempt < XIAOZHI_CONTEXT_DESTROY_ATTEMPTS;
	     attempt++) {
		lws_context_destroy(*context);
		if (*context)
			usleep(1000);
	}
	if (*context)
		lwsl_err("xiaozhi: lws context destroy did not complete\n");
}

struct xiaozhi_transport *xiaozhi_transport_create(
	const struct xiaozhi_transport_options *options,
	const struct xiaozhi_transport_events *events)
{
	struct xiaozhi_transport *transport;

	if (!options || !options->address || !options->path)
		return NULL;
	transport = (struct xiaozhi_transport *)calloc(1, sizeof(*transport));
	if (!transport)
		return NULL;
	if (copy_string(transport->address, sizeof(transport->address),
			options->address) ||
	    copy_string(transport->path, sizeof(transport->path), options->path) ||
	    copy_string(transport->auth_header, sizeof(transport->auth_header),
			options->auth_header) ||
	    copy_string(transport->device_id, sizeof(transport->device_id),
			options->device_id) ||
	    copy_string(transport->client_id, sizeof(transport->client_id),
			options->client_id)) {
		free(transport);
		return NULL;
	}
	transport->port = options->port;
	transport->use_ssl = options->use_ssl;
	transport->allow_insecure = options->allow_insecure;
	transport->timeout_secs = options->timeout_secs;
	transport->log_level = options->log_level;
	if (events)
		transport->events = *events;
	if (pthread_mutex_init(&transport->tx_lock, NULL)) {
		free(transport);
		return NULL;
	}
	return transport;
}

void xiaozhi_transport_destroy(struct xiaozhi_transport *transport)
{
	if (!transport)
		return;
	free_queue(transport);
	pthread_mutex_destroy(&transport->tx_lock);
	free(transport);
}

int xiaozhi_transport_run(struct xiaozhi_transport *transport,
			  int duration_secs,
			  volatile sig_atomic_t *interrupted)
{
	struct lws_context_creation_info info;
	struct lws_client_connect_info connect_info;
	struct lws_context *context = NULL;
	struct lws *wsi;
	uint64_t deadline;
	int service_ret = 0;
	int completed = 0;
	int connected;
	int ready;
	int attempt_finished;
	int stop_requested;
	uint64_t connected_at_ms;
	uint64_t ready_at_ms;

	if (!transport)
		return 1;

	free_queue(transport);
	pthread_mutex_lock(&transport->tx_lock);
	transport->connected = 0;
	transport->ready = 0;
	transport->attempt_finished = 0;
	transport->failed = 0;
	transport->stopping = 0;
	transport->stop_requested = 0;
	transport->rx_active = 0;
	transport->rx_len = 0;
	pthread_mutex_unlock(&transport->tx_lock);

	lws_set_log_level(lws_log_mask(transport->log_level), lws_log);
	memset(&info, 0, sizeof(info));
	info.protocols = protocols;
	info.options = LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
	info.port = CONTEXT_PORT_NO_LISTEN;
	info.timeout_secs = (unsigned int)transport->timeout_secs;
	info.connect_timeout_secs = (unsigned int)transport->timeout_secs;
	info.pcontext = &context;
	info.gid = -1;
	info.uid = -1;

	context = lws_create_context(&info);
	if (!context) {
		lwsl_err("xiaozhi: failed to create lws context\n");
		return 1;
	}
	pthread_mutex_lock(&transport->tx_lock);
	transport->context = context;
	pthread_mutex_unlock(&transport->tx_lock);

	memset(&connect_info, 0, sizeof(connect_info));
	connect_info.context = context;
	connect_info.address = transport->address;
	connect_info.port = transport->port;
	connect_info.ssl_connection = transport->use_ssl ? LCCSCF_USE_SSL : 0;
	if (transport->use_ssl && transport->allow_insecure)
		connect_info.ssl_connection |= LCCSCF_ALLOW_INSECURE;
	connect_info.path = transport->path;
	connect_info.host = transport->address;
	connect_info.local_protocol_name = protocols[0].name;
	connect_info.opaque_user_data = transport;

	deadline = monotonic_ms() +
		(uint64_t)transport->timeout_secs * 1000ULL +
		XIAOZHI_CONNECT_TIMEOUT_GRACE_MS;
	wsi = lws_client_connect_via_info(&connect_info);
	if (!wsi) {
		lwsl_err("xiaozhi: failed to start connection\n");
		pthread_mutex_lock(&transport->tx_lock);
		transport->stopping = 1;
		transport->context = NULL;
		pthread_mutex_unlock(&transport->tx_lock);
		if (transport->events.on_stopping)
			transport->events.on_stopping(transport->events.opaque);
		destroy_context(&context);
		return 1;
	}

	lwsl_user("xiaozhi: connecting to %s://%s:%d%s\n",
		  transport->use_ssl ? "wss" : "ws", transport->address,
		  transport->port, transport->path);

	for (;;) {
		get_transport_state(transport, &connected, &ready,
				    &attempt_finished, &stop_requested,
				    &connected_at_ms, &ready_at_ms);
		if (attempt_finished || stop_requested ||
		    (interrupted && *interrupted))
			break;
		service_ret = lws_service(context, XIAOZHI_SERVICE_MS);
		if (service_ret < 0) {
			notify_error(transport, "lws service failed");
			break;
		}
		/* A close callback or SIGINT can finish the attempt during service. */
		get_transport_state(transport, &connected, &ready,
				    &attempt_finished, &stop_requested,
				    &connected_at_ms, &ready_at_ms);
		if (attempt_finished || stop_requested ||
		    (interrupted && *interrupted))
			break;

		if (!connected && monotonic_ms() >= deadline) {
			notify_error(transport, "connection timeout");
			break;
		}
		if (connected && !ready &&
		    monotonic_ms() >= connected_at_ms +
				  (uint64_t)transport->timeout_secs * 1000ULL) {
			notify_error(transport, "server hello timeout");
			break;
		}
		if (ready && duration_secs > 0 &&
		    monotonic_ms() >= ready_at_ms +
				  (uint64_t)duration_secs * 1000ULL) {
			completed = 1;
			break;
		}
	}

	pthread_mutex_lock(&transport->tx_lock);
	transport->stopping = 1;
	transport->wsi = NULL;
	transport->context = NULL;
	pthread_mutex_unlock(&transport->tx_lock);
	if (transport->events.on_stopping)
		transport->events.on_stopping(transport->events.opaque);
	destroy_context(&context);
	free_queue(transport);

	pthread_mutex_lock(&transport->tx_lock);
	stop_requested = transport->stop_requested;
	pthread_mutex_unlock(&transport->tx_lock);
	if ((interrupted && *interrupted) || completed || stop_requested)
		return 0;
	return 1;
}

int xiaozhi_transport_send_text(struct xiaozhi_transport *transport,
				const char *data, size_t len)
{
	return queue_message(transport, XIAOZHI_TX_CONTROL, data, len);
}

int xiaozhi_transport_send_binary(struct xiaozhi_transport *transport,
				  const void *data, size_t len)
{
	return queue_message(transport, XIAOZHI_TX_AUDIO, data, len);
}

void xiaozhi_transport_mark_ready(struct xiaozhi_transport *transport)
{
	if (!transport)
		return;
	pthread_mutex_lock(&transport->tx_lock);
	transport->ready = 1;
	transport->ready_at_ms = monotonic_ms();
	pthread_mutex_unlock(&transport->tx_lock);
}

void xiaozhi_transport_request_stop(struct xiaozhi_transport *transport)
{
	struct lws_context *context;

	if (!transport)
		return;
	pthread_mutex_lock(&transport->tx_lock);
	transport->stop_requested = 1;
	context = transport->context;
	/* Keep the context alive while asking the service loop to stop. */
	if (context)
		lws_cancel_service(context);
	pthread_mutex_unlock(&transport->tx_lock);
}

int xiaozhi_transport_is_connected(const struct xiaozhi_transport *transport)
{
	int connected;

	if (!transport)
		return 0;
	pthread_mutex_lock((pthread_mutex_t *)&transport->tx_lock);
	connected = transport->connected && !transport->stopping;
	pthread_mutex_unlock((pthread_mutex_t *)&transport->tx_lock);
	return connected;
}
