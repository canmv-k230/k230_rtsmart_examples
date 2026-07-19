#include "xiaozhi_activation.h"

#include <cJSON.h>
#include <libwebsockets.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define XIAOZHI_ACTIVATION_BODY_SIZE 2048

struct activation_session {
	unsigned char body[LWS_PRE + XIAOZHI_ACTIVATION_BODY_SIZE];
	size_t body_len;
	char response[XIAOZHI_MAX_RX_FRAME];
	size_t response_len;
	const struct xiaozhi_activation_options *options;
	struct xiaozhi_activation_result *result;
	int status;
	int body_pending;
	int completed;
	int request_result;
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

static int append_header(struct lws *wsi, const char *name, const char *value,
				 unsigned char **p, unsigned char *end)
{
	if (!value || !value[0])
		return 0;
	return lws_add_http_header_by_name(
		wsi, (const unsigned char *)name, (const unsigned char *)value,
		(int)strlen(value), p, end);
}

static int build_request_body(struct activation_session *session)
{
	cJSON *root = NULL;
	cJSON *application = NULL;
	cJSON *ota = NULL;
	cJSON *board = NULL;
	char *json = NULL;
	int ret = -1;

	root = cJSON_CreateObject();
	application = cJSON_CreateObject();
	ota = cJSON_CreateObject();
	board = cJSON_CreateObject();
	if (!root || !application || !ota || !board)
		goto cleanup;

	cJSON_AddStringToObject(root, "uuid",
				session->options->client_id ?
				session->options->client_id : "");
	cJSON_AddStringToObject(application, "name", "k230_rtos");
	cJSON_AddStringToObject(application, "version", "1.0.0");
	cJSON_AddItemToObject(root, "application", application);
	application = NULL;
	cJSON_AddItemToObject(root, "ota", ota);
	ota = NULL;
	cJSON_AddStringToObject(board, "type", "k230_rtos");
	cJSON_AddStringToObject(board, "name", "k230_rtos");
	cJSON_AddItemToObject(root, "board", board);
	board = NULL;

	json = cJSON_PrintUnformatted(root);
	if (!json || strlen(json) > XIAOZHI_ACTIVATION_BODY_SIZE)
		goto cleanup;
	memcpy(session->body + LWS_PRE, json, strlen(json));
	session->body_len = strlen(json);
	ret = 0;

cleanup:
	if (json)
		cJSON_free(json);
	if (application)
		cJSON_Delete(application);
	if (ota)
		cJSON_Delete(ota);
	if (board)
		cJSON_Delete(board);
	if (root)
		cJSON_Delete(root);
	return ret;
}

static void request_complete(struct lws *wsi, struct activation_session *session,
				     int result)
{
	if (session->completed)
		return;
	session->request_result = result;
	session->completed = 1;
	lws_cancel_service(lws_get_context(wsi));
}

static int callback_activation(struct lws *wsi,
				       enum lws_callback_reasons reason,
				       void *user, void *in, size_t len)
{
	struct activation_session *session;

	(void)user;
	session = (struct activation_session *)lws_get_opaque_user_data(wsi);
	if (!session)
		return 0;

	switch (reason) {
	case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
		request_complete(wsi, session, -1);
		return 0;

	case LWS_CALLBACK_ESTABLISHED_CLIENT_HTTP:
		session->status = (int)lws_http_client_http_response(wsi);
		break;

	case LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER: {
		unsigned char **p = (unsigned char **)in;
		unsigned char *end = *p + len;
		char content_length[24];

		snprintf(content_length, sizeof(content_length), "%zu",
			 session->body_len);
		if (append_header(wsi, "Activation-Version:", "1", p, end) ||
		    append_header(wsi, "Device-Id:", session->options->device_id,
				  p, end) ||
		    append_header(wsi, "Client-Id:", session->options->client_id,
				  p, end) ||
			append_header(wsi, "User-Agent:", "canaan",
					  p, end) ||
			append_header(wsi, "Accept-Language:", "zh-CN", p, end) ||
		    append_header(wsi, "Content-Type:", "application/json", p,
				  end) ||
		    append_header(wsi, "Content-Length:", content_length, p, end))
			return -1;
		if (!session->body_pending) {
			session->body_pending = 1;
			lws_client_http_body_pending(wsi, 1);
			lws_callback_on_writable(wsi);
		}
		break;
	}

	case LWS_CALLBACK_CLIENT_HTTP_WRITEABLE:
		if (session->body_pending) {
			if (lws_write(wsi, session->body + LWS_PRE, session->body_len,
				      LWS_WRITE_HTTP_FINAL) != (int)session->body_len)
				return -1;
			session->body_pending = 0;
			lws_client_http_body_pending(wsi, 0);
		}
		break;

	case LWS_CALLBACK_RECEIVE_CLIENT_HTTP: {
		unsigned char buffer[LWS_PRE + 1024];
		char *p = (char *)buffer + LWS_PRE;
		int available = (int)(sizeof(buffer) - LWS_PRE);

		if (lws_http_client_read(wsi, &p, &available) < 0)
			return -1;
		break;
	}

	case LWS_CALLBACK_RECEIVE_CLIENT_HTTP_READ:
		if (session->response_len + len >= sizeof(session->response))
			return -1;
		memcpy(session->response + session->response_len, in, len);
		session->response_len += len;
		session->response[session->response_len] = '\0';
		break;

	case LWS_CALLBACK_COMPLETED_CLIENT_HTTP:
		request_complete(wsi, session, session->status == 200 ? 0 : -1);
		break;

	case LWS_CALLBACK_CLOSED_CLIENT_HTTP:
		if (!session->completed)
			request_complete(wsi, session, -1);
		break;

	default:
		break;
	}

	return 0;
}

static const struct lws_protocols activation_protocols[] = {
	{
		.name = "xiaozhi-activation",
		.callback = callback_activation,
		.rx_buffer_size = 1024,
	},
	{ 0 }
};

static void activation_lws_log(int level, const char *line)
{
	(void)level;
	fputs(line, stderr);
}

static int activation_lws_log_mask(int log_level)
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
}

static int parse_result(struct activation_session *session)
{
	cJSON *root;
	cJSON *activation;
	cJSON *websocket;
	cJSON *item;
	int ret = 0;

	root = cJSON_ParseWithLength(session->response, session->response_len);
	if (!root)
		return -1;

	activation = cJSON_GetObjectItem(root, "activation");
	if (cJSON_IsObject(activation)) {
		item = cJSON_GetObjectItem(activation, "code");
		if (cJSON_IsString(item)) {
			if (copy_string(session->result->code,
					 sizeof(session->result->code), item->valuestring))
				ret = -1;
			else
				ret = 1;
		}
		item = cJSON_GetObjectItem(activation, "message");
		if (cJSON_IsString(item) &&
		    copy_string(session->result->message,
				 sizeof(session->result->message), item->valuestring))
			ret = -1;
	}

	websocket = cJSON_GetObjectItem(root, "websocket");
	if (cJSON_IsObject(websocket)) {
		item = cJSON_GetObjectItem(websocket, "token");
		if (cJSON_IsString(item) &&
		    copy_string(session->result->token,
				 sizeof(session->result->token), item->valuestring))
			ret = -1;
	}

	cJSON_Delete(root);
	return ret;
}

int xiaozhi_activation_request(
	const struct xiaozhi_activation_options *options,
	struct xiaozhi_activation_result *result)
{
	struct lws_context_creation_info info;
	struct lws_client_connect_info connect_info;
	struct activation_session *session;
	lws_parse_uri_t *uri;
	struct lws_context *context = NULL;
	struct lws *wsi;
	uint64_t deadline;
	int service_ret;
	int ret;
	int ssl;

	if (!options || !options->url || !options->url[0] || !result)
		return -1;
	memset(result, 0, sizeof(*result));

	uri = lws_parse_uri_create(options->url);
	if (!uri || !uri->scheme || !uri->host || !uri->path ||
	    (strcmp(uri->scheme, "http") && strcmp(uri->scheme, "https"))) {
		if (uri)
			lws_parse_uri_destroy(&uri);
		return -1;
	}

	session = (struct activation_session *)calloc(1, sizeof(*session));
	if (!session) {
		lws_parse_uri_destroy(&uri);
		return -1;
	}
	session->options = options;
	session->result = result;
	if (build_request_body(session)) {
		free(session);
		lws_parse_uri_destroy(&uri);
		return -1;
	}

	lws_set_log_level(activation_lws_log_mask(options->log_level),
			  activation_lws_log);
	memset(&info, 0, sizeof(info));
	info.protocols = activation_protocols;
	info.options = LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
	info.port = CONTEXT_PORT_NO_LISTEN;
	info.timeout_secs = (unsigned int)options->timeout_secs;
	info.connect_timeout_secs = (unsigned int)options->timeout_secs;
	info.pcontext = &context;
	info.gid = -1;
	info.uid = -1;
	context = lws_create_context(&info);
	if (!context) {
		free(session);
		lws_parse_uri_destroy(&uri);
		return -1;
	}

	ssl = !strcmp(uri->scheme, "https");
	memset(&connect_info, 0, sizeof(connect_info));
	connect_info.context = context;
	connect_info.address = uri->host;
	connect_info.port = uri->port ? uri->port : (ssl ? 443 : 80);
	connect_info.ssl_connection = ssl ? LCCSCF_USE_SSL : 0;
	if (ssl && options->allow_insecure)
		connect_info.ssl_connection |= LCCSCF_ALLOW_INSECURE;
	connect_info.path = uri->path[0] ? uri->path : "/";
	connect_info.host = uri->host;
	connect_info.origin = uri->host;
	connect_info.method = "POST";
	connect_info.local_protocol_name = activation_protocols[0].name;
	connect_info.opaque_user_data = session;
	wsi = lws_client_connect_via_info(&connect_info);
	if (!wsi) {
		destroy_context(&context);
		free(session);
		lws_parse_uri_destroy(&uri);
		return -1;
	}

	deadline = monotonic_ms() +
		(uint64_t)(options->timeout_secs > 0 ? options->timeout_secs : 10) *
		1000ULL + XIAOZHI_CONNECT_TIMEOUT_GRACE_MS;
	while (!session->completed && monotonic_ms() < deadline) {
		service_ret = lws_service(context, XIAOZHI_SERVICE_MS);
		if (service_ret < 0)
			break;
	}
	ret = session->completed ? session->request_result : -1;
	if (session->completed && session->status)
		printf("xiaozhi: activation HTTP status=%d\n", session->status);
	else if (!session->completed)
		printf("xiaozhi: activation request timed out\n");
	if (!ret && session->status == 200)
		ret = parse_result(session);
	else if (!ret)
		ret = -1;

	destroy_context(&context);
	free(session);
	lws_parse_uri_destroy(&uri);
	return ret;
}
