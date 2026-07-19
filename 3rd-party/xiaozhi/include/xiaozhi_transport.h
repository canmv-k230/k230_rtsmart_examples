#ifndef XIAOZHI_TRANSPORT_H
#define XIAOZHI_TRANSPORT_H

#include <signal.h>
#include <stddef.h>

#include "xiaozhi_config.h"

struct xiaozhi_transport;

struct xiaozhi_transport_options {
	const char *address;
	const char *path;
	int port;
	int use_ssl;
	int allow_insecure;
	const char *auth_header;
	const char *device_id;
	const char *client_id;
	int timeout_secs;
	int log_level;
};

struct xiaozhi_transport_events {
	void *opaque;
	int (*on_connected)(void *opaque);
	int (*on_text)(void *opaque, const char *data, size_t len);
	int (*on_binary)(void *opaque, const unsigned char *data, size_t len);
	void (*on_closed)(void *opaque);
	void (*on_error)(void *opaque, const char *reason);
	void (*on_stopping)(void *opaque);
};

struct xiaozhi_transport *xiaozhi_transport_create(
	const struct xiaozhi_transport_options *options,
	const struct xiaozhi_transport_events *events);

void xiaozhi_transport_destroy(struct xiaozhi_transport *transport);

int xiaozhi_transport_run(struct xiaozhi_transport *transport,
			  int duration_secs,
			  volatile sig_atomic_t *interrupted);

int xiaozhi_transport_send_text(struct xiaozhi_transport *transport,
				const char *data, size_t len);

int xiaozhi_transport_send_binary(struct xiaozhi_transport *transport,
				  const void *data, size_t len);

void xiaozhi_transport_mark_ready(struct xiaozhi_transport *transport);
void xiaozhi_transport_request_stop(struct xiaozhi_transport *transport);
int xiaozhi_transport_is_connected(const struct xiaozhi_transport *transport);

#endif
