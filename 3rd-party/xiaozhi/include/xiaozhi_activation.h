#ifndef XIAOZHI_ACTIVATION_H
#define XIAOZHI_ACTIVATION_H

#include "xiaozhi_config.h"

struct xiaozhi_activation_options {
	const char *url;
	const char *device_id;
	const char *client_id;
	int timeout_secs;
	int allow_insecure;
	int log_level;
};

struct xiaozhi_activation_result {
	char token[XIAOZHI_MAX_TOKEN];
	char code[XIAOZHI_MAX_ACTIVATION_CODE];
	char message[XIAOZHI_MAX_ACTIVATION_MESSAGE];
};

/* Returns 0 when activated, 1 when user activation is required, or -1. */
int xiaozhi_activation_request(
	const struct xiaozhi_activation_options *options,
	struct xiaozhi_activation_result *result);

#endif
