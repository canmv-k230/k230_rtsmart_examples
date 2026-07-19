#ifndef XIAOZHI_PROTOCOL_H
#define XIAOZHI_PROTOCOL_H

#include <stddef.h>

#include "xiaozhi_config.h"

struct xiaozhi_server_hello {
	char session_id[XIAOZHI_MAX_SESSION_ID];
	int sample_rate;
	int channels;
	int frame_duration;
};

int xiaozhi_build_hello(char *buffer, size_t buffer_size);

int xiaozhi_parse_server_hello(const char *data, size_t len,
			       struct xiaozhi_server_hello *hello);

int xiaozhi_build_listen_message(char *buffer, size_t buffer_size,
				 const char *session_id, const char *mode);

int xiaozhi_build_wake_detect_message(char *buffer, size_t buffer_size,
					 const char *session_id, const char *wake_word);

int xiaozhi_build_abort_message(char *buffer, size_t buffer_size,
				const char *session_id, const char *reason);

int xiaozhi_build_mcp_envelope(char *buffer, size_t buffer_size,
				       const char *session_id,
				       const char *payload,
				       size_t payload_len);

#endif
