#include "xiaozhi_protocol.h"

#include <cJSON.h>

#include <stdio.h>
#include <string.h>

static int copy_json_string(char *buffer, size_t buffer_size, cJSON *root)
{
	char *json;
	int written;

	json = cJSON_PrintUnformatted(root);
	if (!json)
		return -1;

	written = snprintf(buffer, buffer_size, "%s", json);
	cJSON_free(json);
	if (written < 0 || (size_t)written >= buffer_size)
		return -1;

	return 0;
}

int xiaozhi_build_hello(char *buffer, size_t buffer_size)
{
	cJSON *root = NULL;
	cJSON *features = NULL;
	cJSON *audio_params = NULL;
	int ret = -1;

	root = cJSON_CreateObject();
	features = cJSON_CreateObject();
	audio_params = cJSON_CreateObject();
	if (!root || !features || !audio_params)
		goto cleanup;

	cJSON_AddStringToObject(root, "type", "hello");
	cJSON_AddNumberToObject(root, "version", XIAOZHI_PROTOCOL_VERSION);
	cJSON_AddItemToObject(root, "features", features);
	features = NULL;
	cJSON_AddBoolToObject(cJSON_GetObjectItem(root, "features"), "mcp", 1);
	cJSON_AddStringToObject(root, "transport", "websocket");
	cJSON_AddStringToObject(audio_params, "format", "opus");
	cJSON_AddNumberToObject(audio_params, "sample_rate",
				XIAOZHI_AUDIO_SAMPLE_RATE);
	cJSON_AddNumberToObject(audio_params, "channels", XIAOZHI_AUDIO_CHANNELS);
	cJSON_AddNumberToObject(audio_params, "frame_duration",
				XIAOZHI_AUDIO_FRAME_DURATION_MS);
	cJSON_AddItemToObject(root, "audio_params", audio_params);
	audio_params = NULL;

	ret = copy_json_string(buffer, buffer_size, root);

cleanup:
	if (features)
		cJSON_Delete(features);
	if (audio_params)
		cJSON_Delete(audio_params);
	if (root)
		cJSON_Delete(root);
	return ret;
}

int xiaozhi_parse_server_hello(const char *data, size_t len,
			       struct xiaozhi_server_hello *hello)
{
	cJSON *root = NULL;
	cJSON *type;
	cJSON *transport;
	cJSON *session_id;
	cJSON *audio_params;
	cJSON *value;

	if (!data || !hello)
		return -1;

	memset(hello, 0, sizeof(*hello));
	hello->sample_rate = XIAOZHI_AUDIO_SERVER_SAMPLE_RATE;
	hello->channels = XIAOZHI_AUDIO_SERVER_CHANNELS;
	hello->frame_duration = XIAOZHI_AUDIO_FRAME_DURATION_MS;

	root = cJSON_ParseWithLength(data, len);
	if (!root)
		return -1;

	type = cJSON_GetObjectItem(root, "type");
	transport = cJSON_GetObjectItem(root, "transport");
	if (!cJSON_IsString(type) || strcmp(type->valuestring, "hello") ||
	    !cJSON_IsString(transport) ||
	    strcmp(transport->valuestring, "websocket")) {
		cJSON_Delete(root);
		return -1;
	}

	session_id = cJSON_GetObjectItem(root, "session_id");
	if (cJSON_IsString(session_id)) {
		if (snprintf(hello->session_id, sizeof(hello->session_id), "%s",
			     session_id->valuestring) >=
		    (int)sizeof(hello->session_id)) {
			cJSON_Delete(root);
			return -1;
		}
	}

	audio_params = cJSON_GetObjectItem(root, "audio_params");
	if (cJSON_IsObject(audio_params)) {
		value = cJSON_GetObjectItem(audio_params, "sample_rate");
		if (cJSON_IsNumber(value) && value->valueint > 0)
			hello->sample_rate = value->valueint;
		value = cJSON_GetObjectItem(audio_params, "channels");
		if (cJSON_IsNumber(value) && value->valueint > 0)
			hello->channels = value->valueint;
		value = cJSON_GetObjectItem(audio_params, "frame_duration");
		if (cJSON_IsNumber(value) && value->valueint > 0)
			hello->frame_duration = value->valueint;
	}

	cJSON_Delete(root);
	return 0;
}

static int build_session_message(char *buffer, size_t buffer_size,
				 const char *session_id, const char *type,
				 const char *field_name, const char *field_value,
				 const char *second_name, const char *second_value)
{
	cJSON *root = cJSON_CreateObject();
	int ret;

	if (!root)
		return -1;
	cJSON_AddStringToObject(root, "session_id", session_id ? session_id : "");
	cJSON_AddStringToObject(root, "type", type);
	if (field_name && field_value)
		cJSON_AddStringToObject(root, field_name, field_value);
	if (second_name && second_value)
		cJSON_AddStringToObject(root, second_name, second_value);
	ret = copy_json_string(buffer, buffer_size, root);
	cJSON_Delete(root);
	return ret;
}

int xiaozhi_build_listen_message(char *buffer, size_t buffer_size,
				 const char *session_id, const char *mode)
{
	return build_session_message(buffer, buffer_size, session_id, "listen",
				     "state", "start", "mode", mode);
}

int xiaozhi_build_wake_detect_message(char *buffer, size_t buffer_size,
					 const char *session_id, const char *wake_word)
{
	return build_session_message(buffer, buffer_size, session_id, "listen",
				     "state", "detect", "text", wake_word);
}

int xiaozhi_build_abort_message(char *buffer, size_t buffer_size,
				const char *session_id, const char *reason)
{
	return build_session_message(buffer, buffer_size, session_id, "abort",
				     "reason", reason, NULL, NULL);
}

int xiaozhi_build_mcp_envelope(char *buffer, size_t buffer_size,
				       const char *session_id,
				       const char *payload,
				       size_t payload_len)
{
	cJSON *root = NULL;
	cJSON *payload_root = NULL;
	int ret = -1;

	if (!payload || !payload_len)
		return -1;

	payload_root = cJSON_ParseWithLength(payload, payload_len);
	if (!payload_root || !cJSON_IsObject(payload_root))
		goto cleanup;

	root = cJSON_CreateObject();
	if (!root)
		goto cleanup;
	cJSON_AddStringToObject(root, "session_id", session_id ? session_id : "");
	cJSON_AddStringToObject(root, "type", "mcp");
	cJSON_AddItemToObject(root, "payload", payload_root);
	payload_root = NULL;
	ret = copy_json_string(buffer, buffer_size, root);

cleanup:
	if (payload_root)
		cJSON_Delete(payload_root);
	if (root)
		cJSON_Delete(root);
	return ret;
}
